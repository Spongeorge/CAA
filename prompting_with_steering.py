"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b
"""

import json
from llama_wrapper import LlamaWrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def process_item_ab(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }

def process_item_open_ended(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question = item["question"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=100
    )
    return {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_tqa_mmlu(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = model.get_logits_from_text(
        user_input=prompt, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        "truthful_qa": process_item_tqa_mmlu,
        "mmlu": process_item_tqa_mmlu,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        "open_ended": get_open_ended_test_data(settings.behavior),
        "truthful_qa": get_truthful_qa_data(),
        "mmlu": get_mmlu_data(),
    }
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        model_name=settings.model_name,
        override_model_weights_path=settings.override_model_weights_path,
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]
    for layer in layers:
        name_path = model.model_name
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        if settings.override_vector is not None:
            vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
        else:
            vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True)
        #if settings.model_size != "7b":
            #vector = vector.half()
        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector
                )
                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)

def test_steering_ts(
    layers: List[int],
    multiplier: float,
    settings: SteeringSettings,
    overwrite=False,
    reward_fn=None,
):
    """
    Thompson-sampling based layer selection:
    - Runs through test_data once.
    - For each example, chooses a layer via Thompson sampling.
    - reward_fn(result) must return a numeric reward in [0,1].
    """

    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        "truthful_qa": process_item_tqa_mmlu,
        "mmlu": process_item_tqa_mmlu,
    }

    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        "open_ended": get_open_ended_test_data(settings.behavior),
        "truthful_qa": get_truthful_qa_data(),
        "mmlu": get_mmlu_data(),
    }

    test_data = test_datasets[settings.type]

    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        model_name=settings.model_name,
        override_model_weights_path=settings.override_model_weights_path,
    )

    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)

    # Bandit parameters
    num_layers = len(layers)
    alpha = np.ones(num_layers)
    beta_val = np.ones(num_layers)

    results = []

    # iterate through dataset ONCE
    for item in tqdm(test_data, desc="TS steering loop"):

        # Thompson sampling: draw a parameter for each layer
        theta = np.random.beta(alpha, beta_val)
        idx = np.argmax(theta)
        chosen_layer = layers[idx]

        # load steering vector for chosen_layer
        name_path = settings.override_vector_model or model.model_name
        if settings.override_vector is not None:
            vector = get_steering_vector(
                settings.behavior, settings.override_vector, name_path, normalized=True
            )
        else:
            vector = get_steering_vector(
                settings.behavior, chosen_layer, name_path, normalized=True
            )
        vector = vector.to(model.device)

        # Run one evaluation on the item using the chosen layer
        model.reset_all()
        model.set_add_activations(chosen_layer, multiplier * vector)

        result = process_methods[settings.type](
            item=item,
            model=model,
            system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
            a_token_id=a_token_id,
            b_token_id=b_token_id,
        )

        # Convert result to scalar reward
        r = reward_fn(result)
        assert 0 <= r <= 1

        # Update posterior parameters
        alpha[idx] += r
        beta_val[idx] += (1 - r)

        # Save result with layer choice
        results.append({
            "item_id": getattr(item, "id", None),
            "chosen_layer": int(chosen_layer),
            "reward": float(r),
            "result": result,
        })

    # save summary + posteriors
    summary = {
        "layers": layers,
        "alpha": alpha.tolist(),
        "beta": beta_val.tolist(),
        "posterior_means": (alpha / (alpha + beta_val)).tolist(),
    }

    save_filename = os.path.join(
        save_results_dir,
        f"ts_results_{settings.behavior}_{settings.type}.json",
    )

    with open(save_filename, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=4)

    print("Saved:", save_filename)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--ts", type=bool, default=False)

    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.model_name = args.model_name
    steering_settings.override_model_weights_path = args.override_model_weights_path

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        if not args.ts:
            test_steering(
                layers=args.layers,
                multipliers=args.multipliers,
                settings=steering_settings,
                overwrite=args.overwrite,
            )
        else:
            test_steering_ts(
                layers=args.layers,
                multipliers=args.multipliers,
                settings=steering_settings,
                overwrite=args.overwrite,
            )
