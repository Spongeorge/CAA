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

def evaluate_final_layers(
    final_layers,
    test_data,
    model,
    vector_cache,
    multiplier,
    settings,
    process_methods,
    a_token_id,
    b_token_id
):
    """
    Evaluate the final selected layers across the entire test set.
    Returns: average_reward, list_of_results
    """

    results = []
    successes = 0.0

    for item in test_data:
        model.reset_all()

        # Inject steering at all chosen layers
        for L in final_layers:
            model.set_add_activations(L, multiplier * vector_cache[L])

        result = process_methods[settings.type](
            item=item,
            model=model,
            system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
            a_token_id=a_token_id,
            b_token_id=b_token_id,
        )

        # Compute reward same way as CMAB
        if settings.type == "ab":
          denom = result["a_prob"] + result["b_prob"]
          if denom == 0:
              reward = 0.0
          else:
              match_val = result["answer_matching_behavior"]
              if "A" in match_val:
                  reward = result["a_prob"] / denom
              elif "B" in match_val:
                  reward = result["b_prob"] / denom
              else:
                  reward = 0.0
        else:
            reward = 0.0 # raise not implemented exception

        successes += reward
        results.append({
            "item": item,
            "model_output": result,
            "reward": reward
        })

    avg_reward = successes / len(test_data)
    return avg_reward, results


def test_steering_cmab(
    layers: List[int],
    K: int,
    multiplier: float,
    settings: SteeringSettings,
    overwrite=False,
    reward_fn=None,
):
    """
    CMAB-SM based layer selection for activation steering.

    - Each layer is treated as an arm.
    - Each evaluation item is a "round".
    - The bandit algorithm selects K layers jointly (an action).
    - Reward comes from model output on that sample.

    Inputs:
        layers: candidate layers (arms)
        K: number of layers to pick jointly
        multiplier: steering multiplier for vector injection
        settings: SteeringSettings
        reward_fn: function(result_dict) -> reward in [0,1]
    """

    # ---------------------------
    # Setup
    # ---------------------------
    save_results_dir = get_results_dir(settings.behavior)
    os.makedirs(save_results_dir, exist_ok=True)

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

    # ---------------------------
    # Load model
    # ---------------------------
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        model_name=settings.model_name,
        override_model_weights_path=settings.override_model_weights_path,
    )

    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)

    # ---------------------------
    # Preload vectors for all layers
    # ---------------------------
    vector_cache = {}
    name_path = settings.override_vector_model or model.model_name

    for layer in layers:
        if settings.override_vector is not None:
            vec = get_steering_vector(
                settings.behavior, settings.override_vector, name_path, normalized=True
            )
        else:
            vec = get_steering_vector(
                settings.behavior, layer, name_path, normalized=True
            )
        vector_cache[layer] = vec.to(model.device)

    # ---------------------------
    # Reward evaluation function (1 round = 1 sample)
    # ---------------------------
    def eval_action(chosen_layers):
        # pick random example
        item = test_data[np.random.randint(0, len(test_data))]

        # reset
        model.reset_all()

        # inject at all chosen layers
        for L in chosen_layers:
            model.set_add_activations(L, multiplier * vector_cache[L])

        # run
        result = process_methods[settings.type](
            item=item,
            model=model,
            system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
            a_token_id=a_token_id,
            b_token_id=b_token_id,
        )

        return reward_fn(result), result, item

    # ---------------------------
    # CMAB-SM MAIN LOOP
    # ---------------------------
    N = len(layers)
    T = len(test_data)  # one CMAB round per sample

    # lambda parameter
    lam = (256 * N * np.log(2 * N * T) / T) ** (1 / 3)

    # Partition arms into groups K+1
    arms = list(range(N))
    groups = [arms[i:i + K + 1] for i in range(0, N, K + 1)]
    if len(groups[-1]) < K + 1:
        need = K + 1 - len(groups[-1])
        groups[-1].extend(arms[:need])

    # Track results
    log = []

    # ---- SORT function
    def SORT(G):
        L = K + 1
        mu = np.zeros(L)
        count = np.zeros(L, dtype=int)

        # each action = selection of K arms out of these K+1
        actions = []
        for i in range(L):
            # drop arm i
            actions.append([layers[G[j]] for j in range(L) if j != i])

        r = 1
        Delta_r = 0.5 ** r

        def n_required(d):
            return int(np.ceil(2 * np.log(T * N * K) / (d * d)))

        n_r = n_required(Delta_r)

        unsorted = set(range(L))

        while Delta_r > lam and unsorted:
            for idx in list(unsorted):
                needed = n_r - count[idx]
                if needed > 0:
                    rewards = []
                    for _ in range(needed):
                        rwd, res, item = eval_action(actions[idx])
                        rewards.append(rwd)
                        log.append({"group": G, "action": actions[idx], "reward": rwd})
                    rewards = np.array(rewards)
                    mu[idx] = (mu[idx] * count[idx] + rewards.sum()) / (count[idx] + needed)
                    count[idx] += needed

            # now attempt sorting
            order = np.argsort(mu)
            placed = []
            for idx in order:
                if idx not in unsorted:
                    continue
                can_place = True
                for j in order:
                    if j == idx or j not in unsorted:
                        continue
                    # overlapping intervals?
                    if not (mu[j] - 2 * Delta_r > mu[idx] or mu[idx] > mu[j] + 2 * Delta_r):
                        can_place = False
                        break
                if can_place:
                    placed.append(idx)
            for idx in placed:
                unsorted.remove(idx)

            r += 1
            Delta_r = 0.5 ** r
            n_r = n_required(Delta_r)

        # final ordering
        order = np.argsort(mu)
        topK = order[-K:]
        return [G[i] for i in topK]

    # ---- MERGE function
    def MERGE(G1, G2):
        i = j = 0
        out = []

        def compare(i, j):
            base = [layers[x] for x in G1]
            rep = [layers[G2[j]] if idx == i else layers[x] for idx, x in enumerate(G1)]

            # sample until separation
            rb = rr = 0
            cb = cr = 0
            r1 = 1
            Delta = 0.5 ** r1

            def n_required(d):
                return int(np.ceil(2 * np.log(T * N * K) / (d * d)))

            n1 = n_required(Delta)

            while True:
                needb = n1 - cb
                if needb > 0:
                    rewards = []
                    for _ in range(needb):
                        rwd, res, item = eval_action(base)
                        rewards.append(rwd)
                        log.append({"merge_base": base, "reward": rwd})
                    rb = (rb * cb + np.sum(rewards)) / (cb + needb)
                    cb += needb

                needr = n1 - cr
                if needr > 0:
                    rewards = []
                    for _ in range(needr):
                        rwd, res, item = eval_action(rep)
                        rewards.append(rwd)
                        log.append({"merge_rep": rep, "reward": rwd})
                    rr = (rr * cr + np.sum(rewards)) / (cr + needr)
                    cr += needr

                if rb + 2 * Delta < rr or rr + 2 * Delta < rb:
                    break

                r1 += 1
                Delta = 0.5 ** r1
                n1 = n_required(Delta)
                if Delta <= lam:
                    break

            return rb, rr

        while len(out) < K and i < K and j < K:
            rb, rr = compare(i, j)
            if rb < rr:
                out.append(G2[j])
                j += 1
            else:
                out.append(G1[i])
                i += 1

        while len(out) < K and i < K:
            out.append(G1[i])
            i += 1
        while len(out) < K and j < K:
            out.append(G2[j])
            j += 1

        return out

    # ---------------------------
    # Run CMAB-SM
    # ---------------------------
    G_star = SORT(groups[0])
    for g in tqdm(groups[1:]):
        G_sorted = SORT(g)
        G_star = MERGE(G_star, G_sorted)

    best_layers = [layers[i] for i in G_star]

    # Save
        # ---------------------------
    # After selecting best layers → evaluate them fully
    # ---------------------------
    final_score, final_eval = evaluate_final_layers(
        final_layers = best_layers,
        test_data   = test_data,
        model       = model,
        vector_cache= vector_cache,
        multiplier  = multiplier,
        settings    = settings,
        process_methods = process_methods,
        a_token_id  = a_token_id,
        b_token_id  = b_token_id
    )

    print("\n==============================")
    print(" CMAB-SM Final Evaluation ")
    print("==============================")
    print("Selected layers:", best_layers)
    print(f"Final average reward: {final_score:.4f}")
    print("==============================\n")

    # Save everything
    save_path = os.path.join(
        save_results_dir,
        f"cmab_results_{settings.behavior}_{settings.type}.json",
    )

    with open(save_path, "w") as f:
        json.dump({
            "best_layers": best_layers,
            "cmab_log": log,
            "final_score": final_score,
            "final_eval": final_eval
        }, f, indent=4)

    print("Saved results to:", save_path)
    return best_layers, final_score



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
    parser.add_argument("--cmab", type=bool, default=False)

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
        if args.cmab:
            def default_reward_fn(result):
              # Normalize
              denom = result["a_prob"] + result["b_prob"]
              if denom == 0:
                  return 0.0

              match_val = result["answer_matching_behavior"]

              if "A" in match_val:
                  return result["a_prob"] / denom
              elif "B" in match_val:
                  return result["b_prob"] / denom
              return 0.0


            K = 3

            # use the first multiplier, or you can modify to run multiple
            multiplier = args.multipliers[0]

            test_steering_cmab(
                layers=args.layers,
                K=K,
                multiplier=multiplier,
                settings=steering_settings,
                overwrite=args.overwrite,
                reward_fn=default_reward_fn,
            )
        elif not args.ts:
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
