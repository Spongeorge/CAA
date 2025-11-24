from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import torch as t
import os


def normalize_vectors(model_name: str, n_layers: int):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in ALL_BEHAVIORS:
            try:
                vec_path = get_vector_path(behavior, layer, model_name)
                vec = t.load(vec_path)
                norm = vec.norm().item()
                vecs[behavior] = vec
                norms[behavior] = norm
                new_path = vec_path.replace("vectors", "normalized_vectors")
                new_paths[behavior] = new_path
            except:
                pass  # TODO: fix this
        print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in ALL_BEHAVIORS:
            try:
                vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
            except:
                pass  # TODO: fix this
        # save the normalized vectors
        for behavior in ALL_BEHAVIORS:
            try:
                if not os.path.exists(os.path.dirname(new_paths[behavior])):
                    os.makedirs(os.path.dirname(new_paths[behavior]))
                t.save(vecs[behavior], new_paths[behavior])
            except:
                pass  # TODO: fix this


if __name__ == "__main__":
    normalize_vectors("meta-llama/Llama-3.2-1B-Instruct", 16)