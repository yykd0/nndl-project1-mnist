import json
import pickle
from pathlib import Path

import mynn as nn
from run_experiments import RESULTS_PATH, create_figures, load_mnist


def load_saved_model(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and payload.get("model_type") == "CNN":
        model = nn.models.Model_CNN()
    else:
        model = nn.models.Model_MLP()
    model.load_model(path)
    return model


if __name__ == "__main__":
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload["results"]
    _, _, test_imgs, test_labs = load_mnist()
    trained_models = {
        item["name"]: load_saved_model(Path(item["checkpoint"]))
        for item in results
    }
    figures = create_figures(results, trained_models, [test_imgs, test_labs])
    payload["figures"] = figures
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("figures regenerated")
