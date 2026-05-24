import argparse
import pickle
from pathlib import Path

import mynn as nn
from run_experiments import FIGURE_DIR, draw_kernel_grid, draw_mlp_weights


BASE_DIR = Path(__file__).resolve().parent


def load_model(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and payload.get("model_type") == "CNN":
        model = nn.models.Model_CNN()
    else:
        model = nn.models.Model_MLP()
    model.load_model(path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (BASE_DIR / model_path).resolve()
    model = load_model(model_path)
    FIGURE_DIR.mkdir(exist_ok=True)

    optimizable = [layer for layer in model.layers if layer.optimizable]
    if isinstance(model, nn.models.Model_CNN):
        output = Path(args.output) if args.output else FIGURE_DIR / "cnn_kernels.png"
        draw_kernel_grid(optimizable[0].params["W"], output)
    else:
        output = Path(args.output) if args.output else FIGURE_DIR / "mlp_first_layer_weights.png"
        draw_mlp_weights(optimizable[0].params["W"], output)
    print(f"saved={output}")
