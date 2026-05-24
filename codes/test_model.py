import argparse
import gzip
import pickle
from pathlib import Path
from struct import unpack

import numpy as np

import mynn as nn


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "MNIST"


def load_test_set():
    with gzip.open(DATA_DIR / "t10k-images-idx3-ubyte.gz", "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(DATA_DIR / "t10k-labels-idx1-ubyte.gz", "rb") as f:
        _, num = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)
    return test_imgs.astype(np.float32) / 255.0, test_labs.astype(np.int64)


def load_model(model_path):
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and payload.get("model_type") == "CNN":
        model = nn.models.Model_CNN()
    else:
        model = nn.models.Model_MLP()
    model.load_model(model_path)
    return model


def evaluate(model, test_imgs, test_labs, batch_size=256):
    runner = nn.runner.RunnerM(
        model,
        nn.optimizer.SGD(0.0, model),
        nn.metric.accuracy,
        nn.op.MultiCrossEntropyLoss(model=model, max_classes=10),
    )
    return runner.evaluate([test_imgs, test_labs], batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="../../submission/checkpoints/cnn_momentum/best_model.pickle",
        help="Path to a saved model pickle.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        cwd_candidate = (Path.cwd() / model_path).resolve()
        if cwd_candidate.exists():
            model_path = cwd_candidate
        else:
            model_path = (BASE_DIR / model_path).resolve()
    test_imgs, test_labs = load_test_set()
    model = load_model(model_path)
    accuracy, loss = evaluate(model, test_imgs, test_labs, batch_size=args.batch_size)
    print(f"model={model_path}")
    print(f"test_accuracy={accuracy:.6f}")
    print(f"test_loss={loss:.6f}")
