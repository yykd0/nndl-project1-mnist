import argparse
from pathlib import Path

import numpy as np

from .data import CLASS_NAMES, load_fashion_mnist
from .engine import evaluate
from .metrics import confusion_matrix
from .model import ThreeLayerMLP

BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(BASE_DIR / "checkpoints" / "best_model.pkl"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    _, _, test_x, test_y = load_fashion_mnist()
    model = ThreeLayerMLP.load(args.model)
    metrics = evaluate(model, (test_x, test_y), batch_size=args.batch_size)
    preds = np.argmax(metrics["logits"], axis=1)
    matrix = confusion_matrix(preds, test_y)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Rows=true labels, columns=predicted labels")
    print("Labels:", ", ".join(f"{i}:{name}" for i, name in enumerate(CLASS_NAMES)))
    print(matrix)


if __name__ == "__main__":
    main()
