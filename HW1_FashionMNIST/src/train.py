import argparse
import json
from pathlib import Path

import numpy as np

from .data import CLASS_NAMES, load_fashion_mnist, make_train_valid_split
from .engine import evaluate, save_json, train_model
from .metrics import confusion_matrix
from .model import ThreeLayerMLP
from .visualize import plot_confusion_matrix, plot_misclassified, plot_training_curves, plot_weight_grid

BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay-step", type=int, default=4)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--valid-size", type=int, default=10000)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = load_fashion_mnist()
    train_set, valid_set = make_train_valid_split(train_x, train_y, valid_size=args.valid_size, seed=args.seed, train_limit=args.train_limit)
    if args.test_limit:
        test_x, test_y = test_x[:args.test_limit], test_y[:args.test_limit]
    model = ThreeLayerMLP(hidden_dim=args.hidden_dim, activation=args.activation, seed=args.seed)
    ckpt = BASE_DIR / "checkpoints" / "best_model.pkl"
    history = train_model(model, train_set, valid_set, ckpt, args.epochs, args.batch_size, args.lr, args.weight_decay, args.lr_decay_step, args.lr_decay_gamma, args.seed)
    best = ThreeLayerMLP.load(ckpt)
    train_metrics = evaluate(best, train_set, args.batch_size)
    valid_metrics = evaluate(best, valid_set, args.batch_size)
    test_metrics = evaluate(best, (test_x, test_y), args.batch_size)
    preds = np.argmax(test_metrics["logits"], axis=1)
    matrix = confusion_matrix(preds, test_y)
    fig_dir = BASE_DIR / "figures"
    plot_training_curves(history, fig_dir)
    plot_confusion_matrix(matrix, fig_dir / "confusion_matrix.png")
    plot_weight_grid(best.first_layer_weights(), fig_dir / "first_layer_weights.png")
    plot_misclassified(test_x, test_y, preds, fig_dir / "misclassified_examples.png")
    payload = {"config": vars(args), "class_names": CLASS_NAMES, "checkpoint": str(ckpt), "history": history, "train": {k: v for k, v in train_metrics.items() if k != "logits"}, "valid": {k: v for k, v in valid_metrics.items() if k != "logits"}, "test": {k: v for k, v in test_metrics.items() if k != "logits"}, "confusion_matrix": matrix.tolist()}
    save_json(BASE_DIR / "results" / "metrics.json", payload)
    print(json.dumps({"valid_accuracy": valid_metrics["accuracy"], "test_accuracy": test_metrics["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
