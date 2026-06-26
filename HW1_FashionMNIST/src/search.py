import argparse
import itertools
from pathlib import Path

from .data import load_fashion_mnist, make_train_valid_split
from .engine import evaluate, save_json, train_model
from .model import ThreeLayerMLP

BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_x, train_y, _, _ = load_fashion_mnist()
    train_limit = 12000 if args.quick else 30000
    valid_size = 3000 if args.quick else 8000
    train_set, valid_set = make_train_valid_split(train_x, train_y, valid_size=valid_size, seed=args.seed, train_limit=train_limit)
    grid = {"lr": [0.05, 0.08, 0.12], "hidden_dim": [64, 128], "weight_decay": [0.0, 1e-4, 5e-4], "activation": ["relu", "tanh"]}
    results = []
    for lr, hidden_dim, weight_decay, activation in itertools.product(grid["lr"], grid["hidden_dim"], grid["weight_decay"], grid["activation"]):
        name = f"h{hidden_dim}_{activation}_lr{lr}_wd{weight_decay}"
        print(f"\n=== {name} ===")
        model = ThreeLayerMLP(hidden_dim=hidden_dim, activation=activation, seed=args.seed)
        ckpt = BASE_DIR / "checkpoints" / "search" / f"{name}.pkl"
        history = train_model(model, train_set, valid_set, ckpt, epochs=args.epochs, batch_size=128, lr=lr, weight_decay=weight_decay, lr_decay_step=max(2, args.epochs // 2), lr_decay_gamma=0.5, seed=args.seed)
        best = ThreeLayerMLP.load(ckpt)
        metrics = evaluate(best, valid_set, batch_size=256)
        results.append({"name": name, "lr": lr, "hidden_dim": hidden_dim, "weight_decay": weight_decay, "activation": activation, "best_valid_accuracy": metrics["accuracy"], "best_valid_loss": metrics["loss"], "checkpoint": str(ckpt), "history": history})
        save_json(BASE_DIR / "results" / "hyperparameter_search.json", {"results": results})
    results.sort(key=lambda item: item["best_valid_accuracy"], reverse=True)
    save_json(BASE_DIR / "results" / "hyperparameter_search.json", {"results": results})
    print("Best configuration:")
    print(results[0])


if __name__ == "__main__":
    main()
