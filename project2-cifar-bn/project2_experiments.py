"""Reproducible experiments for Project 2.

The script trains compact CIFAR-10 classifiers, compares VGG-A with and without
BatchNorm, records loss/gradient statistics, saves figures, and exports metrics
used by the final report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_DATA_DIR = PROJECT_DIR / "data"
DATA_DIR = PROJECT_DIR / "safe_data"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPORTS_DIR / "models"
RESULTS_DIR = REPORTS_DIR / "results"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MIN_BYTES = 100_000_000


@dataclass
class ExperimentConfig:
    name: str
    base_channels: int
    activation: str
    loss: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    use_batch_norm: bool = True
    dropout: float = 0.20


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_random_seeds(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def ensure_cifar_extracted(root: Path = DATA_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-10-batches-py"
    if (extracted / "data_batch_1").exists() and (extracted / "test_batch").exists():
        return

    archive_candidates = [
        root / "cifar-10-python.tar.gz",
        PROJECT_DIR / "runtime_data" / "cifar-10-python.tar.gz",
        PACKAGE_DATA_DIR / "cifar-10-python.tar.gz",
    ]
    archive = next((candidate for candidate in archive_candidates if candidate.exists() and candidate.stat().st_size > CIFAR_MIN_BYTES), None)
    if archive is None:
        archive = root / "cifar-10-python.tar.gz"
        urllib.request.urlretrieve(CIFAR_URL, archive)

    with tarfile.open(archive, "r:gz") as tar:
        root_resolved = root.resolve()
        for member in tar:
            target = root / member.name
            if not target.resolve().is_relative_to(root_resolved):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as handle:
                    handle.write(source.read())


def make_cifar_loaders(
    train_items: int = 6000,
    test_items: int = 1000,
    batch_size: int = 128,
    num_workers: int = 0,
    augment: bool = True,
    seed: int = 2026,
) -> tuple[DataLoader, DataLoader]:
    ensure_cifar_extracted(DATA_DIR)
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ] if augment else []
    train_transform = transforms.Compose(train_steps + [transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=train_transform)
    test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=test_transform)

    generator = torch.Generator().manual_seed(seed)
    if 0 < train_items < len(train_set):
        indices = torch.randperm(len(train_set), generator=generator)[:train_items].tolist()
        train_set = Subset(train_set, indices)
    if 0 < test_items < len(test_set):
        indices = torch.randperm(len(test_set), generator=generator)[:test_items].tolist()
        test_set = Subset(test_set, indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def activation_layer(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class CompactCIFARNet(nn.Module):
    """Compact CNN satisfying the required CIFAR-10 components."""

    def __init__(
        self,
        base_channels: int = 24,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(activation_layer(activation))
            blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(activation_layer(activation))
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if dropout > 0:
                blocks.append(nn.Dropout2d(p=dropout / 2))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 4 * 4, 256),
            activation_layer(activation),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_criterion(name: str) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    raise ValueError(f"Unknown loss: {name}")


def make_optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_total += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return loss_total / max(total, 1), correct / max(total, 1)


def last_linear_gradient(model: nn.Module) -> torch.Tensor | None:
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear) and module.weight.grad is not None:
            return module.weight.grad.detach().flatten().cpu()
    return None


def train_model(
    model: nn.Module,
    name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_path: Path,
) -> dict:
    model.to(device)
    history = {
        "name": name,
        "epochs": [],
        "step_losses": [],
        "grad_cosine": [],
        "grad_diff": [],
        "best_accuracy": 0.0,
        "best_epoch": 0,
    }
    previous_grad = None

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"{name} epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            grad = last_linear_gradient(model)
            if grad is not None:
                if previous_grad is not None and grad.numel() == previous_grad.numel():
                    cosine = torch.nn.functional.cosine_similarity(grad, previous_grad, dim=0).item()
                    diff = torch.norm(grad - previous_grad, p=2).item()
                else:
                    cosine = math.nan
                    diff = math.nan
                history["grad_cosine"].append(cosine)
                history["grad_diff"].append(diff)
                previous_grad = grad

            optimizer.step()
            history["step_losses"].append(loss.item())
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": val_loss,
                "test_accuracy": val_acc,
            }
        )
        if val_acc >= history["best_accuracy"]:
            history["best_accuracy"] = val_acc
            history["best_epoch"] = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "model_name": name,
                },
                model_path,
            )

    return history


def make_compact_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("compact_relu_16_adam_ce", 16, "relu", "cross_entropy", "adam", 1e-3, 1e-4),
        ExperimentConfig("compact_relu_32_adam_ce", 32, "relu", "cross_entropy", "adam", 1e-3, 1e-4),
        ExperimentConfig("compact_gelu_24_adam_ce", 24, "gelu", "cross_entropy", "adam", 1e-3, 1e-4),
        ExperimentConfig("compact_silu_24_adam_smooth", 24, "silu", "label_smoothing", "adam", 1e-3, 1e-4),
        ExperimentConfig("compact_relu_24_sgd_ce", 24, "relu", "cross_entropy", "sgd_momentum", 5e-2, 5e-4),
    ]


def compact_summary_row(config: ExperimentConfig, model: nn.Module, history: dict) -> dict:
    last_epoch = history["epochs"][-1]
    return {
        "name": config.name,
        "base_channels": config.base_channels,
        "activation": config.activation,
        "loss": config.loss,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "parameters": get_number_of_parameters(model),
        "best_test_accuracy": history["best_accuracy"],
        "best_test_error": 1.0 - history["best_accuracy"],
        "best_epoch": history["best_epoch"],
        "final_train_accuracy": last_epoch["train_accuracy"],
        "final_test_accuracy": last_epoch["test_accuracy"],
        "final_train_loss": last_epoch["train_loss"],
        "final_test_loss": last_epoch["test_loss"],
    }


def plot_compact_histories(histories: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for history in histories:
        epochs = [item["epoch"] for item in history["epochs"]]
        axes[0].plot(epochs, [item["test_accuracy"] for item in history["epochs"]], marker="o", label=history["name"])
        axes[1].plot(epochs, [item["train_loss"] for item in history["epochs"]], marker="o", label=history["name"])
    axes[0].set_title("CIFAR-10 test accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "compact_cifar_training.png", dpi=180)
    plt.close(fig)


def plot_filter_visualization(model: CompactCIFARNet) -> None:
    first_conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    weights = first_conv.weight.detach().cpu()
    count = min(16, weights.size(0))
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for ax, filt in zip(axes.flat, weights[:count]):
        img = filt.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.axis("off")
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.suptitle("First-layer filters of the best compact model", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "compact_first_layer_filters.png", dpi=180)
    plt.close(fig)


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_compact_cifar_sweep(
    epochs: int = 2,
    train_items: int = 6000,
    test_items: int = 1000,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 2026,
) -> tuple[list[dict], list[dict]]:
    set_random_seeds(seed)
    train_loader, test_loader = make_cifar_loaders(
        train_items=train_items,
        test_items=test_items,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=True,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    histories = []
    rows = []
    best = {"accuracy": -1.0, "path": None, "config": None, "model": None}

    for config in make_compact_configs():
        set_random_seeds(seed)
        model = CompactCIFARNet(
            base_channels=config.base_channels,
            activation=config.activation,
            use_batch_norm=config.use_batch_norm,
            dropout=config.dropout,
        )
        criterion = make_criterion(config.loss)
        optimizer = make_optimizer(config, model)
        model_path = MODELS_DIR / f"{config.name}.pt"
        history = train_model(model, config.name, train_loader, test_loader, criterion, optimizer, device, epochs, model_path)
        history["config"] = asdict(config)
        histories.append(history)
        row = compact_summary_row(config, model, history)
        rows.append(row)
        if row["best_test_accuracy"] > best["accuracy"]:
            best = {"accuracy": row["best_test_accuracy"], "path": model_path, "config": config, "model": model}

    save_csv(RESULTS_DIR / "compact_cifar_results.csv", rows)
    plot_compact_histories(histories)
    if best["path"] is not None:
        shutil.copyfile(best["path"], MODELS_DIR / "best_compact_cifar.pt")
    if isinstance(best["model"], CompactCIFARNet):
        plot_filter_visualization(best["model"])
    with (RESULTS_DIR / "compact_histories.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    return rows, histories


def envelope(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(curve) for curve in curves if curve)
    aligned = np.array([curve[:min_len] for curve in curves if curve], dtype=float)
    return np.nanmin(aligned, axis=0), np.nanmax(aligned, axis=0)


def finite_mean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    return float(arr.mean())


def finite_max(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    return float(arr.max())


def plot_bn_landscape(losses: dict[str, list[list[float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"VGG_A": "#3B82F6", "VGG_A_BatchNorm": "#F97316"}
    labels = {"VGG_A": "VGG-A without BN", "VGG_A_BatchNorm": "VGG-A with BN"}
    for name, curves in losses.items():
        lo, hi = envelope(curves)
        steps = np.arange(len(lo))
        ax.plot(steps, lo, color=colors[name], linewidth=1.5)
        ax.plot(steps, hi, color=colors[name], linewidth=1.5)
        ax.fill_between(steps, lo, hi, color=colors[name], alpha=0.18, label=labels[name])
    ax.set_title("Loss landscape envelope across learning rates")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bn_loss_landscape.png", dpi=180)
    plt.close(fig)


def plot_bn_metric_bars(rows: list[dict]) -> None:
    aggregate = {}
    for name in ["VGG_A", "VGG_A_BatchNorm"]:
        subset = [row for row in rows if row["model"] == name]
        aggregate[name] = {
            "accuracy": finite_mean([row["test_accuracy"] for row in subset]),
            "grad_cosine": finite_mean([row["mean_grad_cosine"] for row in subset]),
            "grad_diff": finite_mean([row["max_grad_diff"] for row in subset]),
        }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    labels = ["Without BN", "With BN"]
    names = ["VGG_A", "VGG_A_BatchNorm"]
    for ax, metric, title in [
        (axes[0], "accuracy", "Test accuracy"),
        (axes[1], "grad_cosine", "Gradient predictiveness"),
        (axes[2], "grad_diff", "Max gradient difference"),
    ]:
        ax.bar(labels, [aggregate[name][metric] for name in names], color=["#3B82F6", "#F97316"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bn_gradient_metrics.png", dpi=180)
    plt.close(fig)


def run_batchnorm_landscape(
    epochs: int = 1,
    train_items: int = 1024,
    test_items: int = 512,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 2026,
    learning_rates: tuple[float, ...] = (1e-3, 2e-3, 5e-4, 1e-4),
) -> tuple[list[dict], dict[str, list[dict]]]:
    ensure_dirs()
    set_random_seeds(seed)
    train_loader, test_loader = make_cifar_loaders(
        train_items=train_items,
        test_items=test_items,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    histories: dict[str, list[dict]] = {"VGG_A": [], "VGG_A_BatchNorm": []}
    losses: dict[str, list[list[float]]] = {"VGG_A": [], "VGG_A_BatchNorm": []}
    model_factories = {
        "VGG_A": VGG_A,
        "VGG_A_BatchNorm": VGG_A_BatchNorm,
    }

    for model_name, factory in model_factories.items():
        for lr in learning_rates:
            set_random_seeds(seed)
            model = factory()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            run_name = f"{model_name}_lr_{lr:g}"
            model_path = MODELS_DIR / f"{run_name}.pt"
            history = train_model(model, run_name, train_loader, test_loader, criterion, optimizer, device, epochs, model_path)
            history["learning_rate"] = lr
            history["model"] = model_name
            histories[model_name].append(history)
            losses[model_name].append(history["step_losses"])
            last_epoch = history["epochs"][-1]
            rows.append(
                {
                    "model": model_name,
                    "learning_rate": lr,
                    "parameters": get_number_of_parameters(model),
                    "train_loss": last_epoch["train_loss"],
                    "test_loss": last_epoch["test_loss"],
                    "test_accuracy": last_epoch["test_accuracy"],
                    "mean_grad_cosine": finite_mean(history["grad_cosine"]),
                    "max_grad_diff": finite_max(history["grad_diff"]),
                }
            )

    save_csv(RESULTS_DIR / "bn_landscape_results.csv", rows)
    plot_bn_landscape(losses)
    plot_bn_metric_bars(rows)
    with (RESULTS_DIR / "bn_histories.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    return rows, histories


def run_all(args: argparse.Namespace) -> dict:
    ensure_dirs()
    compact_rows, _ = run_compact_cifar_sweep(
        epochs=args.epochs,
        train_items=args.train_items,
        test_items=args.test_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    bn_rows, _ = run_batchnorm_landscape(
        epochs=args.bn_epochs,
        train_items=args.bn_train_items,
        test_items=args.bn_test_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    summary = {
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "compact_setup": {
            "epochs": args.epochs,
            "train_items": args.train_items,
            "test_items": args.test_items,
            "batch_size": args.batch_size,
        },
        "bn_setup": {
            "epochs": args.bn_epochs,
            "train_items": args.bn_train_items,
            "test_items": args.bn_test_items,
            "batch_size": args.batch_size,
            "learning_rates": [1e-3, 2e-3, 5e-4, 1e-4],
        },
        "compact_results": compact_rows,
        "bn_results": bn_rows,
        "figures": [
            str(FIGURES_DIR / "compact_cifar_training.png"),
            str(FIGURES_DIR / "compact_first_layer_filters.png"),
            str(FIGURES_DIR / "bn_loss_landscape.png"),
            str(FIGURES_DIR / "bn_gradient_metrics.png"),
        ],
        "best_model": str(MODELS_DIR / "best_compact_cifar.pt"),
    }
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project 2 CIFAR-10 and BatchNorm experiments.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-items", type=int, default=6000)
    parser.add_argument("--test-items", type=int, default=1000)
    parser.add_argument("--bn-epochs", type=int, default=1)
    parser.add_argument("--bn-train-items", type=int, default=1024)
    parser.add_argument("--bn-test-items", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
