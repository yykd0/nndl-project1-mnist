import argparse
import gzip
import json
import os
import pickle
import time
from pathlib import Path
from struct import unpack

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mynn as nn


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "MNIST"
SUBMISSION_DIR = BASE_DIR.parent.parent / "submission"
FIGURE_DIR = SUBMISSION_DIR / "figures"
CHECKPOINT_DIR = SUBMISSION_DIR / "checkpoints"
RESULTS_PATH = SUBMISSION_DIR / "experiment_results.json"


def load_mnist():
    with gzip.open(DATA_DIR / "train-images-idx3-ubyte.gz", "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(DATA_DIR / "train-labels-idx1-ubyte.gz", "rb") as f:
        _, num = unpack(">2I", f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(DATA_DIR / "t10k-images-idx3-ubyte.gz", "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(DATA_DIR / "t10k-labels-idx1-ubyte.gz", "rb") as f:
        _, num = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    train_imgs = train_imgs.astype(np.float32) / 255.0
    test_imgs = test_imgs.astype(np.float32) / 255.0
    return train_imgs, train_labs.astype(np.int64), test_imgs, test_labs.astype(np.int64)


def make_split(train_imgs, train_labs, seed=309, quick=False):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(train_imgs.shape[0])
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    if quick:
        valid_size = 512
        train_size = 2048
    else:
        valid_size = 10000
        train_size = train_imgs.shape[0] - valid_size
    valid_imgs = train_imgs[:valid_size]
    valid_labs = train_labs[:valid_size]
    train_imgs = train_imgs[valid_size:valid_size + train_size]
    train_labs = train_labs[valid_size:valid_size + train_size]
    return (train_imgs, train_labs), (valid_imgs, valid_labs)


def ensure_dirs():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)


def make_optimizer(name, model, lr):
    if name == "sgd":
        return nn.optimizer.SGD(init_lr=lr, model=model)
    if name == "momentum":
        return nn.optimizer.MomentGD(init_lr=lr, model=model, mu=0.9)
    raise ValueError(name)


def make_scheduler(name, optimizer, milestones=None, gamma=0.5):
    if name == "none":
        return None
    if name == "multistep":
        return nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
    if name == "exponential":
        return nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    raise ValueError(name)


def run_one_experiment(config, train_set, valid_set, test_set):
    np.random.seed(config["seed"])
    model = config["model_factory"]()
    optimizer = make_optimizer(config["optimizer"], model, config["lr"])
    scheduler = make_scheduler(
        config["scheduler"],
        optimizer,
        milestones=config.get("milestones", []),
        gamma=config.get("gamma", 0.5),
    )
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=config["batch_size"],
        scheduler=scheduler,
    )

    save_dir = CHECKPOINT_DIR / config["name"]
    start_time = time.time()
    runner.train(
        train_set,
        valid_set,
        num_epochs=config["epochs"],
        log_iters=config["log_iters"],
        validation_freq=config["validation_freq"],
        eval_batch_size=config["eval_batch_size"],
        save_dir=str(save_dir),
    )
    train_time = time.time() - start_time

    best_path = save_dir / "best_model.pickle"
    if best_path.exists():
        model.load_model(best_path)

    eval_runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        nn.op.MultiCrossEntropyLoss(model=model, max_classes=10),
        batch_size=config["batch_size"],
    )
    train_eval_size = min(10000, train_set[0].shape[0])
    train_score, train_loss = eval_runner.evaluate(
        [train_set[0][:train_eval_size], train_set[1][:train_eval_size]],
        batch_size=config["eval_batch_size"],
    )
    valid_score, valid_loss = eval_runner.evaluate(valid_set, batch_size=config["eval_batch_size"])
    test_score, test_loss = eval_runner.evaluate(test_set, batch_size=config["eval_batch_size"])

    return {
        "name": config["name"],
        "description": config["description"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "scheduler": config["scheduler"],
        "train_time_seconds": train_time,
        "checkpoint": str(best_path),
        "train_subset_accuracy": float(train_score),
        "train_subset_loss": float(train_loss),
        "valid_accuracy": float(valid_score),
        "valid_loss": float(valid_loss),
        "test_accuracy": float(test_score),
        "test_loss": float(test_loss),
        "history": {
            "train_steps": [int(x) for x in runner.train_steps],
            "train_loss": [float(x) for x in runner.train_loss],
            "train_scores": [float(x) for x in runner.train_scores],
            "dev_steps": [int(x) for x in runner.dev_steps],
            "dev_loss": [float(x) for x in runner.dev_loss],
            "dev_scores": [float(x) for x in runner.dev_scores],
        },
    }, model


def get_font(size=18, bold=False):
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_line_chart(series, output_path, title, y_label, width=900, height=520):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    title_font = get_font(24, bold=True)
    small_font = get_font(13)
    left, right, top, bottom = 80, width - 30, 60, height - 70
    draw.text((left, 20), title, fill="#111111", font=title_font)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)

    all_x = []
    all_y = []
    for item in series:
        if len(item["x"]) > 0:
            all_x.extend(item["x"])
            all_y.extend(item["y"])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if y_max == y_min:
        y_max += 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    for i in range(5):
        y = bottom - i * (bottom - top) / 4
        value = y_min + i * (y_max - y_min) / 4
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((8, y - 8), f"{value:.3f}", fill="#555555", font=small_font)
    draw.text((left, height - 36), "训练步数", fill="#333333", font=font)
    draw.text((8, 36), y_label, fill="#333333", font=font)

    def project(x, y):
        if x_max == x_min:
            px = left
        else:
            px = left + (x - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        return px, py

    legend_x = right - 220
    legend_y = top + 8
    for idx, item in enumerate(series):
        points = [project(x, y) for x, y in zip(item["x"], item["y"])]
        if len(points) >= 2:
            draw.line(points, fill=item["color"], width=3)
        for px, py in points[::max(1, len(points) // 80)]:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=item["color"])
        ly = legend_y + idx * 24
        draw.line((legend_x, ly + 8, legend_x + 30, ly + 8), fill=item["color"], width=3)
        draw.text((legend_x + 38, ly), item["label"], fill="#333333", font=small_font)

    img.save(output_path)


def draw_confusion_matrix(matrix, output_path):
    cell = 48
    margin = 100
    size = margin + cell * 10 + 40
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    title_font = get_font(22, bold=True)
    draw.text((margin, 22), "混淆矩阵（CNN + Momentum，测试集）", fill="#111111", font=title_font)
    max_value = max(1, int(matrix.max()))
    for i in range(10):
        draw.text((margin - 36, margin + i * cell + 16), str(i), fill="#333333", font=font)
        draw.text((margin + i * cell + 18, margin - 34), str(i), fill="#333333", font=font)
        for j in range(10):
            value = int(matrix[i, j])
            intensity = int(255 - 190 * value / max_value)
            fill = (intensity, intensity + 10 if intensity < 245 else 255, 255)
            x0 = margin + j * cell
            y0 = margin + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline="#FFFFFF")
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=font)
            tx = x0 + (cell - (bbox[2] - bbox[0])) / 2
            ty = y0 + (cell - (bbox[3] - bbox[1])) / 2
            draw.text((tx, ty), text, fill="#111111", font=font)
    draw.text((margin + 180, size - 28), "预测标签", fill="#333333", font=font)
    draw.text((14, margin + 190), "真实标签", fill="#333333", font=font)
    img.save(output_path)


def draw_image_grid(images, titles, output_path, columns=5, scale=5):
    cell_w, cell_h = 28 * scale + 18, 28 * scale + 44
    rows = int(np.ceil(len(images) / columns))
    img = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(13)
    for idx, image in enumerate(images):
        r, c = divmod(idx, columns)
        x = c * cell_w + 9
        y = r * cell_h + 9
        tile = Image.fromarray(np.uint8(np.clip(image.reshape(28, 28) * 255, 0, 255)), mode="L")
        tile = tile.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST).convert("RGB")
        img.paste(tile, (x, y))
        draw.rectangle((x, y, x + 28 * scale, y + 28 * scale), outline="#333333")
        draw.text((x, y + 28 * scale + 6), titles[idx], fill="#111111", font=font)
    img.save(output_path)


def draw_kernel_grid(weights, output_path, scale=18):
    kernels = weights[:, 0, :, :]
    columns = min(8, kernels.shape[0])
    rows = int(np.ceil(kernels.shape[0] / columns))
    k = kernels.shape[-1]
    cell = k * scale + 24
    img = Image.new("RGB", (columns * cell, rows * cell + 40), "white")
    draw = ImageDraw.Draw(img)
    draw.text((12, 10), "CNN 第一层卷积核可视化", fill="#111111", font=get_font(20, bold=True))
    for idx, kernel in enumerate(kernels):
        r, c = divmod(idx, columns)
        x = c * cell + 12
        y = r * cell + 44
        norm = kernel - kernel.min()
        if norm.max() > 0:
            norm = norm / norm.max()
        tile = Image.fromarray(np.uint8(norm * 255), mode="L")
        tile = tile.resize((k * scale, k * scale), Image.Resampling.NEAREST).convert("RGB")
        img.paste(tile, (x, y))
        draw.rectangle((x, y, x + k * scale, y + k * scale), outline="#333333")
    img.save(output_path)


def draw_mlp_weights(weights, output_path, count=16, scale=4):
    hidden = weights.shape[1]
    count = min(count, hidden)
    columns = 4
    rows = int(np.ceil(count / columns))
    cell = 28 * scale + 22
    img = Image.new("RGB", (columns * cell, rows * cell + 40), "white")
    draw = ImageDraw.Draw(img)
    draw.text((12, 10), "MLP 第一层权重可视化", fill="#111111", font=get_font(20, bold=True))
    for idx in range(count):
        vec = weights[:, idx].reshape(28, 28)
        norm = vec - vec.min()
        if norm.max() > 0:
            norm = norm / norm.max()
        tile = Image.fromarray(np.uint8(norm * 255), mode="L")
        tile = tile.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST).convert("RGB")
        r, c = divmod(idx, columns)
        x = c * cell + 11
        y = r * cell + 44
        img.paste(tile, (x, y))
        draw.rectangle((x, y, x + 28 * scale, y + 28 * scale), outline="#333333")
    img.save(output_path)


def confusion_matrix(preds, labels):
    matrix = np.zeros((10, 10), dtype=np.int64)
    for true, pred in zip(labels, preds):
        matrix[int(true), int(pred)] += 1
    return matrix


def create_figures(results, trained_models, test_set):
    acc_series = []
    loss_series = []
    colors = ["#2563EB", "#059669", "#DC2626", "#7C3AED"]
    for idx, result in enumerate(results):
        history = result["history"]
        color = colors[idx % len(colors)]
        if history["dev_steps"]:
            acc_series.append({
                "x": history["dev_steps"],
                "y": history["dev_scores"],
                "label": result["name"],
                "color": color,
            })
        loss_series.append({
            "x": history["train_steps"],
            "y": history["train_loss"],
            "label": result["name"],
            "color": color,
        })
    draw_line_chart(acc_series, FIGURE_DIR / "validation_accuracy.png", "验证集准确率", "准确率")
    draw_line_chart(loss_series, FIGURE_DIR / "training_loss.png", "训练损失", "损失")

    best_model = trained_models["cnn_momentum"]
    eval_runner = nn.runner.RunnerM(
        best_model,
        nn.optimizer.SGD(0.0, best_model),
        nn.metric.accuracy,
        nn.op.MultiCrossEntropyLoss(best_model, 10),
    )
    logits = eval_runner.predict(test_set[0], batch_size=256)
    preds = np.argmax(logits, axis=1)
    matrix = confusion_matrix(preds, test_set[1])
    draw_confusion_matrix(matrix, FIGURE_DIR / "confusion_matrix.png")

    wrong_idx = np.where(preds != test_set[1])[0][:20]
    if wrong_idx.shape[0] > 0:
        titles = [f"真:{test_set[1][i]} 预:{preds[i]}" for i in wrong_idx]
        draw_image_grid(test_set[0][wrong_idx], titles, FIGURE_DIR / "misclassified_examples.png")

    cnn_conv = [layer for layer in best_model.layers if layer.optimizable][0]
    draw_kernel_grid(cnn_conv.params["W"], FIGURE_DIR / "cnn_kernels.png")

    mlp_model = trained_models["mlp_sgd"]
    mlp_first = [layer for layer in mlp_model.layers if layer.optimizable][0]
    draw_mlp_weights(mlp_first.params["W"], FIGURE_DIR / "mlp_first_layer_weights.png")

    return {
        "validation_accuracy": str(FIGURE_DIR / "validation_accuracy.png"),
        "training_loss": str(FIGURE_DIR / "training_loss.png"),
        "confusion_matrix": str(FIGURE_DIR / "confusion_matrix.png"),
        "misclassified_examples": str(FIGURE_DIR / "misclassified_examples.png"),
        "cnn_kernels": str(FIGURE_DIR / "cnn_kernels.png"),
        "mlp_first_layer_weights": str(FIGURE_DIR / "mlp_first_layer_weights.png"),
    }


def build_configs(quick=False):
    if quick:
        return [
            {
                "name": "mlp_sgd",
                "description": "MLP baseline trained with vanilla SGD.",
                "model_factory": lambda: nn.models.Model_MLP([784, 64, 10], "ReLU"),
                "optimizer": "sgd",
                "scheduler": "none",
                "lr": 0.08,
                "epochs": 1,
                "batch_size": 128,
                "validation_freq": 10,
                "eval_batch_size": 256,
                "log_iters": 5,
                "seed": 309,
            },
            {
                "name": "cnn_sgd",
                "description": "Simple CNN trained with vanilla SGD.",
                "model_factory": lambda: nn.models.Model_CNN(conv_channels=4, kernel_size=5, hidden_dim=32),
                "optimizer": "sgd",
                "scheduler": "none",
                "lr": 0.03,
                "epochs": 1,
                "batch_size": 64,
                "validation_freq": 10,
                "eval_batch_size": 256,
                "log_iters": 5,
                "seed": 310,
            },
            {
                "name": "cnn_momentum",
                "description": "CNN trained with momentum and a multistep learning-rate schedule.",
                "model_factory": lambda: nn.models.Model_CNN(conv_channels=4, kernel_size=5, hidden_dim=32),
                "optimizer": "momentum",
                "scheduler": "multistep",
                "lr": 0.025,
                "milestones": [20],
                "gamma": 0.5,
                "epochs": 1,
                "batch_size": 64,
                "validation_freq": 10,
                "eval_batch_size": 256,
                "log_iters": 5,
                "seed": 311,
            },
        ]
    return [
        {
            "name": "mlp_sgd",
            "description": "MLP baseline trained with vanilla SGD.",
            "model_factory": lambda: nn.models.Model_MLP([784, 256, 10], "ReLU"),
            "optimizer": "sgd",
            "scheduler": "none",
            "lr": 0.08,
            "epochs": 5,
            "batch_size": 128,
            "validation_freq": 391,
            "eval_batch_size": 256,
            "log_iters": 100,
            "seed": 309,
        },
        {
            "name": "cnn_sgd",
            "description": "Simple CNN trained with vanilla SGD.",
            "model_factory": lambda: nn.models.Model_CNN(conv_channels=8, kernel_size=5, hidden_dim=64),
            "optimizer": "sgd",
            "scheduler": "none",
            "lr": 0.03,
            "epochs": 3,
            "batch_size": 64,
            "validation_freq": 782,
            "eval_batch_size": 256,
            "log_iters": 150,
            "seed": 310,
        },
        {
            "name": "cnn_momentum",
            "description": "CNN trained with momentum and a multistep learning-rate schedule.",
            "model_factory": lambda: nn.models.Model_CNN(conv_channels=8, kernel_size=5, hidden_dim=64),
            "optimizer": "momentum",
            "scheduler": "multistep",
            "lr": 0.025,
            "milestones": [782, 1564],
            "gamma": 0.5,
            "epochs": 3,
            "batch_size": 64,
            "validation_freq": 782,
            "eval_batch_size": 256,
            "log_iters": 150,
            "seed": 311,
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test experiment.")
    args = parser.parse_args()

    ensure_dirs()
    train_imgs, train_labs, test_imgs, test_labs = load_mnist()
    train_set, valid_set = make_split(train_imgs, train_labs, quick=args.quick)
    if args.quick:
        test_set = [test_imgs[:512], test_labs[:512]]
    else:
        test_set = [test_imgs, test_labs]

    results = []
    trained_models = {}
    for config in build_configs(quick=args.quick):
        print(f"\n=== Running {config['name']} ===")
        result, model = run_one_experiment(config, train_set, valid_set, test_set)
        results.append(result)
        trained_models[config["name"]] = model
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"quick": args.quick, "results": results}, f, indent=2)

    figures = create_figures(results, trained_models, test_set)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"quick": args.quick, "results": results, "figures": figures}, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
