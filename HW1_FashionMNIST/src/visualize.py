from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import CLASS_NAMES


def _font(size=16, bold=False):
    for path in ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def line_chart(series, output_path, title, y_label):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 900, 520
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 82, w - 34, 62, h - 72
    draw.text((left, 20), title, fill="#111111", font=_font(24, True))
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    xs = [v for item in series for v in item["x"]]
    ys = [v for item in series for v in item["y"]]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if abs(y_max - y_min) < 1e-9:
        y_max += 1.0
    pad = (y_max - y_min) * 0.08
    y_min, y_max = y_min - pad, y_max + pad
    for i in range(5):
        y = bottom - i * (bottom - top) / 4
        value = y_min + i * (y_max - y_min) / 4
        draw.line((left, y, right, y), fill="#E5E7EB")
        draw.text((8, y - 8), f"{value:.3f}", fill="#555555", font=_font(13))
    draw.text((left, h - 38), "Epoch", fill="#333333", font=_font(15))
    draw.text((10, 38), y_label, fill="#333333", font=_font(15))
    def project(x, y):
        px = left if x_max == x_min else left + (x - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        return px, py
    for idx, item in enumerate(series):
        points = [project(x, y) for x, y in zip(item["x"], item["y"])]
        if len(points) >= 2:
            draw.line(points, fill=item["color"], width=3)
        lx, ly = right - 230, top + 12 + idx * 28
        draw.line((lx, ly + 8, lx + 32, ly + 8), fill=item["color"], width=3)
        draw.text((lx + 42, ly), item["label"], fill="#333333", font=_font(13))
    img.save(output_path)


def plot_training_curves(history, figure_dir):
    epochs = history["epoch"]
    line_chart([{"x": epochs, "y": history["train_loss"], "label": "train loss", "color": "#2563EB"}, {"x": epochs, "y": history["valid_loss"], "label": "valid loss", "color": "#DC2626"}], Path(figure_dir) / "loss_curves.png", "Training and Validation Loss", "Loss")
    line_chart([{"x": epochs, "y": history["valid_accuracy"], "label": "valid accuracy", "color": "#059669"}], Path(figure_dir) / "validation_accuracy.png", "Validation Accuracy", "Accuracy")


def plot_confusion_matrix(matrix, output_path):
    output_path = Path(output_path)
    cell, left, top = 54, 130, 112
    img = Image.new("RGB", (left + cell * 10 + 60, top + cell * 10 + 76), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 28), "Fashion-MNIST Confusion Matrix", fill="#111111", font=_font(24, True))
    max_value = max(1, int(matrix.max()))
    for i in range(10):
        draw.text((20, top + i * cell + 18), CLASS_NAMES[i].split("/")[0][:10], fill="#333333", font=_font(12))
        draw.text((left + i * cell + 18, top - 28), str(i), fill="#333333", font=_font(13))
        for j in range(10):
            value = int(matrix[i, j])
            shade = int(255 - 190 * value / max_value)
            x0, y0 = left + j * cell, top + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(shade, min(255, shade + 16), 255), outline="white")
            draw.text((x0 + 12, y0 + 18), str(value), fill="#111111", font=_font(12))
    img.save(output_path)


def plot_weight_grid(weights, output_path, count=25):
    output_path = Path(output_path)
    count, cols, scale = min(count, weights.shape[1]), 5, 5
    rows = int(np.ceil(count / cols))
    cell = 28 * scale + 24
    img = Image.new("RGB", (cols * cell, rows * cell + 52), "white")
    draw = ImageDraw.Draw(img)
    draw.text((14, 14), "First-layer Hidden Unit Weights", fill="#111111", font=_font(22, True))
    for idx in range(count):
        vec = weights[:, idx].reshape(28, 28)
        vmax = max(abs(float(vec.min())), abs(float(vec.max())), 1e-8)
        tile = Image.fromarray(np.uint8(np.clip((vec / vmax + 1) / 2, 0, 1) * 255), "L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST).convert("RGB")
        r, c = divmod(idx, cols)
        img.paste(tile, (c * cell + 12, r * cell + 52))
    img.save(output_path)


def plot_misclassified(images, labels, preds, output_path, max_items=16):
    wrong = np.where(labels != preds)[0][:max_items]
    cols, scale = 4, 5
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    img = Image.new("RGB", (cols * (28 * scale + 42), rows * (28 * scale + 62) + 42), "white")
    draw = ImageDraw.Draw(img)
    draw.text((12, 12), "Misclassified Test Examples", fill="#111111", font=_font(22, True))
    for out_idx, idx in enumerate(wrong):
        r, c = divmod(out_idx, cols)
        x, y = c * (28 * scale + 42) + 20, r * (28 * scale + 62) + 48
        tile = Image.fromarray(np.uint8(images[idx].reshape(28, 28) * 255), "L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST).convert("RGB")
        img.paste(tile, (x, y))
        draw.text((x, y + 28 * scale + 8), f"T:{CLASS_NAMES[int(labels[idx])][:9]} P:{CLASS_NAMES[int(preds[idx])][:9]}", fill="#111111", font=_font(12))
    img.save(output_path)
