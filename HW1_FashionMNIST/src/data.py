import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", [
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
    ]),
    "train_labels": ("train-labels-idx1-ubyte.gz", [
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
    ]),
    "test_images": ("t10k-images-idx3-ubyte.gz", [
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
    ]),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", [
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
    ]),
}
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _is_valid_gzip(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception:
        return False


def download_fashion_mnist(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, urls in FILES.values():
        path = raw_dir / filename
        if _is_valid_gzip(path):
            continue
        errors = []
        for url in urls:
            tmp = path.with_suffix(path.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            try:
                print(f"Downloading {filename} ...")
                urllib.request.urlretrieve(url, tmp)
                if not _is_valid_gzip(tmp):
                    raise ValueError("incomplete gzip archive")
                tmp.replace(path)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                if tmp.exists():
                    tmp.unlink()
        else:
            raise RuntimeError(f"Could not download {filename}. " + " | ".join(errors))


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_fashion_mnist(raw_dir=RAW_DIR, download=True):
    raw_dir = Path(raw_dir)
    if download:
        download_fashion_mnist(raw_dir)
    return (
        _read_images(raw_dir / FILES["train_images"][0]),
        _read_labels(raw_dir / FILES["train_labels"][0]),
        _read_images(raw_dir / FILES["test_images"][0]),
        _read_labels(raw_dir / FILES["test_labels"][0]),
    )


def make_train_valid_split(images, labels, valid_size=10000, seed=42, train_limit=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(images.shape[0])
    if train_limit is not None:
        idx = idx[:min(images.shape[0], valid_size + train_limit)]
    images, labels = images[idx], labels[idx]
    return (images[valid_size:], labels[valid_size:]), (images[:valid_size], labels[:valid_size])


def iterate_minibatches(x, y, batch_size, shuffle=True, seed=None):
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = idx[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]
