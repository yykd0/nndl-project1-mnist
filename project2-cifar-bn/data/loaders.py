"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tarfile
import urllib.request
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets


PACKAGE_DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DATA_DIR.parent
DATA_DIR = PROJECT_DIR / "safe_data"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MIN_BYTES = 100_000_000


def _archive_candidates(root):
    return [
        Path(root) / "cifar-10-python.tar.gz",
        PROJECT_DIR / "runtime_data" / "cifar-10-python.tar.gz",
        PACKAGE_DATA_DIR / "cifar-10-python.tar.gz",
    ]


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def _ensure_cifar_extracted(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-10-batches-py"
    if (extracted / "data_batch_1").exists() and (extracted / "test_batch").exists():
        return

    archive = next((candidate for candidate in _archive_candidates(root) if candidate.exists() and candidate.stat().st_size > CIFAR_MIN_BYTES), None)
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


def get_cifar_loader(root=None, batch_size=128, train=True, shuffle=True, num_workers=0, n_items=-1):
    if root is None:
        root = DATA_DIR
    _ensure_cifar_extracted(root)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    dataset = datasets.CIFAR10(root=str(root), train=train, download=not (Path(root) / "cifar-10-batches-py").exists(), transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
