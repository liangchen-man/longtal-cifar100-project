from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def _optional_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

        return torch, DataLoader, Dataset, Subset, WeightedRandomSampler
    except Exception:
        return None, None, None, None, None


def _optional_torchvision():
    try:
        from torchvision import datasets, transforms

        return datasets, transforms
    except Exception:
        return None, None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, *_ = _optional_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def img_num_per_cls(
    cls_num: int = 100,
    max_num: int = 500,
    imb_type: str = "exp",
    imb_ratio: float = 100,
) -> list[int]:
    """Return class counts for the standard CIFAR-LT exp/step protocols."""
    if imb_ratio <= 0:
        raise ValueError("imb_ratio must be positive")
    if imb_type == "exp":
        return [
            max(1, int(max_num * ((1.0 / imb_ratio) ** (i / (cls_num - 1.0)))))
            for i in range(cls_num)
        ]
    if imb_type == "step":
        return [int(max_num)] * (cls_num // 2) + [
            max(1, int(max_num / imb_ratio))
        ] * (cls_num - cls_num // 2)
    if imb_type in {"none", "balanced"}:
        return [int(max_num)] * cls_num
    raise ValueError(f"Unknown imbalance type: {imb_type}")


def build_indices(targets: list[int] | np.ndarray, counts: list[int], seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    indices: list[int] = []
    for class_id, count in enumerate(counts):
        class_indices = np.where(targets == class_id)[0].copy()
        rng.shuffle(class_indices)
        indices.extend(class_indices[: min(count, len(class_indices))].tolist())
    rng.shuffle(indices)
    return indices


def _cifar_transform(train: bool = True, augment: bool = True):
    _, transforms = _optional_torchvision()
    if transforms is None:
        raise RuntimeError(
            "torchvision is required for real CIFAR-100-LT loading. "
            "Install torch/torchvision or use data.dataset=synthetic for smoke checks."
        )
    if train and augment:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_cifar100lt(
    root: str = "./data",
    imb_type: str = "exp",
    imb_ratio: float = 100,
    seed: int = 42,
    augment: bool = True,
    download: bool = True,
):
    """Build train LT subset, balanced test set, and same-distribution LT test subset."""
    datasets, _ = _optional_torchvision()
    _, _, _, Subset, _ = _optional_torch()
    if datasets is None or Subset is None:
        raise RuntimeError("torch and torchvision are required to build CIFAR-100-LT datasets")

    train = datasets.CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=_cifar_transform(True, augment),
    )
    test = datasets.CIFAR100(
        root=root,
        train=False,
        download=download,
        transform=_cifar_transform(False, False),
    )
    train_counts = img_num_per_cls(100, 500, imb_type, imb_ratio)
    test_counts = img_num_per_cls(100, 100, imb_type, imb_ratio)
    train_set = Subset(train, build_indices(train.targets, train_counts, seed))
    test_imb = Subset(test, build_indices(test.targets, test_counts, seed + 1))
    return train_set, test, test_imb, train_counts, test_counts, train.classes


def _draw_synthetic_image(class_id: int, rng: np.random.Generator, image_size: int = 32) -> np.ndarray:
    """Generate a CIFAR-sized class-coded image for offline protocol tests."""
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)

    base = np.array(
        [
            (37 * class_id + 29) % 255,
            (67 * class_id + 83) % 255,
            (97 * class_id + 151) % 255,
        ],
        dtype=np.float32,
    )
    image[:] = base * 0.35 + 35

    period = 4 + class_id % 7
    phase = class_id % period
    orientation = class_id % 4
    if orientation == 0:
        mask = ((xx + phase) % period) < max(2, period // 2)
    elif orientation == 1:
        mask = ((yy + phase) % period) < max(2, period // 2)
    elif orientation == 2:
        mask = ((xx + yy + phase) % period) < max(2, period // 2)
    else:
        mask = ((xx - yy + phase) % period) < max(2, period // 2)
    image[mask] += np.roll(base, class_id % 3) * 0.45

    cx = 5 + (class_id * 11) % (image_size - 10)
    cy = 5 + (class_id * 17) % (image_size - 10)
    radius = 2 + class_id % 5
    patch = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
    image[patch] = 255 - base * 0.25

    # Encode coarse/fine superclass-like cues; this makes the synthetic smoke
    # dataset non-random while preserving a genuine long-tail label protocol.
    band = class_id // 20
    image[band : band + 3, :, band % 3] = 240
    image[:, (class_id % 20) : (class_id % 20) + 2, (band + 1) % 3] = 225

    noise = rng.normal(0, 12, size=image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def make_synthetic_cifar100lt_arrays(
    cls_num: int = 100,
    max_train: int = 30,
    max_test: int = 8,
    imb_type: str = "exp",
    imb_ratio: float = 100,
    seed: int = 42,
    image_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[str]]:
    """Return numpy arrays for offline checks when CIFAR/torch are unavailable."""
    rng = np.random.default_rng(seed)
    train_counts = img_num_per_cls(cls_num, max_train, imb_type, imb_ratio)
    test_counts = img_num_per_cls(cls_num, max_test, imb_type, imb_ratio)
    class_names = [f"synthetic_{i:03d}" for i in range(cls_num)]

    def build_split(counts: list[int], split_seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
        images: list[np.ndarray] = []
        labels: list[int] = []
        split_rng = np.random.default_rng(seed + split_seed_offset)
        for class_id, count in enumerate(counts):
            for _ in range(count):
                images.append(_draw_synthetic_image(class_id, split_rng, image_size))
                labels.append(class_id)
        order = split_rng.permutation(len(labels))
        return np.stack(images, axis=0)[order], np.asarray(labels, dtype=np.int64)[order]

    x_train, y_train = build_split(train_counts, 0)
    x_test_bal, y_test_bal = build_split([max_test] * cls_num, 10_000)
    x_test_imb, y_test_imb = build_split(test_counts, 20_000)
    return (
        x_train,
        y_train,
        x_test_bal,
        y_test_bal,
        x_test_imb,
        y_test_imb,
        train_counts,
        test_counts,
        class_names,
    )


@dataclass
class DatasetBundle:
    train: Any
    balanced_test: Any
    imbalanced_test: Any
    train_counts: list[int]
    imbalanced_test_counts: list[int]
    class_names: list[str]


def build_array_dataset_bundle(cfg: dict[str, Any]) -> DatasetBundle:
    data_cfg = cfg.get("data", {})
    dataset = data_cfg.get("dataset", "cifar100").lower()
    if dataset not in {"synthetic", "synthetic_cifar100"}:
        raise ValueError("Array dataset bundle is only available for data.dataset=synthetic")
    arrays = make_synthetic_cifar100lt_arrays(
        cls_num=int(data_cfg.get("num_classes", 100)),
        max_train=int(data_cfg.get("synthetic_max_train_per_class", 30)),
        max_test=int(data_cfg.get("synthetic_test_per_class", 8)),
        imb_type=data_cfg.get("imbalance_type", "exp"),
        imb_ratio=float(data_cfg.get("imbalance_ratio", 100)),
        seed=int(cfg.get("seed", 42)),
        image_size=int(data_cfg.get("image_size", 32)),
    )
    x_train, y_train, x_bal, y_bal, x_imb, y_imb, counts, test_counts, names = arrays
    return DatasetBundle(
        train=(x_train, y_train),
        balanced_test=(x_bal, y_bal),
        imbalanced_test=(x_imb, y_imb),
        train_counts=counts,
        imbalanced_test_counts=test_counts,
        class_names=names,
    )


def _array_to_tensor(image: np.ndarray):
    torch, *_ = _optional_torch()
    if torch is None:
        raise RuntimeError("torch is required for tensor conversion")
    arr = image.astype(np.float32) / 255.0
    arr = (arr - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def make_torch_array_dataset(images: np.ndarray, labels: np.ndarray):
    torch, _, Dataset, _, _ = _optional_torch()
    if torch is None or Dataset is None:
        raise RuntimeError("torch is required for synthetic neural-network training")

    class ArrayDataset(Dataset):
        targets = labels.tolist()

        def __len__(self):
            return int(labels.shape[0])

        def __getitem__(self, index):
            return _array_to_tensor(images[index]), int(labels[index])

    return ArrayDataset()


def dataset_targets(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_targets = dataset_targets(dataset.dataset)
        return [int(base_targets[i]) for i in dataset.indices]
    if hasattr(dataset, "labels"):
        return [int(x) for x in dataset.labels]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def make_loaders(cfg: dict[str, Any]):
    torch, DataLoader, _, _, WeightedRandomSampler = _optional_torch()
    if torch is None or DataLoader is None:
        raise RuntimeError("torch is required for neural-network loaders")

    data_cfg = cfg.get("data", {})
    dataset = data_cfg.get("dataset", "cifar100").lower()
    if dataset in {"synthetic", "synthetic_cifar100"}:
        bundle = build_array_dataset_bundle(cfg)
        train_set = make_torch_array_dataset(*bundle.train)
        balanced_test = make_torch_array_dataset(*bundle.balanced_test)
        imbalanced_test = make_torch_array_dataset(*bundle.imbalanced_test)
        counts = bundle.train_counts
        test_counts = bundle.imbalanced_test_counts
        class_names = bundle.class_names
    elif dataset in {"cifar100", "cifar-100"}:
        train_set, balanced_test, imbalanced_test, counts, test_counts, class_names = build_cifar100lt(
            root=data_cfg.get("root", "./data"),
            imb_type=data_cfg.get("imbalance_type", "exp"),
            imb_ratio=float(data_cfg.get("imbalance_ratio", 100)),
            seed=int(cfg.get("seed", 42)),
            augment=bool(data_cfg.get("augment", True)),
            download=bool(data_cfg.get("download", True)),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 4))
    sampler = None
    shuffle = True
    sampler_name = data_cfg.get("sampler", "none").lower()
    if sampler_name in {"class_balanced", "weighted"}:
        targets = dataset_targets(train_set)
        weights = [1.0 / max(1, counts[label]) for label in targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
    )
    balanced_loader = DataLoader(
        balanced_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
    )
    imbalanced_loader = DataLoader(
        imbalanced_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
    )
    data = (
        train_set,
        balanced_test,
        imbalanced_test,
        counts,
        test_counts,
        class_names,
    )
    return data, train_loader, balanced_loader, imbalanced_loader
