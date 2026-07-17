from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


DATASETS = ("cifar10", "cifar100", "cinic10", "tinyimagenet")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True


def dataset_targets(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        return [int(value) for value in dataset.targets]
    for name in ("samples", "_samples", "imgs"):
        if hasattr(dataset, name):
            return [int(target) for _, target in getattr(dataset, name)]
    raise AttributeError(f"{type(dataset).__name__} does not expose labels")


def stratified_indices(
    dataset: Dataset, *, train_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    set_seed(seed)
    per_class: defaultdict[int, List[int]] = defaultdict(list)
    for index, target in enumerate(dataset_targets(dataset)):
        per_class[int(target)].append(int(index))
    train_indices: List[int] = []
    heldout_indices: List[int] = []
    for indices in per_class.values():
        random.shuffle(indices)
        split = int(len(indices) * float(train_fraction))
        train_indices.extend(indices[:split])
        heldout_indices.extend(indices[split:])
    return train_indices, heldout_indices


def dataset_stats(name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if name in {"cifar10", "cifar100", "cinic10"}:
        return CIFAR_MEAN, CIFAR_STD
    if name == "tinyimagenet":
        return TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
    raise ValueError(f"Unsupported image dataset: {name}")


def training_transform(name: str, augmentation: str) -> transforms.Compose:
    mean, std = dataset_stats(name)
    size = 64 if name == "tinyimagenet" else 32
    operations = []
    if augmentation == "stdaug":
        operations.extend(
            [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()]
        )
    elif name == "tinyimagenet":
        operations.append(transforms.Resize((64, 64)))
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)


def evaluation_transform(name: str) -> transforms.Compose:
    mean, std = dataset_stats(name)
    operations = []
    if name == "tinyimagenet":
        operations.append(transforms.Resize((64, 64)))
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)


def _find_directory(candidates: Sequence[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"{label} directory not found under {candidates[0].parent}")


def _cinic_train_root(data_root: Path) -> Path:
    return _find_directory(
        [
            data_root / "cinic10" / "train",
            data_root / "cinic10" / "CINIC-10" / "train",
            data_root / "CINIC-10" / "train",
            data_root / "cinic-10" / "train",
            data_root / "train",
        ],
        "CINIC-10 train",
    )


def _tiny_train_root(data_root: Path) -> Path:
    return _find_directory(
        [
            data_root / "tinyimagenet" / "tiny-imagenet-200" / "train",
            data_root / "tiny-imagenet-200" / "train",
        ],
        "Tiny-ImageNet train",
    )


def _cifar_dataset(name: str, data_root: Path, transform: transforms.Compose) -> Dataset:
    constructor = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    roots = [data_root / name, data_root]
    for root in roots:
        try:
            return constructor(root=str(root), train=True, download=False, transform=transform)
        except RuntimeError:
            continue
    return constructor(root=str(roots[0]), train=True, download=True, transform=transform)


def load_dataset(
    name: str, data_root: Path, *, augmentation: str | None
) -> Tuple[Dataset, int]:
    if name not in DATASETS:
        raise ValueError(f"Unsupported image dataset: {name}")
    transform = (
        evaluation_transform(name)
        if augmentation is None
        else training_transform(name, augmentation)
    )
    if name in {"cifar10", "cifar100"}:
        dataset = _cifar_dataset(name, Path(data_root), transform)
        return dataset, 10 if name == "cifar10" else 100
    if name == "cinic10":
        return datasets.ImageFolder(str(_cinic_train_root(Path(data_root))), transform), 10
    return datasets.ImageFolder(str(_tiny_train_root(Path(data_root))), transform), 200


def build_model(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    feature_extractor = models.resnet18(weights=None)
    feature_extractor.fc = nn.Identity()
    classifier = nn.Linear(512, int(num_classes))
    return feature_extractor, classifier


class ImageClassifier(nn.Module):
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(inputs))


def _safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(model_dir: Path, device: torch.device) -> nn.Module:
    split = json.loads((model_dir / "data_split.json").read_text(encoding="utf-8"))
    if str(split.get("arch", "resnet18")) != "resnet18":
        raise ValueError(f"{model_dir}: only ResNet-18 checkpoints are supported")
    feature_state = _safe_load(model_dir / "FE.pth")
    classifier_state = _safe_load(model_dir / "CF.pth")
    num_classes = int(classifier_state["weight"].shape[0])
    feature_extractor, classifier = build_model(num_classes)
    feature_extractor.load_state_dict(feature_state)
    classifier.load_state_dict(classifier_state)
    model = ImageClassifier(feature_extractor, classifier)
    model.to(device=device, memory_format=torch.channels_last).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample_index = self.indices[index]
        inputs, target = self.dataset[sample_index]
        return inputs, target, sample_index


def sanitize_tensor(tensor: torch.Tensor, clip_value: float = 1e6) -> torch.Tensor:
    tensor = tensor.float()
    if torch.isfinite(tensor).all():
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=clip_value, neginf=-clip_value)


def augment_std_views(
    inputs: torch.Tensor,
    *,
    repeats: int,
    seed: int,
    view_codes: torch.Tensor | None,
    pad: int = 4,
) -> torch.Tensor:
    batch_size, channels, height, width = inputs.shape
    total = batch_size * int(repeats)
    repeated = inputs[:, None].expand(-1, repeats, -1, -1, -1).reshape(
        total, channels, height, width
    )
    padded = F.pad(repeated, (pad, pad, pad, pad), mode="reflect")
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(int(seed))
    offset_count = 2 * pad + 1
    transform_count = offset_count * offset_count * 2
    if view_codes is None:
        keys = torch.rand((batch_size, transform_count), generator=generator, device=inputs.device)
        codes = torch.topk(keys, k=int(repeats), dim=1, largest=False).indices.reshape(total)
    else:
        if tuple(view_codes.shape) != (batch_size, int(repeats)):
            raise ValueError("view-code shape does not match batch size and view count")
        codes = view_codes.to(inputs.device, dtype=torch.long).reshape(total)
    crop_codes = codes % (offset_count * offset_count)
    top = crop_codes // offset_count
    left = crop_codes % offset_count
    flip_mask = codes >= offset_count * offset_count
    padded = torch.where(
        flip_mask[:, None, None, None], torch.flip(padded, dims=[3]), padded
    )
    windows = padded.unfold(2, height, 1).unfold(3, width, 1)
    rows = torch.arange(total, device=inputs.device)
    output = windows[rows, :, top, left, :, :]
    return output.contiguous(memory_format=torch.channels_last)


def augment_noaug_views(
    inputs: torch.Tensor,
    *,
    repeats: int,
    seed: int,
    mean_values: Sequence[float],
    std_values: Sequence[float],
) -> torch.Tensor:
    batch_size, channels, height, width = inputs.shape
    total = batch_size * int(repeats)
    repeated = inputs[:, None].expand(-1, repeats, -1, -1, -1).reshape(
        total, channels, height, width
    )
    padded = F.pad(repeated, (1, 1, 1, 1), mode="reflect")
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(int(seed))
    top = torch.randint(0, 3, (total,), generator=generator, device=inputs.device)
    left = torch.randint(0, 3, (total,), generator=generator, device=inputs.device)
    windows = padded.unfold(2, height, 1).unfold(3, width, 1)
    rows = torch.arange(total, device=inputs.device)
    output = windows[rows, :, top, left, :, :].contiguous()
    std = torch.tensor(std_values, dtype=output.dtype, device=output.device).view(
        1, channels, 1, 1
    )
    mean = torch.tensor(mean_values, dtype=output.dtype, device=output.device).view(
        1, channels, 1, 1
    )
    noise = torch.randn(
        output.shape, generator=generator, device=output.device, dtype=output.dtype
    ) * ((1.0 / 255.0) / std)
    output = torch.clamp(output + noise, min=(0.0 - mean) / std, max=(1.0 - mean) / std)
    return output.contiguous(memory_format=torch.channels_last)
