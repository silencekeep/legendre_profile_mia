#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


TABULAR_DATASETS: Dict[str, Dict[str, Any]] = {
    "purchase100": {
        "input_dim": 600,
        "num_classes": 100,
        "hidden_dims": [128],
        "default_epochs": 30,
        "default_lr": 0.1,
        "default_weight_decay": 0.0,
        "default_optimizer": "sgd",
        "default_batch_size": 256,
    },
    "texas100": {
        "input_dim": 6169,
        "num_classes": 100,
        "hidden_dims": [1024, 512, 256, 128],
        "default_epochs": 40,
        "default_lr": 0.001,
        "default_weight_decay": 0.0,
        "default_optimizer": "adamw",
        "default_batch_size": 512,
    },
    "location30": {
        "input_dim": 446,
        "num_classes": 30,
        "hidden_dims": [512, 256, 128],
        "default_epochs": 40,
        "default_lr": 0.001,
        "default_weight_decay": 0.0,
        "default_optimizer": "adamw",
        "default_batch_size": 256,
    },
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_arrays(value: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(value, dict):
        x, y = value["data"], value["targets"]
    else:
        x, y = value.data, value.targets
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _download(url: str, path: Path) -> None:
    ensure_dir(path.parent)
    if path.is_file() and path.stat().st_size > 0:
        return
    urlretrieve(url, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {path}")


def _save_pool_cache(
    root: Path, name: str, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    split = int(len(y) * 0.75)
    main = {"data": x[:split], "targets": y[:split]}
    population = {"data": x[split:], "targets": y[split:]}
    with (root / f"{name}.pkl").open("wb") as handle:
        pickle.dump(main, handle)
    with (root / f"{name}_population.pkl").open("wb") as handle:
        pickle.dump(population, handle)
    return x, y


def _load_cached_pool(root: Path, name: str) -> Tuple[np.ndarray, np.ndarray] | None:
    main_path = root / f"{name}.pkl"
    population_path = root / f"{name}_population.pkl"
    if not main_path.is_file() or not population_path.is_file():
        return None
    with main_path.open("rb") as handle:
        main = pickle.load(handle)
    with population_path.open("rb") as handle:
        population = pickle.load(handle)
    x_main, y_main = _cache_arrays(main)
    x_population, y_population = _cache_arrays(population)
    return (
        np.concatenate([x_main, x_population], axis=0),
        np.concatenate([y_main, y_population], axis=0),
    )


def _load_purchase100(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    cached = _load_cached_pool(root, "purchase100")
    if cached is not None:
        return cached
    raw_path = root / "dataset_purchase"
    if not raw_path.is_file():
        archive = root / "dataset_purchase.tgz"
        _download("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz", archive)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(root)
    values = pd.read_csv(raw_path, header=None).to_numpy()
    return _save_pool_cache(
        root,
        "purchase100",
        values[:, 1:].astype(np.float32),
        values[:, 0].astype(np.int64) - 1,
    )


def _load_texas100(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    cached = _load_cached_pool(root, "texas100")
    if cached is not None:
        return cached
    candidates = [root / "dataset_texas", root / "texas" / "100", root / "texas100"]
    raw_dir = next(
        (path for path in candidates if (path / "feats").is_file() and (path / "labels").is_file()),
        None,
    )
    if raw_dir is None:
        archive = root / "dataset_texas.tgz"
        _download("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz", archive)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(root)
        raw_dir = next(
            (path for path in candidates if (path / "feats").is_file() and (path / "labels").is_file()),
            None,
        )
    if raw_dir is None:
        raise FileNotFoundError("Texas100 feats/labels were not found")
    x = pd.read_csv(raw_dir / "feats", header=None).to_numpy().astype(np.float32)
    y = pd.read_csv(raw_dir / "labels", header=None).to_numpy().reshape(-1).astype(np.int64) - 1
    return _save_pool_cache(root, "texas100", x, y)


def _load_location30(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    cached = _load_cached_pool(root, "location30")
    if cached is not None:
        return cached
    candidates = [
        root / "location" / "data_complete.npz",
        root / "location30" / "data_complete.npz",
        root / "data_complete.npz",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        path = candidates[0]
        _download(
            "https://raw.githubusercontent.com/jjy1994/MemGuard/master/data/location/data_complete.npz",
            path,
        )
    with np.load(path) as values:
        x = np.asarray(values["x"], dtype=np.float32)
        y = np.asarray(values["y"], dtype=np.int64) - 1
    return _save_pool_cache(root, "location30", x, y)


def load_tabular_pool(name: str, data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    root = ensure_dir(Path(data_root))
    if name == "purchase100":
        return _load_purchase100(root)
    if name == "texas100":
        return _load_texas100(root)
    if name == "location30":
        return _load_location30(root)
    raise ValueError(f"Unsupported tabular dataset: {name}")


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.x[index]),
            torch.tensor(self.y[index], dtype=torch.long),
            int(index),
        )


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, int(hidden)), nn.ReLU(inplace=True)])
            previous = int(hidden)
        layers.append(nn.Linear(previous, int(num_classes)))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.flatten(1))


def _optimizer(
    name: str, parameters: Iterable[nn.Parameter], lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    if name.lower() == "sgd":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
        )
    if name.lower() == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_tabular_model_for_indices(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    device: torch.device,
    seed: int,
    early_stop_acc: float | None = None,
    early_stop_min_epochs: int = 1,
) -> Tuple[nn.Module, Dict[str, float]]:
    set_seed(seed)
    spec = TABULAR_DATASETS[name]
    dataset = TabularDataset(x, y)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        Subset(dataset, [int(value) for value in train_indices]),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    model = TabularMLP(spec["input_dim"], spec["num_classes"], spec["hidden_dims"])
    model.to(device)
    optimizer = _optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state = None
    best_metrics: Dict[str, float] = {"acc": -1.0, "loss": float("inf")}
    for epoch in range(int(epochs)):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, labels, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item()) * int(labels.numel())
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            count += int(labels.numel())
        metrics = {"loss": loss_sum / count, "acc": correct / count}
        print(
            f"  epoch {epoch + 1}/{epochs} | loss={metrics['loss']:.4f} | "
            f"acc={metrics['acc']:.4f}",
            flush=True,
        )
        if metrics["acc"] > best_metrics["acc"] + 1e-12 or (
            abs(metrics["acc"] - best_metrics["acc"]) <= 1e-12
            and metrics["loss"] < best_metrics["loss"]
        ):
            best_metrics = {**metrics, "best_epoch": float(epoch + 1)}
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if (
            early_stop_acc is not None
            and epoch + 1 >= int(early_stop_min_epochs)
            and metrics["acc"] >= float(early_stop_acc)
        ):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics["restored_best_train_state"] = 1.0
    model.to(device)
    return model, best_metrics


def collect_model_pass_tabular(
    *,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    view_count: int = 1,
    view_type: str = "identity",
    seed: int = 0,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    del seed
    if int(view_count) != 1 or str(view_type) != "identity":
        raise ValueError("The tabular pipeline uses one identity view")
    model.eval()
    n = len(y)
    loss = np.zeros(n, dtype=np.float32)
    true_prob = np.zeros(n, dtype=np.float32)
    max_other_prob = np.zeros(n, dtype=np.float32)
    margin = np.zeros(n, dtype=np.float32)
    logits_cache = None
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        inputs = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(y[start:end]).to(device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
        probs = torch.softmax(logits.float(), dim=1)
        rows = torch.arange(end - start, device=device)
        loss[start:end] = F.cross_entropy(
            logits.float(), labels, reduction="none"
        ).cpu().numpy()
        true_prob[start:end] = probs[rows, labels].cpu().numpy()
        other = probs.clone()
        other[rows, labels] = -1.0
        max_other_prob[start:end] = other.max(dim=1).values.cpu().numpy()
        true_logits = logits.float()[rows, labels]
        masked_logits = logits.float().clone()
        masked_logits[rows, labels] = float("-inf")
        margin[start:end] = (true_logits - masked_logits.max(dim=1).values).cpu().numpy()
        if logits_cache is None:
            logits_cache = np.zeros((n, int(logits.shape[1])), dtype=np.float32)
        logits_cache[start:end] = logits.float().cpu().numpy()
    log_true = np.log(np.clip(true_prob, 1e-30, 1.0)).astype(np.float32)
    return {
        "loss": loss,
        "true_prob": true_prob,
        "max_other_prob": max_other_prob,
        "margin": margin,
        "view_mean": log_true,
        "view_std": np.zeros(n, dtype=np.float32),
        "class_label": np.asarray(y, dtype=np.int64),
        "logits": logits_cache,
    }


def save_torch_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    model_name: str,
    task: str,
    role_name: str,
    role_kind: str,
    extra_meta: Dict[str, Any] | None = None,
) -> Path:
    ensure_dir(output_dir)
    path = output_dir / "model_state.pt"
    state = {
        key: value.detach().cpu().to(torch.float16)
        if value.is_floating_point()
        else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(state, path)
    metadata = {
        "task": task,
        "role_name": role_name,
        "role_kind": role_kind,
        "model_name": model_name,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_format": "state_dict_fp16",
        "tensor_count": len(state),
    }
    if extra_meta:
        metadata.update(extra_meta)
    (output_dir / "model_state_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return path


def _stratified_indices(
    labels: np.ndarray, train_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    groups: defaultdict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        groups[int(label)].append(int(index))
    train: List[int] = []
    heldout: List[int] = []
    for indices in groups.values():
        random.shuffle(indices)
        split = int(len(indices) * float(train_fraction))
        train.extend(indices[:split])
        heldout.extend(indices[split:])
    return np.asarray(train, dtype=np.int64), np.asarray(heldout, dtype=np.int64)


def _complementary_membership(
    sample_count: int, reference_count: int, seed: int
) -> np.ndarray:
    if int(reference_count) % 2 != 0:
        raise ValueError("reference count must be even")
    rng = np.random.default_rng(int(seed))
    membership = np.zeros((int(reference_count), int(sample_count)), dtype=bool)
    for pair in range(int(reference_count) // 2):
        first = rng.random(int(sample_count)) < 0.5
        membership[2 * pair] = first
        membership[2 * pair + 1] = ~first
    if not np.all(membership.sum(axis=0) == int(reference_count) // 2):
        raise RuntimeError("Reference membership balance check failed")
    return membership


def _write_role(
    role_dir: Path,
    *,
    name: str,
    role_kind: str,
    seed: int,
    train_indices: np.ndarray,
    sample_count: int,
    metrics: Dict[str, float],
    stats: Dict[str, np.ndarray],
    model: nn.Module,
) -> None:
    ensure_dir(role_dir)
    member = np.zeros(int(sample_count), dtype=bool)
    member[np.asarray(train_indices, dtype=np.int64)] = True
    np.savez_compressed(
        role_dir / "signals.npz",
        loss=stats["loss"],
        true_prob=stats["true_prob"],
        max_other_prob=stats["max_other_prob"],
        class_label=stats["class_label"],
        logits=stats["logits"],
    )
    np.savez_compressed(
        role_dir / "view_stats.npz",
        view_mean=stats["view_mean"],
        view_std=stats["view_std"],
        margin=stats["margin"],
    )
    (role_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (role_dir / "data_split.json").write_text(
        json.dumps(
            {
                "role_name": name,
                "role_kind": role_kind,
                "seed": int(seed),
                "clf_train_idx": np.flatnonzero(member).astype(int).tolist(),
                "clf_val_idx": np.flatnonzero(~member).astype(int).tolist(),
                "member_count": int(member.sum()),
                "nonmember_count": int((~member).sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_torch_checkpoint(
        output_dir=role_dir,
        model=model,
        model_name="mlp",
        task=name.split("/", 1)[0],
        role_name=name,
        role_kind=role_kind,
        extra_meta={"task_kind": "tabular"},
    )


def train_dataset(args: argparse.Namespace, name: str, device: torch.device) -> None:
    x, y = load_tabular_pool(name, args.data_root)
    task_root = ensure_dir(args.output_root / name)
    target_train, _ = _stratified_indices(y, args.target_train_fraction, args.seed)
    target_model, target_metrics = train_tabular_model_for_indices(
        name,
        x,
        y,
        target_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        device=device,
        seed=args.seed + 7,
    )
    target_stats = collect_model_pass_tabular(
        model=target_model, x=x, y=y, batch_size=args.score_batch_size, device=device
    )
    _write_role(
        task_root / "roles" / "target",
        name=f"{name}/target",
        role_kind="target",
        seed=args.seed,
        train_indices=target_train,
        sample_count=len(y),
        metrics=target_metrics,
        stats=target_stats,
        model=target_model,
    )
    target_model.to("cpu")
    del target_model

    membership = _complementary_membership(len(y), args.reference_count, args.seed + 17)
    for reference_id in range(args.reference_count):
        role_name = f"shadow_{reference_id:03d}"
        role_dir = task_root / "shadows" / role_name
        if all((role_dir / file_name).is_file() for file_name in (
            "signals.npz", "metrics.json", "data_split.json", "model_state.pt"
        )):
            print(f"[skip] {name}/{role_name}", flush=True)
            continue
        train_indices = np.flatnonzero(membership[reference_id])
        print(
            f"[tabular] {name} reference {reference_id + 1}/{args.reference_count} "
            f"train={len(train_indices)}",
            flush=True,
        )
        model, metrics = train_tabular_model_for_indices(
            name,
            x,
            y,
            train_indices,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            device=device,
            seed=args.seed + 1000 + reference_id,
        )
        stats = collect_model_pass_tabular(
            model=model, x=x, y=y, batch_size=args.score_batch_size, device=device
        )
        _write_role(
            role_dir,
            name=f"{name}/{role_name}",
            role_kind="reference",
            seed=args.seed + 1000 + reference_id,
            train_indices=train_indices,
            sample_count=len(y),
            metrics=metrics,
            stats=stats,
            model=model,
        )
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (task_root / "training_manifest.json").write_text(
        json.dumps(
            {
                "dataset": name,
                "samples": len(y),
                "reference_count": args.reference_count,
                "out_references_per_sample": args.reference_count // 2,
                "target_train_fraction": args.target_train_fraction,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "optimizer": args.optimizer,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular target and reference models.")
    parser.add_argument("--dataset", action="append", choices=sorted(TABULAR_DATASETS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-count", type=int, default=256)
    parser.add_argument("--target-train-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.time()
    for name in args.dataset:
        train_dataset(args, name, device)
    print(f"Completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
