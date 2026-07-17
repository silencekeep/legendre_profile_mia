#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(os.environ.get("INF2GUARD_REPO_ROOT", Path.cwd())).resolve()
LAB_ROOT = REPO_ROOT / "decision-boundary-lab"
from legendre_mia.models import image as image_models


DATASET_NAMES = ("cifar10", "cifar100", "tinyimagenet", "cinic10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train class-balanced image reference models."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=LAB_ROOT / "reference_models",
    )
    parser.add_argument("--data-root", type=Path, default=LAB_ROOT / "datasets")
    parser.add_argument("--datasets", type=str, default="cifar10,cifar100,tinyimagenet")
    parser.add_argument("--augmentation", choices=("stdaug", "noaug"), default="stdaug")
    parser.add_argument("--architecture", choices=("resnet18",), default="resnet18")
    parser.add_argument("--num-references", type=int, default=256)
    parser.add_argument("--design-seed", type=int, default=20260515)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-train-workers", type=int, default=1)
    parser.add_argument("--reference-ids", type=str, default="0-255")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def parse_datasets(text: str) -> List[str]:
    datasets = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [item for item in datasets if item not in DATASET_NAMES]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}; supported={DATASET_NAMES}")
    return datasets


def parse_id_list(text: str, upper: int) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    dedup = sorted(set(out))
    invalid = [item for item in dedup if item < 0 or item >= upper]
    if invalid:
        raise ValueError(f"Invalid reference ids for count={upper}: {invalid}")
    return dedup


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def dataset_data_root(data_root: Path, dataset_name: str) -> Path:
    return data_root / dataset_name


def dataset_tag(dataset_name: str, augmentation: str = "stdaug") -> str:
    return f"{dataset_name}_{augmentation}"


def dataset_output_root(args: argparse.Namespace, dataset_name: str) -> Path:
    return args.output_root / dataset_tag(dataset_name, args.augmentation)


def shadow_dir(args: argparse.Namespace, dataset_name: str, shadow_id: int) -> Path:
    return dataset_output_root(args, dataset_name) / f"shadow_{shadow_id:03d}"


def get_labels(dataset: object) -> np.ndarray:
    return np.asarray(image_models.dataset_targets(dataset), dtype=np.int64)


def set_seed(seed: int) -> None:
    image_models.set_seed(seed)


def build_membership_design(
    labels: np.ndarray,
    *,
    num_references: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if num_references % 2 != 0:
        raise ValueError(f"num_references must be even; got {num_references}")
    rng = np.random.default_rng(seed)
    n_samples = int(labels.size)
    num_pairs = num_references // 2
    membership = np.zeros((num_references, n_samples), dtype=bool)
    class_counts = Counter(labels.tolist())
    class_summaries: Dict[str, Dict[str, int]] = {}

    for cls in sorted(class_counts):
        cls_indices = np.flatnonzero(labels == cls).astype(np.int64)
        if cls_indices.size % 2 != 0:
            raise ValueError(
                f"Class {cls} has odd sample count {cls_indices.size}; exact 50/50 split impossible."
            )
        half = int(cls_indices.size // 2)
        for pair_id in range(num_pairs):
            perm = rng.permutation(cls_indices)
            even_in = perm[:half]
            odd_in = perm[half:]
            membership[2 * pair_id, even_in] = True
            membership[2 * pair_id + 1, odd_in] = True
        class_summaries[str(cls)] = {
            "samples": int(cls_indices.size),
            "per_shadow_in": int(half),
            "per_shadow_out": int(half),
        }

    validate_membership(labels, membership)
    per_sample_in = membership.sum(axis=0)
    per_model_in = membership.sum(axis=1)
    summary = {
        "design": "exact class-balanced complementary split",
        "seed": int(seed),
        "num_references": int(num_references),
        "num_pairs": int(num_pairs),
        "n_samples": int(n_samples),
        "per_sample_in_min": int(per_sample_in.min()),
        "per_sample_in_max": int(per_sample_in.max()),
        "per_sample_out_min": int((num_references - per_sample_in).min()),
        "per_sample_out_max": int((num_references - per_sample_in).max()),
        "per_model_in_min": int(per_model_in.min()),
        "per_model_in_max": int(per_model_in.max()),
        "per_model_out_min": int((n_samples - per_model_in).min()),
        "per_model_out_max": int((n_samples - per_model_in).max()),
        "class_count_min": int(min(class_counts.values())),
        "class_count_max": int(max(class_counts.values())),
        "per_reference_per_class_in_min": int(min(class_counts.values()) // 2),
        "per_reference_per_class_in_max": int(max(class_counts.values()) // 2),
        "pair_rule": "For every sample x and pair p, exactly one of rows 2p and 2p+1 is IN.",
        "subset_rule": "Any complete set of pairs preserves exact per-sample 50/50 IN/OUT.",
        "class_summaries": class_summaries,
    }
    return membership, summary


def validate_membership(labels: np.ndarray, membership: np.ndarray) -> None:
    num_references, n_samples = membership.shape
    if num_references % 2 != 0:
        raise RuntimeError("membership has an odd number of references")
    num_pairs = num_references // 2

    pair_sums = membership[0::2].astype(np.int8) + membership[1::2].astype(np.int8)
    if int(pair_sums.min()) != 1 or int(pair_sums.max()) != 1:
        raise RuntimeError("Complementary-pair invariant failed: a pair is not exactly one IN/one OUT.")

    per_sample_in = membership.sum(axis=0)
    if int(per_sample_in.min()) != num_pairs or int(per_sample_in.max()) != num_pairs:
        raise RuntimeError("Sample-level exact half-IN balance failed.")

    per_model_in = membership.sum(axis=1)
    if int(per_model_in.min()) != n_samples // 2 or int(per_model_in.max()) != n_samples // 2:
        raise RuntimeError("Model-level exact half-dataset balance failed.")

    for cls in sorted(set(labels.tolist())):
        cls_indices = np.flatnonzero(labels == cls)
        cls_size = int(cls_indices.size)
        expected = cls_size // 2
        if cls_size % 2 != 0:
            raise RuntimeError(f"Class {cls} has odd sample count {cls_size}.")
        per_model_cls = membership[:, cls_indices].sum(axis=1)
        if int(per_model_cls.min()) != expected or int(per_model_cls.max()) != expected:
            raise RuntimeError(f"Per-class balance failed for class {cls}.")


def materialize_dataset(args: argparse.Namespace, dataset_name: str) -> None:
    out_root = dataset_output_root(args, dataset_name)
    existing_summary_path = out_root / "design_summary.json"
    existing_summary = {}
    if existing_summary_path.is_file():
        try:
            existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_summary = {}
    design_seed = int(args.design_seed) + stable_dataset_offset(dataset_name)
    split_design_changed = bool(
        args.overwrite_splits
        or existing_summary.get("seed") != design_seed
        or existing_summary.get("num_references") != int(args.num_references)
        or existing_summary.get("augmentation") != str(args.augmentation)
        or existing_summary.get("arch") != str(args.architecture)
    )
    dataset, _ = image_models.load_dataset(
        dataset_name, args.data_root, augmentation=args.augmentation
    )
    labels = get_labels(dataset)
    membership, summary = build_membership_design(
        labels,
        num_references=int(args.num_references),
        seed=design_seed,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "membership_matrix.npy", membership)
    summary.update(
        {
            "dataset": dataset_name,
            "dataset_tag": dataset_tag(dataset_name, args.augmentation),
            "augmentation": args.augmentation,
            "arch": args.architecture,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "Adam",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [24],
            "scheduler_gamma": 0.1,
        }
    )
    write_json(out_root / "design_summary.json", summary)

    pair_manifest = []
    for pair_id in range(int(args.num_references) // 2):
        pair_manifest.append(
            {
                "pair_id": int(pair_id),
                "shadow_even": int(2 * pair_id),
                "shadow_odd": int(2 * pair_id + 1),
            }
        )
    write_json(out_root / "pair_manifest.json", pair_manifest)

    for shadow_id in range(int(args.num_references)):
        sdir = out_root / f"shadow_{shadow_id:03d}"
        split_path = sdir / "data_split.json"
        if split_path.exists() and not split_design_changed:
            continue
        train_idx = np.flatnonzero(membership[shadow_id]).astype(int).tolist()
        val_idx = np.flatnonzero(~membership[shadow_id]).astype(int).tolist()
        split_payload = {
            "clf_train_idx": train_idx,
            "clf_val_idx": val_idx,
            "dataset": dataset_name,
            "dataset_tag": dataset_tag(dataset_name, args.augmentation),
            "augmentation": args.augmentation,
            "arch": args.architecture,
            "seed": int(args.seed_base) + int(shadow_id),
            "shadow_id": int(shadow_id),
            "pair_id": int(shadow_id // 2),
            "pair_role": "even" if shadow_id % 2 == 0 else "odd",
            "split_design": "exact_class_balanced_complementary_pairs",
            "num_references": int(args.num_references),
            "num_pairs": int(args.num_references // 2),
            "design_seed": design_seed,
        }
        sdir.mkdir(parents=True, exist_ok=True)
        write_json(split_path, split_payload)

    print(
        f"[prepared] {dataset_tag(dataset_name, args.augmentation)} samples={len(labels)} "
        f"references={args.num_references} out={out_root}",
        flush=True,
    )


def stable_dataset_offset(dataset_name: str) -> int:
    return {"cifar10": 0, "cifar100": 1000, "tinyimagenet": 2000, "cinic10": 3000}[dataset_name]


def completed(sdir: Path, args: argparse.Namespace) -> bool:
    if not (sdir / "FE.pth").is_file() or not (sdir / "CF.pth").is_file() or not (sdir / "data_split.json").is_file():
        return False
    summary_path = sdir / "train_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = {
        "status": "completed",
        "architecture": str(args.architecture),
        "augmentation": str(args.augmentation),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "design_seed": int(args.design_seed),
    }
    try:
        shadow_id = int(sdir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return False
    expected["seed"] = int(args.seed_base) + shadow_id
    return all(summary.get(key) == value for key, value in expected.items())


def load_split(sdir: Path) -> Tuple[List[int], List[int], Dict[str, object]]:
    payload = json.loads((sdir / "data_split.json").read_text(encoding="utf-8"))
    return [int(x) for x in payload["clf_train_idx"]], [int(x) for x in payload["clf_val_idx"]], payload


def make_train_loader(dataset: object, train_idx: Sequence[int], args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(train_idx)),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )


def train_one_reference(
    dataset: object,
    num_classes: int,
    dataset_name: str,
    shadow_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    sdir = shadow_dir(args, dataset_name, shadow_id)
    train_idx, val_idx, split = load_split(sdir)
    seed = int(split["seed"])
    set_seed(seed)
    loader = make_train_loader(dataset, train_idx, args)

    fe, cf = image_models.build_model(num_classes)
    fe.to(device)
    cf.to(device)
    optimizer = optim.Adam(list(fe.parameters()) + list(cf.parameters()), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[24], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    final_train_loss = None
    final_train_acc = None

    for epoch in range(int(args.epochs)):
        fe.train()
        cf.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = cf(fe(inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * int(inputs.size(0))
            predicted = outputs.argmax(dim=1)
            train_correct += int(predicted.eq(targets).sum().item())
            train_total += int(targets.size(0))

        train_loss /= train_total
        train_acc = train_correct / train_total
        final_train_loss = train_loss
        final_train_acc = train_acc
        scheduler.step()

        if epoch % 10 == 0:
            print(
                f"Shadow {sdir.name} Epoch {epoch + 1}: "
                f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}",
                flush=True,
            )

    sdir.mkdir(parents=True, exist_ok=True)
    torch.save(fe.state_dict(), sdir / "FE.pth")
    torch.save(cf.state_dict(), sdir / "CF.pth")
    elapsed = time.time() - start
    summary = {
        "status": "completed",
        "dataset": dataset_name,
        "dataset_tag": dataset_tag(dataset_name, args.augmentation),
        "architecture": args.architecture,
        "augmentation": args.augmentation,
        "shadow_id": int(shadow_id),
        "pair_id": int(shadow_id // 2),
        "pair_role": "even" if shadow_id % 2 == 0 else "odd",
        "seed": int(seed),
        "design_seed": int(args.design_seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "optimizer": "Adam",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": "MultiStepLR",
        "scheduler_milestones": [24],
        "scheduler_gamma": 0.1,
        "train_size": int(len(train_idx)),
        "heldout_size": int(len(val_idx)),
        "final_train_loss": float(final_train_loss),
        "final_train_acc": float(final_train_acc),
        "runtime_sec": float(elapsed),
    }
    write_json(sdir / "train_summary.json", summary)
    append_jsonl(
        dataset_output_root(args, dataset_name) / f"worker_{args.worker_id:02d}_events.jsonl",
        {
            "event": "completed",
            "time": time.time(),
            "dataset": dataset_name,
            "shadow_id": int(shadow_id),
            "runtime_sec": float(elapsed),
            "final_train_acc": float(final_train_acc),
        },
    )
    print(
        f"[done] {dataset_tag(dataset_name, args.augmentation)}/{sdir.name} "
        f"runtime={elapsed:.1f}s final_train_acc={final_train_acc:.4f}",
        flush=True,
    )

    del loader, fe, cf, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_prepared(args: argparse.Namespace, dataset_name: str) -> None:
    out_root = dataset_output_root(args, dataset_name)
    matrix_path = out_root / "membership_matrix.npy"
    summary_path = out_root / "design_summary.json"
    if not matrix_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing prepared design under {out_root}")
    dataset, _ = image_models.load_dataset(
        dataset_name, args.data_root, augmentation=args.augmentation
    )
    labels = get_labels(dataset)
    membership = np.load(matrix_path)
    validate_membership(labels, membership)
    print(
        f"[validated] {dataset_tag(dataset_name, args.augmentation)} design={matrix_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if int(args.num_references) % 2 != 0:
        raise ValueError("--num-references must be even")
    if int(args.num_train_workers) < 1:
        raise ValueError("--num-train-workers must be >= 1")
    if int(args.worker_id) < 0 or int(args.worker_id) >= int(args.num_train_workers):
        raise ValueError("--worker-id must be in [0, num-train-workers)")

    datasets = parse_datasets(args.datasets)
    reference_ids = parse_id_list(args.reference_ids, int(args.num_references))

    if not args.skip_prepare:
        for name in datasets:
            materialize_dataset(args, name)
    else:
        for name in datasets:
            validate_prepared(args, name)

    if args.prepare_only:
        print("[prepare-only] complete", flush=True)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(
        f"[worker] id={args.worker_id}/{args.num_train_workers} device={device} "
        f"datasets={datasets} ids={len(reference_ids)}",
        flush=True,
    )

    trained = 0
    for name in datasets:
        dataset, num_classes = image_models.load_dataset(
            name, args.data_root, augmentation=args.augmentation
        )
        worker_ids = [
            item for item in reference_ids if item % int(args.num_train_workers) == int(args.worker_id)
        ]

        for reference_id in worker_ids:
            sdir = shadow_dir(args, name, reference_id)
            if args.skip_existing and completed(sdir, args):
                print(
                    f"[skip] {dataset_tag(name, args.augmentation)}/{sdir.name}",
                    flush=True,
                )
                continue
            print(
                f"Training reference {reference_id + 1}/{args.num_references} "
                f"(id={reference_id:03d}, seed={args.seed_base + reference_id}): {sdir}",
                flush=True,
            )
            train_one_reference(dataset, num_classes, name, reference_id, args, device)
            trained += 1

    print(f"[worker-complete] id={args.worker_id} trained={trained}", flush=True)


if __name__ == "__main__":
    main()
