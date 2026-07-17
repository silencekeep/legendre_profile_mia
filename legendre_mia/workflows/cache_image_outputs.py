#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("INF2GUARD_REPO_ROOT", Path.cwd())).resolve()
VALIDATE_ROOT = SCRIPT_ROOT
PROJECT_ROOT = REPO_ROOT / "decision-boundary-lab"
DATA_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_OUT_ROOT = VALIDATE_ROOT / "outputs" / "image_model_outputs"

if str(VALIDATE_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATE_ROOT))

from legendre_mia.models import image as image_models  # noqa: E402


BASE_DATASETS = ("cifar10", "cifar100", "tinyimagenet", "cinic10")
AUGMENTATIONS = ("stdaug", "noaug")
SUPPORTED_DATASET_TAGS = tuple(
    f"{dataset}_{augmentation}"
    for dataset in BASE_DATASETS
    for augmentation in AUGMENTATIONS
)
DEFAULT_DATASET_TAGS = tuple(f"{dataset}_stdaug" for dataset in BASE_DATASETS)
DATASET_NAMES = {
    f"{dataset}_{augmentation}": dataset
    for dataset in BASE_DATASETS
    for augmentation in AUGMENTATIONS
}
DATASET_BATCH_SIZE = {
    "cifar10": 512,
    "cifar100": 512,
    "tinyimagenet": 256,
    "cinic10": 512,
}


@dataclass(frozen=True)
class Job:
    dataset_tag: str
    dataset_name: str
    augmentation: str
    model_name: str
    model_dir: Path
    ordinal: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Model-first multi-view probability bank exporter. "
            "Each model is forwarded once on all dataset samples and writes true-class "
            "probability plus strongest competing-class probability."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASET_TAGS))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--shadow-root", type=Path, default=PROJECT_ROOT / "shadows")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--include-targets", action="store_true", help="Also export the independent target model for each dataset.")
    parser.add_argument("--target-only", action="store_true", help="Export only independent target models, not shadows.")
    parser.add_argument("--shadow-names", type=str, default="", help="Optional comma-separated shadow names to export for every dataset.")
    parser.add_argument("--shadow-start", type=int, default=-1, help="Optional inclusive numeric shadow id lower bound.")
    parser.add_argument("--shadow-end", type=int, default=-1, help="Optional inclusive numeric shadow id upper bound.")
    parser.add_argument("--max-shadows-per-dataset", type=int, default=0, help="Optional cap after filtering; 0 means no cap.")
    parser.add_argument("--views", type=int, default=64)
    parser.add_argument(
        "--unique-stdaug-views",
        action="store_true",
        help="Draw crop/flip stdaug transforms without replacement within each sample.",
    )
    parser.add_argument(
        "--sample-index-views",
        action="store_true",
        help=(
            "Make stdaug view identities depend on dataset sample_idx instead of loader batch_counter. "
            "This keeps views invariant when batch-size changes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers-total", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=2048)
    parser.add_argument(
        "--cifar10-batch-size",
        type=int,
        default=0,
        help="Optional CIFAR-10 loader batch-size override. 0 keeps the default.",
    )
    parser.add_argument(
        "--cifar100-batch-size",
        type=int,
        default=0,
        help="Optional CIFAR-100 loader batch-size override. 0 keeps the default.",
    )
    parser.add_argument(
        "--tiny-batch-size",
        type=int,
        default=0,
        help="Optional TinyImageNet loader batch-size override. 0 keeps the default.",
    )
    parser.add_argument(
        "--cinic10-batch-size",
        type=int,
        default=0,
        help="Optional CINIC-10 loader batch-size override. 0 keeps the default.",
    )
    parser.add_argument(
        "--include-center-view",
        action="store_true",
        help="Force view 0 to be the unflipped center crop, then fill remaining stdaug views without replacement.",
    )
    parser.add_argument("--num-loader-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dataset_batch_size(dataset_tag: str, args: argparse.Namespace) -> int:
    dataset_name, _ = parse_dataset_tag(dataset_tag)
    if dataset_name == "cifar10" and int(args.cifar10_batch_size) > 0:
        return int(args.cifar10_batch_size)
    if dataset_name == "cifar100" and int(args.cifar100_batch_size) > 0:
        return int(args.cifar100_batch_size)
    if dataset_name == "tinyimagenet" and int(args.tiny_batch_size) > 0:
        return int(args.tiny_batch_size)
    if dataset_name == "cinic10" and int(args.cinic10_batch_size) > 0:
        return int(args.cinic10_batch_size)
    return int(DATASET_BATCH_SIZE[dataset_name])


def set_perf_flags() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_dataset_tag(dataset_tag: str) -> tuple[str, str]:
    for augmentation in AUGMENTATIONS:
        marker = f"_{augmentation}"
        if dataset_tag.endswith(marker):
            dataset_name = dataset_tag[: -len(marker)]
            if dataset_name in BASE_DATASETS:
                return dataset_name, augmentation
    raise ValueError(f"Unsupported dataset tag: {dataset_tag}")


def shadow_id_from_name(name: str) -> int:
    if not name.startswith("shadow_"):
        raise ValueError(f"Unsupported shadow name: {name}")
    return int(name.split("_", 1)[1])


def filter_shadow_dirs(dirs: Sequence[Path], args: argparse.Namespace) -> List[Path]:
    out = list(dirs)
    if args.shadow_names.strip():
        wanted = set(parse_csv(args.shadow_names))
        by_name = {path.name: path for path in out}
        missing = sorted(name for name in wanted if name not in by_name)
        if missing:
            raise FileNotFoundError(f"Requested shadows missing: {missing}")
        out = [by_name[name] for name in sorted(wanted, key=shadow_id_from_name)]
    if int(args.shadow_start) >= 0:
        out = [path for path in out if shadow_id_from_name(path.name) >= int(args.shadow_start)]
    if int(args.shadow_end) >= 0:
        out = [path for path in out if shadow_id_from_name(path.name) <= int(args.shadow_end)]
    if int(args.max_shadows_per_dataset) > 0:
        out = out[: int(args.max_shadows_per_dataset)]
    return out


def list_jobs(args: argparse.Namespace) -> List[Job]:
    jobs: List[Job] = []
    ordinal = 0
    for dataset_tag in parse_csv(args.datasets):
        if dataset_tag not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset tag: {dataset_tag}")
        dataset_name, augmentation = parse_dataset_tag(dataset_tag)
        if args.include_targets or args.target_only:
            target_dir = args.model_root / dataset_tag
            if not target_dir.exists():
                raise FileNotFoundError(f"Missing target model dir: {target_dir}")
            jobs.append(
                Job(
                    dataset_tag=dataset_tag,
                    dataset_name=dataset_name,
                    augmentation=augmentation,
                    model_name="target",
                    model_dir=target_dir,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
        if args.target_only:
            continue
        model_root = args.shadow_root / dataset_tag
        if not model_root.exists():
            raise FileNotFoundError(f"Missing shadow root: {model_root}")
        dirs = sorted(
            path
            for path in model_root.glob("shadow_*")
            if path.is_dir()
            and (path / "FE.pth").exists()
            and (path / "CF.pth").exists()
            and (path / "data_split.json").exists()
        )
        dirs = filter_shadow_dirs(dirs, args)
        for model_dir in dirs:
            jobs.append(
                Job(
                    dataset_tag=dataset_tag,
                    dataset_name=dataset_name,
                    augmentation=augmentation,
                    model_name=model_dir.name,
                    model_dir=model_dir,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return jobs


def split_for_worker(jobs: Sequence[Job], worker_index: int, num_workers_total: int) -> List[Job]:
    if not (0 <= worker_index < num_workers_total):
        raise ValueError(f"worker-index must be in [0,{num_workers_total}), got {worker_index}")
    return [job for i, job in enumerate(jobs) if i % num_workers_total == worker_index]


def make_all_sample_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    indices = list(range(len(dataset)))
    subset = image_models.IndexedDataset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def dataset_view_seed(base_seed: int, dataset_tag: str, batch_counter: int) -> int:
    dataset_offsets = {
        "cifar10": 10_000_000,
        "cifar100": 20_000_000,
        "tinyimagenet": 30_000_000,
        "cinic10": 40_000_000,
    }
    dataset_name, _ = parse_dataset_tag(dataset_tag)
    return int(base_seed + dataset_offsets[dataset_name] + batch_counter * 1009)


def dataset_seed(base_seed: int, dataset_tag: str) -> int:
    dataset_offsets = {
        "cifar10": 10_000_000,
        "cifar100": 20_000_000,
        "tinyimagenet": 30_000_000,
        "cinic10": 40_000_000,
    }
    dataset_name, _ = parse_dataset_tag(dataset_tag)
    return int(base_seed + dataset_offsets[dataset_name])


def splitmix64_np(x: np.ndarray) -> np.ndarray:
    x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return x ^ (x >> np.uint64(31))


def sample_index_unique_stdaug_codes(
    sample_idx: np.ndarray,
    *,
    repeats: int,
    seed: int,
    pad: int = 4,
    allow_hflip: bool = True,
    include_center_view: bool = False,
) -> torch.Tensor:
    max_off = 2 * pad + 1
    transform_count = max_off * max_off * (2 if allow_hflip else 1)
    if repeats > transform_count:
        raise ValueError(
            f"Cannot draw {repeats} unique stdaug views from only {transform_count} crop/flip transforms."
        )
    idx = np.asarray(sample_idx, dtype=np.uint64)[:, None]
    codes = np.arange(transform_count, dtype=np.uint64)[None, :]
    keys = splitmix64_np(
        idx * np.uint64(0xD1B54A32D192ED03)
        + codes * np.uint64(0xABC98388FB8FAC03)
        + np.uint64(seed)
    )
    if include_center_view:
        center_code = pad * max_off + pad  # unflipped center crop.
        if repeats == 1:
            selected = np.full((sample_idx.shape[0], 1), center_code, dtype=np.int64)
            return torch.from_numpy(selected)
        keys[:, center_code] = np.iinfo(np.uint64).max
        rest = repeats - 1
        selected_rest = np.argpartition(keys, kth=rest - 1, axis=1)[:, :rest]
        selected_keys = np.take_along_axis(keys, selected_rest, axis=1)
        order = np.argsort(selected_keys, axis=1, kind="stable")
        selected_rest = np.take_along_axis(selected_rest, order, axis=1).astype(np.int64, copy=False)
        center = np.full((sample_idx.shape[0], 1), center_code, dtype=np.int64)
        return torch.from_numpy(np.concatenate([center, selected_rest], axis=1))

    selected = np.argpartition(keys, kth=repeats - 1, axis=1)[:, :repeats]
    selected_keys = np.take_along_axis(keys, selected, axis=1)
    order = np.argsort(selected_keys, axis=1, kind="stable")
    selected = np.take_along_axis(selected, order, axis=1).astype(np.int64, copy=False)
    return torch.from_numpy(selected)


def load_member_mask(model_dir: Path, n_samples: int) -> np.ndarray:
    split = json.loads((model_dir / "data_split.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split.get("clf_train_idx", []), dtype=np.int64)
    mask = np.zeros(n_samples, dtype=np.bool_)
    train_idx = train_idx[(0 <= train_idx) & (train_idx < n_samples)]
    mask[train_idx] = True
    return mask


def score_batch(
    model: torch.nn.Module,
    x_aug: torch.Tensor,
    y_rep: torch.Tensor,
    *,
    sample_count: int,
    views: int,
    microbatch: int,
) -> tuple[np.ndarray, np.ndarray]:
    amp_enabled = x_aug.device.type == "cuda"
    true_chunks: List[torch.Tensor] = []
    comp_chunks: List[torch.Tensor] = []
    for start in range(0, x_aug.shape[0], microbatch):
        end = min(start + microbatch, x_aug.shape[0])
        with torch.no_grad():
            with torch.autocast(device_type=x_aug.device.type, enabled=amp_enabled):
                logits = model(x_aug[start:end])
            logits = image_models.sanitize_tensor(logits)
            probs = F.softmax(logits, dim=1)
            y_mb = y_rep[start:end]
            true_prob = torch.gather(probs, dim=1, index=y_mb[:, None]).squeeze(1)
            probs.scatter_(1, y_mb[:, None], -1.0)
            max_other_prob = probs.max(dim=1).values
        true_chunks.append(true_prob.float().detach().cpu())
        comp_chunks.append(max_other_prob.float().detach().cpu())
    true_matrix = torch.cat(true_chunks, dim=0).view(sample_count, views).numpy().astype(np.float16)
    comp_matrix = torch.cat(comp_chunks, dim=0).view(sample_count, views).numpy().astype(np.float16)
    return true_matrix, comp_matrix


def write_dataset_meta(
    *,
    out_dir: Path,
    dataset_tag: str,
    dataset_name: str,
    augmentation: str,
    loader: DataLoader,
    views: int,
    seed: int,
    batch_size: int,
    unique_stdaug_views: bool,
    sample_index_views: bool,
    include_center_view: bool,
) -> None:
    meta_path = out_dir / "dataset_meta.npz"
    manifest_path = out_dir / "dataset_manifest.json"
    if meta_path.exists() and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        expected = {
            "dataset_tag": dataset_tag,
            "dataset_name": dataset_name,
            "augmentation": augmentation,
            "views": int(views),
            "seed": int(seed),
            "unique_stdaug_views": bool(unique_stdaug_views),
            "sample_index_views": bool(sample_index_views),
            "include_center_view": bool(include_center_view),
            "batch_size": int(batch_size),
        }
        if all(existing.get(key) == value for key, value in expected.items()):
            return
    sample_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    for _, y_cpu, idx_cpu in loader:
        sample_parts.append(idx_cpu.numpy().astype(np.int64))
        label_parts.append(y_cpu.numpy().astype(np.int64))
    sample_idx = np.concatenate(sample_parts, axis=0)
    class_label = np.concatenate(label_parts, axis=0)
    tmp_path = out_dir / f"dataset_meta.worker{os.getpid()}.tmp.npz"
    np.savez(
        tmp_path,
        sample_idx=sample_idx,
        class_label=class_label,
    )
    os.replace(tmp_path, meta_path)
    manifest = {
        "dataset_tag": dataset_tag,
        "dataset_name": dataset_name,
        "augmentation": augmentation,
        "n_samples": int(len(sample_idx)),
        "views": int(views),
        "seed": int(seed),
        "unique_stdaug_views": bool(unique_stdaug_views),
        "sample_index_views": bool(sample_index_views),
        "include_center_view": bool(include_center_view),
        "batch_size": int(batch_size),
        "view_seed_rule": (
            "sample-index splitmix64 transform ordering"
            if sample_index_views
            else "seed + dataset_offset + batch_counter * 1009"
        ),
        "arrays": {
            "sample_idx": {"shape": list(sample_idx.shape), "dtype": str(sample_idx.dtype)},
            "class_label": {"shape": list(class_label.shape), "dtype": str(class_label.dtype)},
        },
    }
    tmp_json = out_dir / f"dataset_manifest.worker{os.getpid()}.tmp.json"
    tmp_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_json, manifest_path)


def checkpoint_fingerprint(model_dir: Path) -> Dict[str, Dict[str, int]]:
    fingerprint: Dict[str, Dict[str, int]] = {}
    for name in ("FE.pth", "CF.pth", "data_split.json"):
        path = model_dir / name
        stat = path.stat()
        fingerprint[name] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return fingerprint


def cache_is_current(
    summary_path: Path,
    *,
    job: Job,
    args: argparse.Namespace,
) -> bool:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "dataset_tag": job.dataset_tag,
        "dataset_name": job.dataset_name,
        "augmentation": job.augmentation,
        "model_name": job.model_name,
        "views": int(args.views),
        "unique_stdaug_views": bool(args.unique_stdaug_views),
        "sample_index_views": bool(args.sample_index_views),
        "include_center_view": bool(args.include_center_view),
        "seed": int(args.seed),
        "checkpoint_fingerprint": checkpoint_fingerprint(job.model_dir),
    }
    return all(summary.get(key) == value for key, value in expected.items())


def export_job(
    *,
    job: Job,
    args: argparse.Namespace,
    device: torch.device,
    dataset: torch.utils.data.Dataset,
    loader: DataLoader,
    batch_size: int,
) -> Dict[str, Any]:
    dataset_out = args.output_root / job.dataset_tag
    model_out_dir = dataset_out / "models"
    model_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_out_dir / f"{job.model_name}.npz"
    done_path = model_out_dir / f"{job.model_name}.done"
    summary_path = model_out_dir / f"{job.model_name}.json"
    if (
        done_path.exists()
        and out_path.exists()
        and summary_path.exists()
        and not args.force
        and cache_is_current(summary_path, job=job, args=args)
    ):
        return {
            "dataset_tag": job.dataset_tag,
            "model_name": job.model_name,
            "status": "skipped_done",
            "path": str(out_path),
        }

    started = time.time()
    model = image_models.load_checkpoint(job.model_dir, device=device)
    mean, std = image_models.dataset_stats(job.dataset_name)
    sample_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    true_parts: List[np.ndarray] = []
    comp_parts: List[np.ndarray] = []
    total_batches = len(loader)
    progress_every = max(1, total_batches // 5)
    for batch_counter, (x_cpu, y_cpu, idx_cpu) in enumerate(loader):
        x = x_cpu.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        y = y_cpu.to(device=device, non_blocking=True)
        view_codes = None
        if job.augmentation != "stdaug" and (
            args.sample_index_views or args.unique_stdaug_views or args.include_center_view
        ):
            raise ValueError(
                "sample-index/unique/center-view options are only valid for stdaug"
            )
        if args.sample_index_views:
            if not args.unique_stdaug_views:
                raise ValueError("--sample-index-views currently requires --unique-stdaug-views")
            view_codes = sample_index_unique_stdaug_codes(
                idx_cpu.numpy(),
                repeats=args.views,
                seed=dataset_seed(args.seed, job.dataset_tag),
                allow_hflip=True,
                include_center_view=bool(args.include_center_view),
            )
        elif args.include_center_view:
            raise ValueError("--include-center-view requires --sample-index-views --unique-stdaug-views")
        if job.augmentation == "stdaug":
            x_aug = image_models.augment_std_views(
                x,
                repeats=args.views,
                seed=dataset_view_seed(args.seed, job.dataset_tag, batch_counter),
                view_codes=view_codes,
            )
        else:
            x_aug = image_models.augment_noaug_views(
                x,
                repeats=args.views,
                seed=dataset_view_seed(args.seed, job.dataset_tag, batch_counter),
                mean_values=mean,
                std_values=std,
            )
        y_rep = y.repeat_interleave(args.views)
        true_matrix, comp_matrix = score_batch(
            model,
            x_aug,
            y_rep,
            sample_count=int(x_cpu.shape[0]),
            views=args.views,
            microbatch=args.microbatch,
        )
        sample_parts.append(idx_cpu.numpy().astype(np.int64))
        label_parts.append(y_cpu.numpy().astype(np.int64))
        true_parts.append(true_matrix)
        comp_parts.append(comp_matrix)
        done = batch_counter + 1
        if done == 1 or done == total_batches or done % progress_every == 0:
            print(
                f"[{job.dataset_tag}:{job.model_name}] batch {done}/{total_batches} "
                f"samples={min(done * batch_size, len(loader.dataset))}/{len(loader.dataset)}",
                flush=True,
            )
        del x, y, x_aug, y_rep

    sample_idx = np.concatenate(sample_parts, axis=0).astype(np.int64)
    class_label = np.concatenate(label_parts, axis=0).astype(np.int64)
    true_prob = np.concatenate(true_parts, axis=0).astype(np.float16)
    max_other_prob = np.concatenate(comp_parts, axis=0).astype(np.float16)
    member_mask_full = load_member_mask(job.model_dir, len(dataset))
    member = member_mask_full[sample_idx].astype(np.bool_)

    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        sample_idx=sample_idx,
        class_label=class_label,
        member=member,
        true_prob=true_prob,
        max_other_prob=max_other_prob,
    )
    os.replace(tmp_path, out_path)
    runtime_sec = time.time() - started
    summary = {
        "dataset_tag": job.dataset_tag,
        "dataset_name": job.dataset_name,
        "augmentation": job.augmentation,
        "model_name": job.model_name,
        "model_dir": str(job.model_dir),
        "output_npz": str(out_path),
        "views": int(args.views),
        "unique_stdaug_views": bool(args.unique_stdaug_views),
        "sample_index_views": bool(args.sample_index_views),
        "include_center_view": bool(args.include_center_view),
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(batch_size),
        "microbatch": int(args.microbatch),
        "n_samples": int(len(sample_idx)),
        "member_count": int(member.sum()),
        "nonmember_count": int((~member).sum()),
        "arrays": {
            "sample_idx": {"shape": list(sample_idx.shape), "dtype": str(sample_idx.dtype)},
            "class_label": {"shape": list(class_label.shape), "dtype": str(class_label.dtype)},
            "member": {"shape": list(member.shape), "dtype": str(member.dtype)},
            "true_prob": {"shape": list(true_prob.shape), "dtype": str(true_prob.dtype)},
            "max_other_prob": {"shape": list(max_other_prob.shape), "dtype": str(max_other_prob.dtype)},
        },
        "runtime_sec": float(runtime_sec),
        "updated_at": time.time(),
        "checkpoint_fingerprint": checkpoint_fingerprint(job.model_dir),
    }
    tmp_summary = summary_path.with_suffix(".tmp.json")
    tmp_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp_summary, summary_path)
    done_path.write_text("ok\n", encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset_tag": job.dataset_tag,
        "model_name": job.model_name,
        "status": "exported",
        "runtime_sec": float(runtime_sec),
        "path": str(out_path),
    }


def write_root_manifest(args: argparse.Namespace, all_jobs: Sequence[Job]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": f"model-first {int(args.views)}-view true/competitor probability evidence bank",
        "notes": [
            "Inference/export only; no training.",
            "Each model is forwarded once on all samples with deterministic shared views.",
            "Per-model files store true_prob and max_other_prob, enough for probability and margin CMS.",
            "No full logits are stored.",
        ],
        "output_root": str(args.output_root),
        "views": int(args.views),
        "unique_stdaug_views": bool(args.unique_stdaug_views),
        "sample_index_views": bool(args.sample_index_views),
        "include_center_view": bool(args.include_center_view),
        "seed": int(args.seed),
        "datasets": parse_csv(args.datasets),
        "job_counts": {
            dataset_tag: int(sum(1 for job in all_jobs if job.dataset_tag == dataset_tag))
            for dataset_tag in parse_csv(args.datasets)
        },
        "total_jobs": int(len(all_jobs)),
    }
    path = args.output_root / "manifest.json"
    tmp = args.output_root / f"manifest.worker{int(args.worker_index):02d}.{os.getpid()}.tmp.json"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    set_perf_flags()
    all_jobs = list_jobs(args)
    write_root_manifest(args, all_jobs)
    jobs = split_for_worker(all_jobs, args.worker_index, args.num_workers_total)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"[worker {args.worker_index}/{args.num_workers_total}] jobs={len(jobs)} "
        f"device={device} output={args.output_root}",
        flush=True,
    )
    datasets: Dict[str, torch.utils.data.Dataset] = {}
    loaders: Dict[str, DataLoader] = {}
    batch_sizes: Dict[str, int] = {}
    completed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    started_all = time.time()
    for job in jobs:
        try:
            if job.dataset_tag not in datasets:
                dataset, _ = image_models.load_dataset(
                    job.dataset_name, args.data_root, augmentation=None
                )
                batch_size = dataset_batch_size(job.dataset_tag, args)
                loader = make_all_sample_loader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=args.num_loader_workers,
                )
                datasets[job.dataset_tag] = dataset
                loaders[job.dataset_tag] = loader
                batch_sizes[job.dataset_tag] = batch_size
                dataset_out = args.output_root / job.dataset_tag
                dataset_out.mkdir(parents=True, exist_ok=True)
                write_dataset_meta(
                    out_dir=dataset_out,
                    dataset_tag=job.dataset_tag,
                    dataset_name=job.dataset_name,
                    augmentation=job.augmentation,
                    loader=loader,
                    views=args.views,
                    seed=args.seed,
                    batch_size=batch_size,
                    unique_stdaug_views=bool(args.unique_stdaug_views),
                    sample_index_views=bool(args.sample_index_views),
                    include_center_view=bool(args.include_center_view),
                )
            print(f"[start] {job.dataset_tag}/{job.model_name}", flush=True)
            result = export_job(
                job=job,
                args=args,
                device=device,
                dataset=datasets[job.dataset_tag],
                loader=loaders[job.dataset_tag],
                batch_size=batch_sizes[job.dataset_tag],
            )
            completed.append(result)
            print(f"[done] {job.dataset_tag}/{job.model_name} {result}", flush=True)
        except Exception as exc:  # keep the worker alive for the remaining queue
            failed.append(
                {
                    "dataset_tag": job.dataset_tag,
                    "model_name": job.model_name,
                    "error": repr(exc),
                }
            )
            print(f"[error] {job.dataset_tag}/{job.model_name}: {exc!r}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    status = {
        "worker_index": int(args.worker_index),
        "num_workers_total": int(args.num_workers_total),
        "device": str(device),
        "completed": completed,
        "failed": failed,
        "runtime_sec": float(time.time() - started_all),
        "updated_at": time.time(),
    }
    status_path = args.output_root / f"worker_{args.worker_index:02d}_status.json"
    tmp_status = status_path.with_suffix(".tmp.json")
    tmp_status.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp_status, status_path)
    if failed:
        raise SystemExit(f"Worker {args.worker_index} finished with {len(failed)} failures")


if __name__ == "__main__":
    main()
