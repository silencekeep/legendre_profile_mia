from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .core import (
    anchor_tail_features,
    exact_report_metrics,
    fit_image_readout,
    legendre_coefficients,
    standardize_image_triplet,
)

@dataclass
class ImageCase:
    pseudo_id: int
    c_target: np.ndarray
    c_reference: np.ndarray
    labels: np.ndarray


@dataclass
class ImageCell:
    dataset: str
    budget: str
    sample_idx: np.ndarray
    target_c: np.ndarray
    target_reference_c: np.ndarray
    target_labels: np.ndarray
    pseudo_cases: List[ImageCase]
    reference_count: np.ndarray
    views: int


def resolve_model_dir(cache_root: Path, dataset: str) -> Path:
    candidates = [
        Path(cache_root) / dataset / "models",
        Path(cache_root) / "models",
        Path(cache_root),
    ]
    for candidate in candidates:
        if (candidate / "target.npz").is_file():
            return candidate
    raise FileNotFoundError(
        f"No target.npz for {dataset}; checked: {', '.join(str(path) for path in candidates)}"
    )


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        return {key: np.asarray(item[key]) for key in item.files}


def load_image_cell(
    *,
    dataset: str,
    budget: str,
    cache_root: Path,
    shadow_count: int,
    reference_count: int,
    views: int,
    pseudo_ids: Sequence[int],
    k: int,
) -> ImageCell:
    model_dir = resolve_model_dir(cache_root, dataset)
    target = _load_npz(model_dir / "target.npz")

    target_views = target["true_prob"]
    if target_views.ndim != 2 or target_views.shape[1] < int(views):
        raise ValueError(
            f"{dataset}/{budget}: target cache has "
            f"{target_views.shape}, needs {views} views"
        )

    # View-average then Legendre project (paper Section III-B).
    target_bar = target_views[:, : int(views)].astype(np.float64).mean(axis=1)
    target_c = legendre_coefficients(target_bar[:, None], int(k))

    target_labels = target["member"].astype(bool)
    sample_idx = target.get(
        "sample_idx",
        np.arange(target_labels.size, dtype=np.int64),
    ).astype(np.int64)

    n_samples = target_labels.size

    total = int(shadow_count) + int(reference_count)

    # Accumulate transformed coefficients across OUT reference models.
    # Supervision shadows (0..shadow_count-1) are treated as references
    # for OUT accumulation but also provide pseudo-target features.
    out_coeff_sum = np.zeros(
        (n_samples, int(k) + 1),
        dtype=np.float64,
    )
    out_count = np.zeros(n_samples, dtype=np.float64)

    wanted = {int(value) for value in pseudo_ids}
    pseudo_raw: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for model_id in range(total):
        model_path = model_dir / f"shadow_{model_id:03d}.npz"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        model_data = _load_npz(model_path)
        probability = model_data["true_prob"]
        member = model_data["member"].astype(bool)

        if probability.ndim != 2 or probability.shape[1] < int(views):
            raise ValueError(
                f"{model_path}: probability shape {probability.shape}, "
                f"needs {views} views"
            )
        if probability.shape[0] != n_samples or member.shape != target_labels.shape:
            raise ValueError(
                f"{model_path}: cache shape mismatch "
                f"probability={probability.shape}, member={member.shape}"
            )

        # Average views before Legendre projection.
        p_bar = probability[:, : int(views)].astype(np.float64).mean(axis=1)
        coeff = legendre_coefficients(p_bar[:, None], int(k))

        out = ~member
        out_coeff_sum[out] += coeff[out]
        out_count[out] += 1.0

        if model_id in wanted:
            pseudo_raw[model_id] = (coeff, member)

    if set(pseudo_raw) != wanted:
        raise RuntimeError(
            f"{dataset}/{budget}: missing pseudo shadows "
            f"{sorted(wanted - set(pseudo_raw))}"
        )

    if np.any(out_count <= 0):
        raise RuntimeError(
            f"{dataset}/{budget}: some candidates have no OUT references"
        )

    target_reference_c = out_coeff_sum / out_count[:, None]

    pseudo_cases: List[ImageCase] = []

    for pseudo_id in [int(value) for value in pseudo_ids]:
        pseudo_c, member = pseudo_raw[pseudo_id]
        out = ~member

        # Leave the pseudo-target model out of its reference set whenever
        # that model is an OUT model for the candidate.
        loo_sum = out_coeff_sum - np.where(
            out[:, None],
            pseudo_c,
            0.0,
        )
        loo_count = out_count - out.astype(np.float64)

        if np.any(loo_count <= 0):
            raise RuntimeError(
                f"{dataset}/{budget}: pseudo {pseudo_id} has empty "
                "leave-one-out reference set"
            )

        pseudo_cases.append(
            ImageCase(
                pseudo_id=pseudo_id,
                c_target=pseudo_c,
                c_reference=loo_sum / loo_count[:, None],
                labels=member,
            )
        )

    return ImageCell(
        dataset=dataset,
        budget=budget,
        sample_idx=sample_idx,
        target_c=target_c,
        target_reference_c=target_reference_c,
        target_labels=target_labels,
        pseudo_cases=pseudo_cases,
        reference_count=out_count,
        views=int(views),
    )


def run_cell(
    cell: ImageCell,
    attack_config: Mapping[str, Any],
    *,
    device: str,
    score_dir: Path,
    readout_dir: Path,
) -> Dict[str, Any]:
    k = int(attack_config["legendre_k"])

    # All pseudo shadows are used for training (paper: 8 shadows, all for fitting).
    train_target_c = np.concatenate(
        [case.c_target for case in cell.pseudo_cases],
        axis=0,
    )
    train_reference_c = np.concatenate(
        [case.c_reference for case in cell.pseudo_cases],
        axis=0,
    )
    train_labels = np.concatenate(
        [case.labels for case in cell.pseudo_cases],
        axis=0,
    ).astype(bool)

    x_train_raw = anchor_tail_features(
        train_target_c,
        train_reference_c,
        k,
    )
    x_eval_raw = anchor_tail_features(
        cell.target_c,
        cell.target_reference_c,
        k,
    )
    mean = x_train_raw.mean(axis=0, keepdims=True).astype(np.float64)
    std = x_train_raw.std(axis=0, keepdims=True).astype(np.float64)
    std = np.where(std < 1e-6, 1.0, std)

    x_train_std = ((x_train_raw.astype(np.float64) - mean) / std).astype(np.float32)
    x_eval_std = ((x_eval_raw.astype(np.float64) - mean) / std).astype(np.float32)

    readout = fit_image_readout(
        x_train_std,
        train_labels,
        None,
        None,
        x_eval_std,
        hidden=int(attack_config["hidden"]),
        dropout=float(attack_config["dropout"]),
        weight_decay=float(attack_config["weight_decay"]),
        learning_rate=float(attack_config["learning_rate"]),
        feature_dropout=float(attack_config.get("feature_dropout", 0.0)),
        input_noise=float(attack_config.get("input_noise", 0.0)),
        gate_l1=float(attack_config["gate_l1"]),
        objective=str(attack_config.get("training_objective", "tail01")),
        epochs=int(attack_config["epochs"]),
        patience=int(attack_config.get("patience", 30)),
        batch_size=int(attack_config["batch_size"]),
        seed=int(attack_config["seed"]),
        device=device,
    )
    scores = readout.scores
    score_dir.mkdir(parents=True, exist_ok=True)
    readout_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"{cell.dataset}_{cell.budget}_Ours_Legendre_K4.npz"
    np.savez_compressed(
        score_path,
        labels=cell.target_labels.astype(np.int8),
        scores=scores.astype(np.float32),
        sample_idx=cell.sample_idx.astype(np.int64),
        method=np.asarray("Ours_Legendre_K4"),
        dataset=np.asarray(cell.dataset),
        budget=np.asarray(cell.budget),
        source=np.asarray("legendre_mia"),
        score_sign=np.asarray(1),
        orientation_source=np.asarray("larger_is_member"),
    )
    readout_path = readout_dir / f"{cell.dataset}_{cell.budget}_readout.pt"
    torch.save(
        {
            "state_dict": readout.state_dict,
            "standardize_mean": mean.squeeze(0).astype(np.float32),
            "standardize_std": std.squeeze(0).astype(np.float32),
            "attack_config": dict(attack_config),
            "dataset": cell.dataset,
            "budget": cell.budget,
        },
        readout_path,
    )
    exact_metrics = exact_report_metrics(cell.target_labels, scores)
    return {
        "modality": "image",
        "dataset": cell.dataset,
        "budget": cell.budget,
        "method": "Ours_Legendre_K4",
        "score_path": str(score_path.resolve()),
        "readout_path": str(readout_path.resolve()),
        "samples": int(cell.target_labels.size),
        "positive_samples": int(cell.target_labels.sum()),
        "negative_samples": int((~cell.target_labels).sum()),
        "views": int(cell.views),
        "reference_count_min": float(cell.reference_count.min()),
        "reference_count_mean": float(cell.reference_count.mean()),
        "reference_count_max": float(cell.reference_count.max()),
        "AUC": float(exact_metrics["AUC"]),
        "TPR@0FP": float(exact_metrics["TPR@0FP"]),
        "TPR@1%FPR": float(exact_metrics["TPR@1%FPR"]),
        "TPR@0.1%FPR": float(exact_metrics["TPR@0.1%FPR"]),
        "TPR@0.01%FPR": float(exact_metrics["TPR@0.01%FPR"]),
        "TPR@0.001%FPR": float(exact_metrics["TPR@0.001%FPR"]),
        **readout.metadata,
    }


def run_image_attacks(
    config: Mapping[str, Any],
    *,
    cache_roots: Mapping[str, Path],
    output_root: Path,
    datasets: Sequence[str] | None = None,
    budgets: Sequence[str] = ("large", "small"),
    device: str | None = None,
) -> pd.DataFrame:
    image_config = config["image"]
    attack_config = image_config["attack"]
    selected_datasets = list(datasets or image_config["datasets"])
    selected_budgets = list(budgets)
    score_dir = Path(output_root) / "scores" / "image"
    readout_dir = Path(output_root) / "readouts" / "image"
    rows: List[Dict[str, Any]] = []
    for dataset in selected_datasets:
        for budget in selected_budgets:
            budget_config = image_config["budgets"][budget]
            print(f"[image-attack] {dataset}/{budget}", flush=True)
            cell = load_image_cell(
                dataset=dataset,
                budget=budget,
                cache_root=Path(cache_roots[dataset]),
                shadow_count=int(attack_config["shadow_models"]),
                reference_count=int(budget_config["reference_models"]),
                views=int(budget_config["views"]),
                pseudo_ids=attack_config["pseudo_ids"],
                k=int(attack_config["legendre_k"]),
            )
            expected_refs = int(budget_config["out_references_per_sample"])
            if not np.all(cell.reference_count == expected_refs):
                raise RuntimeError(
                    f"{dataset}/{budget}: OUT count range "
                    f"{cell.reference_count.min()}..{cell.reference_count.max()}, expected {expected_refs}"
                )
            rows.append(
                run_cell(
                    cell,
                    attack_config,
                    device=str(device or attack_config["device"]),
                    score_dir=score_dir,
                    readout_dir=readout_dir,
                )
            )
            pd.DataFrame(rows).to_csv(
                Path(output_root) / "image_ours_metrics.csv", index=False
            )
            del cell
            gc.collect()

    result = pd.DataFrame(rows)
    result.to_csv(Path(output_root) / "image_ours_metrics.csv", index=False)
    (Path(output_root) / "image_attack_config.json").write_text(
        json.dumps(
            {
                "cache_roots": {key: str(value) for key, value in cache_roots.items()},
                "datasets": selected_datasets,
                "budgets": selected_budgets,
                "attack": attack_config,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result
