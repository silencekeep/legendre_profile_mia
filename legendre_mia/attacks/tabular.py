from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .core import (
    anchor_tail_features,
    apply_standardizer,
    balanced_subsample,
    exact_report_metrics,
    fit_standardizer,
    GatedImageReadout,
    psi_table,
)


@dataclass
class TabularCache:
    dataset: str
    dataset_root: Path
    sample_idx: np.ndarray
    class_label: np.ndarray
    target_prob: np.ndarray
    target_member: np.ndarray
    shadow_prob: np.ndarray
    shadow_member: np.ndarray


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        return {key: np.asarray(item[key]) for key in item.files}


def _models_dir(cache_root: Path, dataset: str) -> Path:
    candidates = [
        Path(cache_root) / dataset / "models",
        Path(cache_root) / "models",
    ]
    for candidate in candidates:
        if (candidate / "target.npz").is_file():
            return candidate
    raise FileNotFoundError(
        f"{dataset}: target output not found under {cache_root}"
    )


def load_tabular_cache(cache_root: Path, dataset: str, count: int = 256) -> TabularCache:
    model_dir = _models_dir(cache_root, dataset)
    target = _load_npz(model_dir / "target.npz")
    target_prob = target["true_prob"].astype(np.float32).reshape(-1)
    target_member = target["member"].astype(bool)
    sample_idx = target.get(
        "sample_idx", np.arange(target_member.size, dtype=np.int64)
    ).astype(np.int64)
    class_label = target["class_label"].astype(np.int64)
    probabilities = []
    memberships = []
    for shadow_id in range(int(count)):
        path = model_dir / f"shadow_{shadow_id:03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        shadow = _load_npz(path)
        probabilities.append(shadow["true_prob"].astype(np.float32).reshape(-1))
        memberships.append(shadow["member"].astype(bool))
    return TabularCache(
        dataset=dataset,
        dataset_root=Path(cache_root) / dataset,
        sample_idx=sample_idx,
        class_label=class_label,
        target_prob=target_prob,
        target_member=target_member,
        shadow_prob=np.stack(probabilities, axis=1),
        shadow_member=np.stack(memberships, axis=1),
    )


def scalar_legendre_table(probability: np.ndarray, k: int) -> np.ndarray:
    return psi_table(np.asarray(probability, dtype=np.float64), int(k)).T


def build_features(
    target_probability: np.ndarray,
    shadow_probability: np.ndarray,
    reference_mask: np.ndarray,
    *,
    budget: int,
    k: int,
    order: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    selected, counts = select_reference_mask(reference_mask, int(budget), order)
    c_target = scalar_legendre_table(target_probability, int(k))
    c_reference = np.zeros_like(c_target)
    for shadow_id in range(shadow_probability.shape[1]):
        rows = selected[:, shadow_id]
        if np.any(rows):
            c_reference[rows] += scalar_legendre_table(
                shadow_probability[rows, shadow_id], int(k)
            )
    c_reference /= np.maximum(counts.astype(np.float64)[:, None], 1.0)
    features = anchor_tail_features(
        c_target,
        c_reference,
        k,
    )
    return features, {
        "ref_count_min": float(counts.min()),
        "ref_count_mean": float(counts.mean()),
        "ref_count_max": float(counts.max()),
    }


def select_reference_mask(
    reference_mask: np.ndarray,
    budget: int,
    order: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    counts_available = reference_mask.sum(axis=1).astype(np.int16)
    if int(budget) >= int(counts_available.max()):
        return reference_mask.copy(), counts_available
    selected = np.zeros_like(reference_mask, dtype=bool)
    counts = np.zeros(reference_mask.shape[0], dtype=np.int16)
    for shadow_id in order:
        take = reference_mask[:, shadow_id] & (counts < int(budget))
        selected[take, shadow_id] = True
        counts[take] += 1
        if int(counts.min()) >= int(budget):
            break
    return selected, counts


def fit_tabular_readout_gated(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    dim: int,
    hidden: int,
    dropout: float,
    weight_decay: float,
    learning_rate: float,
    gate_l1: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = GatedImageReadout(int(dim), int(hidden), float(dropout)).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    y_bool = np.asarray(y_train, dtype=bool)
    pos_weight = float((~y_bool).sum() / max(1, y_bool.sum()))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=dev)
    )
    x_train_t = torch.tensor(np.asarray(x_train, dtype=np.float32), device=dev)
    y_train_t = torch.tensor(np.asarray(y_train, dtype=np.float32), device=dev)
    x_eval_t = torch.tensor(np.asarray(x_eval, dtype=np.float32), device=dev)
    rng = np.random.default_rng(int(seed))
    n_rows = x_train.shape[0]

    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(n_rows)
        for start in range(0, n_rows, int(batch_size)):
            index = order[start : start + int(batch_size)]
            xb = x_train_t[index]
            yb = y_train_t[index]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if float(gate_l1) > 0.0:
                loss = loss + float(gate_l1) * model.gates().mean()
            loss.backward()
            optimizer.step()

    # Final iterate.
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, x_eval.shape[0], 131072):
            xb = torch.tensor(x_eval[start : start + 131072], dtype=torch.float32, device=dev)
            chunks.append(model(xb).detach().cpu().numpy())
    scores = np.concatenate(chunks).astype(np.float64)
    gates = model.gates().detach().cpu().numpy().astype(np.float64)
    metadata = {
        "epochs_trained": float(epochs),
        "gate_mean": float(gates.mean()),
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
    }
    return scores, metadata


def run_dataset(
    data: TabularCache,
    attack_config: Mapping[str, Any],
    *,
    device: str,
    score_dir: Path,
    readout_dir: Path,
) -> Dict[str, Any]:
    reference_pool = int(attack_config["reference_pool"])
    shadow_count = int(attack_config.get("shadow_models", 8))
    total = int(reference_pool) + int(shadow_count)

    if data.shadow_prob.shape[1] != total:
        raise ValueError(
            f"{data.dataset}: loaded {data.shadow_prob.shape[1]} references, "
            f"expected {total} ({shadow_count} shadows + {reference_pool} references)"
        )

    # First shadow_count models = supervision shadows, rest = references.
    shadow_prob = data.shadow_prob
    shadow_member = data.shadow_member
    reference_order = np.random.default_rng(
        int(attack_config["reference_seed"])
    ).permutation(reference_pool + shadow_count)

    k = int(attack_config["legendre_k"])

    # Build features for supervision shadows.
    train_features: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []

    for shadow_id in range(shadow_count):
        reference_mask = ~shadow_member.copy()
        # Exclude shadow models from reference set.
        for sid in range(shadow_count):
            reference_mask[:, sid] = False
        features, _ = build_features(
            shadow_prob[:, shadow_id],
            shadow_prob,
            reference_mask,
            budget=reference_pool,
            k=k,
            order=reference_order,
        )
        labels = shadow_member[:, shadow_id]
        features, labels = balanced_subsample(
            features,
            labels,
            int(attack_config["max_train_rows_per_pseudo"]),
            int(attack_config["seed"]) + shadow_id,
        )
        train_features.append(features)
        train_labels.append(labels)

    # Target evaluation: use all reference models (OUT from all of them).
    reference_mask = np.zeros_like(shadow_member, dtype=bool)
    for sid in range(shadow_count, total):
        reference_mask[:, sid] = True
    x_eval, target_reference = build_features(
        data.target_prob,
        shadow_prob,
        reference_mask,
        budget=reference_pool,
        k=k,
        order=reference_order,
    )
    expected_references = int(attack_config["out_references_per_sample"])
    if any(
        target_reference[key] != expected_references
        for key in ("ref_count_min", "ref_count_mean", "ref_count_max")
    ):
        raise RuntimeError(
            f"{data.dataset}: target OUT-reference audit failed: {target_reference}"
        )

    x_train = np.concatenate(train_features, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    mean, std = fit_standardizer(x_train)
    x_train_std = apply_standardizer(x_train, mean, std)
    x_eval_std = apply_standardizer(x_eval, mean, std)

    scores, metadata = fit_tabular_readout_gated(
        x_train_std,
        y_train,
        x_eval_std,
        dim=x_train.shape[1],
        hidden=int(attack_config.get("hidden", 8)),
        dropout=float(attack_config.get("dropout", 0.05)),
        weight_decay=float(attack_config.get("weight_decay", 0.01)),
        learning_rate=float(attack_config["learning_rate"]),
        gate_l1=float(attack_config.get("gate_l1", 0.0001)),
        epochs=int(attack_config["epochs"]),
        batch_size=int(attack_config["batch_size"]),
        seed=int(attack_config["seed"]),
        device=device,
    )

    report_metrics = exact_report_metrics(data.target_member, scores)

    score_dir.mkdir(parents=True, exist_ok=True)
    readout_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"{data.dataset}_Ours-Legendre.npz"
    np.savez_compressed(
        score_path,
        dataset=np.asarray(data.dataset),
        method=np.asarray("Ours-Legendre"),
        scores=scores.astype(np.float32),
        labels=data.target_member.astype(np.int8),
        sample_idx=data.sample_idx.astype(np.int64),
        source=np.asarray("legendre_mia"),
        score_sign=np.asarray(1),
        orientation_source=np.asarray("larger_is_member"),
    )
    readout_path = readout_dir / f"{data.dataset}_readout.pt"
    torch.save(
        {
            "state_dict": {},
            "standardize_mean": mean,
            "standardize_std": std,
            "attack_config": dict(attack_config),
            "dataset": data.dataset,
        },
        readout_path,
    )
    return {
        "modality": "tabular",
        "dataset": data.dataset,
        "budget": "fair256",
        "method": "Ours-Legendre",
        "score_path": str(score_path.resolve()),
        "readout_path": str(readout_path.resolve()),
        "samples": int(data.target_member.size),
        "positive_samples": int(data.target_member.sum()),
        "negative_samples": int((~data.target_member).sum()),
        "reference_pool": reference_pool,
        "reference_count_min": target_reference["ref_count_min"],
        "reference_count_mean": target_reference["ref_count_mean"],
        "reference_count_max": target_reference["ref_count_max"],
        "k": k,
        "score_sign": 1,
        "AUC": float(report_metrics["AUC"]),
        "TPR@0FP": float(report_metrics["TPR@0FP"]),
        "TPR@1%FPR": float(report_metrics["TPR@1%FPR"]),
        "TPR@0.1%FPR": float(report_metrics["TPR@0.1%FPR"]),
        "TPR@0.01%FPR": float(report_metrics["TPR@0.01%FPR"]),
        "TPR@0.001%FPR": float(report_metrics["TPR@0.001%FPR"]),
        **metadata,
    }


def run_tabular_attacks(
    config: Mapping[str, Any],
    *,
    cache_root: Path,
    output_root: Path,
    datasets: Sequence[str] | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    tabular_config = config["tabular"]
    attack_config = tabular_config["attack"]
    torch.set_num_threads(int(attack_config["num_threads"]))
    try:
        torch.set_num_interop_threads(int(attack_config["num_interop_threads"]))
    except RuntimeError as error:
        if torch.get_num_interop_threads() != int(attack_config["num_interop_threads"]):
            raise RuntimeError(
                "Tabular bitwise replay requires a fresh process with "
                f"num_interop_threads={attack_config['num_interop_threads']}"
            ) from error
    selected_datasets = list(datasets or tabular_config["datasets"])
    score_dir = Path(output_root) / "scores" / "tabular"
    readout_dir = Path(output_root) / "readouts" / "tabular"
    rows: List[Dict[str, Any]] = []
    for dataset in selected_datasets:
        print(f"[tabular-attack] {dataset}", flush=True)
        data = load_tabular_cache(
            Path(cache_root),
            dataset,
            int(attack_config["reference_pool"]) + int(attack_config.get("shadow_models", 8)),
        )
        rows.append(
            run_dataset(
                data,
                attack_config,
                device=str(device or attack_config["device"]),
                score_dir=score_dir,
                readout_dir=readout_dir,
            )
        )
        pd.DataFrame(rows).to_csv(
            Path(output_root) / "tabular_ours_metrics.csv", index=False
        )
        del data
        gc.collect()
    result = pd.DataFrame(rows)
    result.to_csv(Path(output_root) / "tabular_ours_metrics.csv", index=False)
    (Path(output_root) / "tabular_attack_config.json").write_text(
        json.dumps(
            {
                "cache_root": str(Path(cache_root).resolve()),
                "datasets": selected_datasets,
                "attack": attack_config,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result
