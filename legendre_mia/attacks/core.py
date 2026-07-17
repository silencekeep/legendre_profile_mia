from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_auc_score, roc_curve


EPS = 1e-12
FPR_POINTS = (0.01, 0.001, 0.0001, 0.00001)
FPR_NAMES = ("TPR@1%FPR", "TPR@0.1%FPR", "TPR@0.01%FPR", "TPR@0.001%FPR")


def psi_table(points: np.ndarray, k: int) -> np.ndarray:
    """Integrated shifted-Legendre basis Psi_0..Psi_k in float64."""
    if int(k) < 0:
        raise ValueError("k must be non-negative")
    p = np.clip(np.asarray(points, dtype=np.float64), 0.0, 1.0)
    x = 2.0 * p - 1.0
    legendre = np.empty((int(k) + 2,) + p.shape, dtype=np.float64)
    legendre[0] = 1.0
    legendre[1] = x
    for order in range(1, int(k) + 1):
        legendre[order + 1] = (
            (2 * order + 1) * x * legendre[order] - order * legendre[order - 1]
        ) / (order + 1)

    psi = np.empty((int(k) + 1,) + p.shape, dtype=np.float64)
    psi[0] = p
    for order in range(1, int(k) + 1):
        low = ((-1.0) ** (order + 1)) - ((-1.0) ** (order - 1))
        psi[order] = (
            (legendre[order + 1] - legendre[order - 1]) - low
        ) / (2.0 * (2 * order + 1))
    return psi


def legendre_coefficients(points: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return psi_table(arr, int(k)).mean(axis=2).T.astype(np.float64, copy=False)

def anchor_tail_features(
    c_target: np.ndarray,
    c_reference: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Construct the nonredundant Anchor–Tail feature vector.

    K = 0: [Q0, B0]
    K = 1: [Q0, B0, Q1, B1]
    K >= 2: [Q0, B0, Q1, B1, Gamma2, ..., GammaK]

    Parameters
    ----------
    c_target:
        [num_samples, k+1] Legendre coefficients of the target model.
    c_reference:
        [num_samples, k+1] Legendre coefficients of the reference model.
    k:
        Legendre order.
    """
    order = int(k)
    if order < 0:
        raise ValueError("k must be non-negative")

    q = np.asarray(c_target, dtype=np.float64)
    b = np.asarray(c_reference, dtype=np.float64)

    if q.ndim != 2 or b.ndim != 2:
        raise ValueError("coefficient arrays must be two-dimensional")
    if q.shape != b.shape:
        raise ValueError(
            f"target/reference coefficient shape mismatch: {q.shape} vs {b.shape}"
        )
    if q.shape[1] < order + 1:
        raise ValueError(
            f"need coefficients through order {order}, got shape {q.shape}"
        )

    blocks = [
        q[:, 0:1],  # Q0
        b[:, 0:1],  # B0
    ]

    if order >= 1:
        blocks.extend(
            [
                q[:, 1:2],  # Q1
                b[:, 1:2],  # B1
            ]
        )

    if order >= 2:
        gamma = q - b
        blocks.append(gamma[:, 2 : order + 1])

    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def legendre_features(
    p_target: np.ndarray,
    p_reference: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Build Anchor–Tail features from stabilized model responses.

    p_target:
        [num_samples] or [num_samples, 1].
        Each row must already be the view-averaged response of the
        query model.

    p_reference:
        [num_samples, num_reference_models].
        Each entry must already be the view-averaged response of one
        reference model.
    """
    order = int(k)
    if order < 0:
        raise ValueError("k must be non-negative")

    target = np.asarray(p_target, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[1] != 1:
        raise ValueError(
            "p_target must contain one stabilized response per sample; "
            f"got shape {target.shape}"
        )

    reference = np.asarray(p_reference, dtype=np.float64)
    if reference.ndim == 1:
        reference = reference[:, None]
    if reference.ndim != 2:
        raise ValueError(
            f"p_reference must be [samples, references], got {reference.shape}"
        )
    if reference.shape[0] != target.shape[0]:
        raise ValueError("target/reference sample count mismatch")

    c_target = legendre_coefficients(target, order)
    c_reference = legendre_coefficients(reference, order)

    return anchor_tail_features(c_target, c_reference, order)


def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the float32 standardizer used by the tabular pipeline."""
    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0)
    std = x64.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(x) - mean[None, :]) / std[None, :]).astype(np.float32)


def standardize_image_triplet(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_evaluation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the image path's float64-fit/float32-output transform."""
    train64 = np.asarray(x_train, dtype=np.float64)
    mean64 = train64.mean(axis=0, keepdims=True)
    std64 = train64.std(axis=0, keepdims=True)
    std64 = np.where(std64 < 1e-6, 1.0, std64)
    return (
        ((np.asarray(x_train, dtype=np.float64) - mean64) / std64).astype(np.float32),
        ((np.asarray(x_validation, dtype=np.float64) - mean64) / std64).astype(np.float32),
        ((np.asarray(x_evaluation, dtype=np.float64) - mean64) / std64).astype(np.float32),
        mean64.squeeze(0).astype(np.float32),
        std64.squeeze(0).astype(np.float32),
    )


def orient_scores(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    labels_i = np.asarray(labels, dtype=np.int32)
    scores_f = np.asarray(scores, dtype=np.float64)
    raw_auc = float(roc_auc_score(labels_i, scores_f))
    if raw_auc < 0.5:
        return -scores_f, True, 1.0 - raw_auc
    return scores_f, False, raw_auc


def conservative_tpr_at_fp(labels: np.ndarray, scores: np.ndarray, fp_budget: int) -> float:
    labels_b = np.asarray(labels, dtype=bool)
    scores_f = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels_b.sum())
    n_neg = int((~labels_b).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    budget = min(max(int(fp_budget), 0), n_neg)
    valid = ~np.isnan(scores_f)
    values, inverse = np.unique(scores_f[valid], return_inverse=True)
    block_tp = np.bincount(
        inverse, weights=labels_b[valid].astype(np.int64), minlength=len(values)
    )
    block_fp = np.bincount(
        inverse, weights=(~labels_b[valid]).astype(np.int64), minlength=len(values)
    )
    tp = 0
    fp = 0
    for block_idx in range(len(values) - 1, -1, -1):
        n_tp = int(block_tp[block_idx])
        n_fp = int(block_fp[block_idx])
        if n_fp == 0:
            tp += n_tp
            continue
        if fp + n_fp <= budget:
            tp += n_tp
            fp += n_fp
            if fp >= budget:
                break
            continue
        break
    return float(tp / n_pos)


def conservative_tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    n_neg = int((~np.asarray(labels, dtype=bool)).sum())
    return conservative_tpr_at_fp(labels, scores, max(1, int(math.ceil(float(fpr) * n_neg))))


def fractional_tpr_at_fp(labels: np.ndarray, scores: np.ndarray, fp_budget: int) -> float:
    labels_b = np.asarray(labels, dtype=bool)
    scores_f = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels_b.sum())
    if n_pos == 0:
        return float("nan")
    tp = 0.0
    fp = 0.0
    for value in np.unique(scores_f)[::-1]:
        block = scores_f == value
        n_tp = int(labels_b[block].sum())
        n_fp = int((~labels_b[block]).sum())
        if n_fp == 0:
            tp += n_tp
            continue
        if fp + n_fp <= int(fp_budget):
            tp += n_tp
            fp += n_fp
            continue
        remain = max(0.0, int(fp_budget) - fp)
        tp += (remain / n_fp) * n_tp
        break
    return float(tp / n_pos)


def pseudo_validation_tpr_at_fp(
    labels: np.ndarray, scores: np.ndarray, fp_budget: int
) -> float:
    """Compute the tie-block score used for pseudo-validation.

    Member-only score blocks remain credited after the FP budget is exhausted
    until the next block containing a non-member. This function is restricted
    to image readout early stopping. Reported metrics use the exact ROC-step
    convention in :func:`exact_report_metrics`.
    """
    labels_b = np.asarray(labels, dtype=bool)
    scores_f = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels_b.sum())
    n_neg = int((~labels_b).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    budget = min(max(int(fp_budget), 0), n_neg)
    tp = 0.0
    fp = 0.0
    for value in np.unique(scores_f)[::-1]:
        block = scores_f == value
        n_tp = int(labels_b[block].sum())
        n_fp = int((~labels_b[block]).sum())
        if n_fp == 0:
            tp += n_tp
            continue
        if fp + n_fp <= budget:
            tp += n_tp
            fp += n_fp
            continue
        remain = max(0.0, budget - fp)
        tp += (remain / n_fp) * n_tp
        break
    return float(tp / n_pos)


def pseudo_validation_tpr_at_fpr(
    labels: np.ndarray, scores: np.ndarray, fpr: float
) -> float:
    n_neg = int((~np.asarray(labels, dtype=bool)).sum())
    return pseudo_validation_tpr_at_fp(
        labels, scores, max(1, int(math.ceil(float(fpr) * n_neg)))
    )


def fractional_tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    n_neg = int((~np.asarray(labels, dtype=bool)).sum())
    return fractional_tpr_at_fp(labels, scores, max(1, int(math.ceil(float(fpr) * n_neg))))


def partial_auc(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(
        np.asarray(labels, dtype=np.int32), np.asarray(scores, dtype=np.float64)
    )
    keep = fpr <= float(max_fpr)
    x = fpr[keep]
    y = tpr[keep]
    if x.size == 0:
        x = np.asarray([0.0])
        y = np.asarray([0.0])
    if x[-1] < float(max_fpr):
        x = np.r_[x, float(max_fpr)]
        y = np.r_[y, float(np.interp(float(max_fpr), fpr, tpr))]
    return float(sklearn_auc(x, y) / float(max_fpr))


def image_training_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Metrics used only to select a pseudo-validation epoch."""
    oriented, _, auc_value = orient_scores(labels, scores)
    fpr, tpr, _ = roc_curve(np.asarray(labels, dtype=np.int32), oriented)
    return {
        "AUC": auc_value,
        "BA": float(np.max(0.5 * (tpr + 1.0 - fpr))),
        "pAUC@0.1": partial_auc(labels, oriented, 0.001),
        "TPR@1%": pseudo_validation_tpr_at_fpr(labels, oriented, 0.01),
        "TPR@0.1%": pseudo_validation_tpr_at_fpr(labels, oriented, 0.001),
        "TPR@0.01%": pseudo_validation_tpr_at_fpr(labels, oriented, 0.0001),
        "TPR@0.001%": pseudo_validation_tpr_at_fpr(labels, oriented, 0.00001),
        "TPR@50FP": pseudo_validation_tpr_at_fp(labels, oriented, 50),
    }


def tabular_report_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    oriented, _, auc_value = orient_scores(labels, scores)
    fpr, tpr, _ = roc_curve(np.asarray(labels, dtype=np.int32), oriented)
    return {
        "AUC": auc_value,
        "BA": float(np.max(0.5 * (tpr + 1.0 - fpr))),
        "pAUC@0.1": partial_auc(labels, oriented, 0.1),
        "TPR@1%": fractional_tpr_at_fpr(labels, oriented, 0.01),
        "TPR@0.1%": fractional_tpr_at_fpr(labels, oriented, 0.001),
        "TPR@0.01%": fractional_tpr_at_fpr(labels, oriented, 0.0001),
        "TPR@0.001%": fractional_tpr_at_fpr(labels, oriented, 0.00001),
        "TPR@50FP": fractional_tpr_at_fp(labels, oriented, 50),
    }


def tpr_at_zero_fp(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(
        np.asarray(labels, dtype=np.int32),
        np.asarray(scores, dtype=np.float64),
        drop_intermediate=False,
    )
    return float(np.max(tpr[fpr == 0.0])) if np.any(fpr == 0.0) else 0.0


def exact_report_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Paper Eq. (6): max TPR at FPR <= alpha. No score flipping."""
    fpr, tpr, _ = roc_curve(
        np.asarray(labels, dtype=np.int32),
        np.asarray(scores, dtype=np.float64),
        drop_intermediate=False,
    )
    result: Dict[str, float] = {
        "AUC": float(roc_auc_score(np.asarray(labels, dtype=np.int32), scores)),
        "TPR@0FP": tpr_at_zero_fp(labels, scores),
    }
    for name, requested_fpr in zip(FPR_NAMES, FPR_POINTS):
        mask = fpr <= float(requested_fpr)
        result[name] = float(np.max(tpr[mask])) if np.any(mask) else 0.0
    return result


def exact_roc_points(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    fpr, tpr, threshold = roc_curve(
        np.asarray(labels, dtype=np.int32),
        np.asarray(scores, dtype=np.float64),
        drop_intermediate=False,
    )
    return pd.DataFrame(
        {
            "point_idx": np.arange(fpr.size, dtype=np.int64),
            "fpr": fpr.astype(np.float64),
            "tpr": tpr.astype(np.float64),
            "threshold": threshold.astype(np.float64),
        }
    )


def exact_rank_walk(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    labels_b = np.asarray(labels, dtype=bool)
    scores_f64 = np.asarray(scores, dtype=np.float64)
    indices = (
        np.arange(labels_b.size, dtype=np.int64)
        if sample_idx is None
        else np.asarray(sample_idx, dtype=np.int64)
    )
    order = np.argsort(-scores_f64, kind="mergesort")
    sorted_labels = labels_b[order]
    cum_tp = np.cumsum(sorted_labels.astype(np.int64))
    cum_fp = np.cumsum((~sorted_labels).astype(np.int64))
    n_pos = max(int(labels_b.sum()), 1)
    n_neg = max(int((~labels_b).sum()), 1)
    return pd.DataFrame(
        {
            "rank": np.arange(1, order.size + 1, dtype=np.int64),
            "sample_row": order.astype(np.int64),
            "sample_idx": indices[order],
            "label": sorted_labels.astype(np.int8),
            "score": scores_f64[order].astype(np.float64),
            "cum_tp": cum_tp,
            "cum_fp": cum_fp,
            "tpr": cum_tp / float(n_pos),
            "fpr": cum_fp / float(n_neg),
        }
    )


def image_training_key(metrics: Mapping[str, float], objective: str) -> Tuple[float, ...]:
    if objective == "tail001":
        order = ("TPR@0.001%", "TPR@0.01%", "TPR@0.1%", "TPR@1%", "pAUC@0.1", "AUC", "BA")
    elif objective == "balanced":
        order = ("AUC", "BA", "TPR@0.1%", "TPR@0.01%", "TPR@0.001%", "pAUC@0.1")
    elif objective == "allmetric":
        values = [float(metrics[key]) for key in ("AUC", "BA", "pAUC@0.1", "TPR@1%", "TPR@0.1%", "TPR@0.01%", "TPR@0.001%")]
        return (min(values), float(metrics["TPR@0.01%"]), float(metrics["TPR@0.001%"]), float(metrics["AUC"]), float(metrics["BA"]))
    else:
        order = ("TPR@0.01%", "TPR@0.001%", "TPR@0.1%", "TPR@1%", "pAUC@0.1", "AUC", "BA")
    return tuple(float(metrics[key]) for key in order)


class GatedImageReadout(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((int(dim),), 4.0))
        self.skip = nn.Linear(int(dim), 1)
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = x * self.gates().view(1, -1)
        return (self.skip(gated) + self.net(gated)).squeeze(-1)


class TabularReadout(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ReadoutResult:
    scores: np.ndarray
    metadata: Dict[str, float]
    state_dict: Dict[str, torch.Tensor]


def _cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fit_image_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    x_eval: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    weight_decay: float,
    learning_rate: float,
    feature_dropout: float,
    input_noise: float,
    gate_l1: float,
    objective: str,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
) -> ReadoutResult:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = GatedImageReadout(x_train.shape[1], int(hidden), float(dropout)).to(dev)
    try:
        linear = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            tol=1e-4,
            max_iter=500,
            fit_intercept=True,
        )
        linear.fit(x_train, np.asarray(y_train, dtype=np.int32))
        with torch.no_grad():
            gates = model.gates().detach().cpu().numpy().astype(np.float32)
            scale = np.maximum(gates, 1e-3)
            model.skip.weight.copy_(
                torch.from_numpy(linear.coef_.astype(np.float32) / scale[None, :]).to(dev)
            )
            model.skip.bias.copy_(torch.from_numpy(linear.intercept_.astype(np.float32)).to(dev))
    except Exception:
        pass

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

    n_rows = x_train_t.shape[0]
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(n_rows)
        for start in range(0, n_rows, max(1, int(batch_size))):
            index = order[start : start + max(1, int(batch_size))]
            xb = x_train_t[index]
            yb = y_train_t[index]
            if float(feature_dropout) > 0.0:
                xb = nn.functional.dropout(xb, p=float(feature_dropout), training=True)
            if float(input_noise) > 0.0:
                xb = xb + torch.randn_like(xb) * float(input_noise)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if float(gate_l1) > 0.0:
                loss = loss + float(gate_l1) * model.gates().mean()
            loss.backward()
            optimizer.step()

    # Final iterate — no early stopping.
    model.eval()
    with torch.no_grad():
        scores = model(x_eval_t).detach().cpu().numpy().astype(np.float64)
        gates = model.gates().detach().cpu().numpy().astype(np.float64)
    metadata = {
        "epochs_trained": float(epochs),
        "gate_mean": float(gates.mean()),
        "gate_min": float(gates.min()),
        "gate_max": float(gates.max()),
        "gate_active_frac_0p2": float((gates >= 0.2).mean()),
        "gate_active_frac_0p5": float((gates >= 0.5).mean()),
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
    }
    return ReadoutResult(scores=scores, metadata=metadata, state_dict=_cpu_state(model))


def tabular_selection_objective(metrics: Mapping[str, float]) -> float:
    return float(
        0.2 * metrics["AUC"]
        + 0.2 * metrics["BA"]
        + metrics["pAUC@0.1"]
        + 2.0 * metrics["TPR@1%"]
        + 3.0 * metrics["TPR@0.1%"]
        + 4.0 * metrics["TPR@0.01%"]
        + 4.0 * metrics["TPR@0.001%"]
    )


def fit_tabular_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_eval: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    weight_decay: float,
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
) -> ReadoutResult:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = TabularReadout(x_train.shape[1], int(hidden), float(dropout)).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    criterion = nn.BCEWithLogitsLoss()
    x_val_t = torch.tensor(np.asarray(x_val, dtype=np.float32), device=dev)
    best_state = _cpu_state(model)
    best_objective = -float("inf")
    best_epoch = -1
    best_sign = 1.0
    stale = 0
    rng = np.random.default_rng(int(seed))
    n_rows = x_train.shape[0]

    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(n_rows)
        for start in range(0, n_rows, int(batch_size)):
            index = order[start : start + int(batch_size)]
            xb = torch.tensor(x_train[index], dtype=torch.float32, device=dev)
            yb = torch.tensor(y_train[index].astype(np.float32), device=dev)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_scores = model(x_val_t).detach().cpu().numpy()
        metrics = tabular_report_metrics(y_val, val_scores)
        sign = 1.0
        if float(roc_auc_score(np.asarray(y_val, dtype=np.int32), val_scores)) < 0.5:
            sign = -1.0
            metrics = tabular_report_metrics(y_val, -val_scores)
        objective = tabular_selection_objective(metrics)
        if objective > best_objective + 1e-10:
            best_objective = objective
            best_epoch = epoch + 1
            best_state = _cpu_state(model)
            best_sign = sign
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, x_eval.shape[0], 131072):
            xb = torch.tensor(x_eval[start : start + 131072], dtype=torch.float32, device=dev)
            chunks.append(model(xb).detach().cpu().numpy())
    scores = np.concatenate(chunks).astype(np.float64) * float(best_sign)
    return ReadoutResult(
        scores=scores,
        metadata={
            "best_epoch": float(best_epoch),
            "pseudo_val_objective": float(best_objective),
            "score_sign": float(best_sign),
            "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
        },
        state_dict=best_state,
    )


def balanced_subsample(
    x: np.ndarray, y: np.ndarray, max_rows: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if int(max_rows) <= 0 or x.shape[0] <= int(max_rows):
        return x, y
    labels = np.asarray(y, dtype=bool)
    rng = np.random.default_rng(int(seed))
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    half = int(max_rows) // 2
    take_positive = min(half, positive.size)
    take = np.concatenate(
        [
            rng.choice(positive, size=take_positive, replace=False),
            rng.choice(
                negative,
                size=min(int(max_rows) - take_positive, negative.size),
                replace=False,
            ),
        ]
    )
    rng.shuffle(take)
    return x[take], labels[take]
