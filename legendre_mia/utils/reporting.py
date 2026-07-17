from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..attacks.core import (
    exact_rank_walk,
    exact_report_metrics,
    exact_roc_points,
)


IMAGE_TITLES = {
    "cifar10_stdaug": "(a) CIFAR-10",
    "cifar10_noaug": "(a) CIFAR-10",
    "cifar100_stdaug": "(b) CIFAR-100",
    "cifar100_noaug": "(b) CIFAR-100",
    "cinic10_stdaug": "(c) CINIC-10",
    "cinic10_noaug": "(c) CINIC-10",
    "tinyimagenet_stdaug": "(d) Tiny-ImageNet",
    "tinyimagenet_noaug": "(d) Tiny-ImageNet",
}
TABULAR_TITLES = {
    "purchase100": "(a) Purchase100",
    "texas100": "(b) Texas100",
    "location30": "(c) Location30",
}


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        return {key: np.asarray(item[key]) for key in item.files}


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_score_asset(path: Path, modality: str) -> Dict[str, Any]:
    item = _read_npz(path)
    labels = item["labels"].astype(bool)
    scores = item["scores"].astype(np.float64)
    dataset = _scalar_text(item["dataset"])
    budget = (
        _scalar_text(item["budget"])
        if "budget" in item
        else "fair256"
    )
    method = _scalar_text(item["method"])
    sample_idx = item.get(
        "sample_idx", np.arange(labels.size, dtype=np.int64)
    ).astype(np.int64)
    return {
        "path": path,
        "modality": modality,
        "dataset": dataset,
        "budget": budget,
        "method": method,
        "labels": labels,
        "scores": scores,
        "sample_idx": sample_idx,
        "score_array_sha256": _array_sha256(scores.astype(np.float32)),
        "label_array_sha256": _array_sha256(labels.astype(np.int8)),
    }


def discover_assets(output_root: Path) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for modality in ("image", "tabular"):
        root = Path(output_root) / "scores" / modality
        if not root.exists():
            continue
        for path in sorted(root.glob("*.npz")):
            assets.append(load_score_asset(path, modality))
    return assets


def export_tables(
    assets: Sequence[Mapping[str, Any]], output_root: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_dir = Path(output_root) / "metrics"
    data_dir = Path(output_root) / "loglogroc_data"
    table_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[Dict[str, Any]] = []
    roc_by_modality: Dict[str, List[pd.DataFrame]] = {"image": [], "tabular": []}
    rank_by_modality: Dict[str, List[pd.DataFrame]] = {"image": [], "tabular": []}

    for asset in assets:
        labels = np.asarray(asset["labels"], dtype=bool)
        scores = np.asarray(asset["scores"], dtype=np.float64)
        exact = exact_report_metrics(labels, scores)
        row = {
            "modality": asset["modality"],
            "dataset": asset["dataset"],
            "budget": asset["budget"],
            "method": asset["method"],
            "AUC": float(exact["AUC"]),
            "TPR@0FP": float(exact["TPR@0FP"]),
            "TPR@1%FPR": float(exact["TPR@1%FPR"]),
            "TPR@0.1%FPR": float(exact["TPR@0.1%FPR"]),
            "TPR@0.01%FPR": float(exact["TPR@0.01%FPR"]),
            "TPR@0.001%FPR": float(exact["TPR@0.001%FPR"]),
        }
        row.update(
            {
                "samples": int(labels.size),
                "members": int(labels.sum()),
                "nonmembers": int((~labels).sum()),
                "score_path": str(Path(asset["path"]).resolve()),
                "score_array_sha256": asset["score_array_sha256"],
                "label_array_sha256": asset["label_array_sha256"],
            }
        )
        metric_rows.append(row)

        roc = exact_roc_points(labels, scores)
        roc.insert(0, "method", asset["method"])
        roc.insert(0, "budget", asset["budget"])
        roc.insert(0, "dataset", asset["dataset"])
        roc_by_modality[str(asset["modality"])].append(roc)
        rank = exact_rank_walk(labels, scores, np.asarray(asset["sample_idx"]))
        rank.insert(0, "method", asset["method"])
        rank.insert(0, "budget", asset["budget"])
        rank.insert(0, "dataset", asset["dataset"])
        rank_by_modality[str(asset["modality"])].append(rank)

    metrics = pd.DataFrame(metric_rows)
    image_metrics = metrics[metrics["modality"].eq("image")].copy()
    tabular_metrics = metrics[metrics["modality"].eq("tabular")].copy()
    image_metrics.to_csv(table_dir / "image_metrics.csv", index=False)
    tabular_metrics.to_csv(table_dir / "tabular_metrics.csv", index=False)
    metrics.to_csv(table_dir / "all_metrics.csv", index=False)

    for modality in ("image", "tabular"):
        if not roc_by_modality[modality]:
            continue
        roc = pd.concat(roc_by_modality[modality], ignore_index=True)
        rank = pd.concat(rank_by_modality[modality], ignore_index=True)
        with gzip.open(
            data_dir / f"{modality}_ours_exact_roc_points.csv.gz",
            "wt",
            encoding="utf-8",
        ) as handle:
            roc.to_csv(handle, index=False)
        with gzip.open(
            data_dir / f"{modality}_ours_rank_walk.csv.gz",
            "wt",
            encoding="utf-8",
        ) as handle:
            rank.to_csv(handle, index=False)
    image_roc = (
        pd.concat(roc_by_modality["image"], ignore_index=True)
        if roc_by_modality["image"]
        else pd.DataFrame()
    )
    tabular_roc = (
        pd.concat(roc_by_modality["tabular"], ignore_index=True)
        if roc_by_modality["tabular"]
        else pd.DataFrame()
    )
    return metrics, image_roc, tabular_roc


def _visible_step(curve: pd.DataFrame, floor: float) -> Tuple[np.ndarray, np.ndarray]:
    if curve.empty:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    fpr = curve["fpr"].to_numpy(dtype=np.float64)
    tpr = curve["tpr"].to_numpy(dtype=np.float64)
    finite = np.isfinite(fpr) & np.isfinite(tpr)
    fpr = fpr[finite]
    tpr = tpr[finite]
    at_floor = fpr <= float(floor)
    tpr_floor = float(np.max(tpr[at_floor])) if np.any(at_floor) else 0.0
    interior = (fpr > float(floor)) & (tpr > 0.0)
    x = np.r_[float(floor), fpr[interior]]
    y = np.r_[max(tpr_floor, float(floor)), np.maximum(tpr[interior], float(floor))]
    if x[-1] < 1.0 or y[-1] < 1.0:
        x = np.r_[x, 1.0]
        y = np.r_[y, 1.0]
    changed = np.r_[True, (np.diff(x) != 0.0) | (np.diff(y) != 0.0)]
    return x[changed], y[changed]


def _set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "legend.framealpha": 0.88,
        }
    )


def _plot_panels(
    roc: pd.DataFrame,
    *,
    datasets: Sequence[str],
    titles: Mapping[str, str],
    output_stem: Path,
    floor: float,
    figsize: Tuple[float, float],
    budget: str | None = None,
) -> None:
    if roc.empty:
        return
    _set_ieee_style()
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        cell = roc[roc["dataset"].eq(dataset)]
        if budget is not None:
            cell = cell[cell["budget"].eq(budget)]
        x, y = _visible_step(cell, float(floor))
        if x.size == 0:
            axis.set_visible(False)
            continue
        axis.step(
            x,
            y,
            where="post",
            color="#d43f35",
            linewidth=2.3,
            label="Ours",
            zorder=10,
        )
        axis.plot(
            [floor, 1.0],
            [floor, 1.0],
            linestyle="--",
            color="0.72",
            linewidth=1.0,
            zorder=1,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(float(floor), 1.0)
        axis.set_ylim(float(floor), 1.0)
        axis.set_title(titles[dataset], fontsize=16, y=1.04)
        axis.set_xlabel("False Positive Rate", fontsize=13)
        axis.set_ylabel("True Positive Rate", fontsize=13)
        axis.tick_params(axis="both", which="major", labelsize=10, length=4.0)
        axis.tick_params(axis="both", which="minor", length=2.4)
        axis.legend(loc="lower right", fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.84, wspace=0.38)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def export_figures(
    config: Mapping[str, Any],
    image_roc: pd.DataFrame,
    tabular_roc: pd.DataFrame,
    output_root: Path,
) -> None:
    figure_dir = Path(output_root) / "figures"
    floor = float(config["reporting"]["loglog_floor"])
    budget_order = list(config["image"]["budgets"])
    present_budgets = list(dict.fromkeys(image_roc.get("budget", pd.Series(dtype=str))))
    for budget in [
        *[value for value in budget_order if value in present_budgets],
        *[value for value in present_budgets if value not in budget_order],
    ]:
        if image_roc.empty or not image_roc["budget"].eq(budget).any():
            continue
        present = set(image_roc.loc[image_roc["budget"].eq(budget), "dataset"])
        _plot_panels(
            image_roc,
            datasets=[
                dataset for dataset in config["image"]["datasets"] if dataset in present
            ],
            titles=IMAGE_TITLES,
            output_stem=figure_dir / f"image_{budget}_ours_loglogroc",
            floor=floor,
            figsize=(17.6, 4.1),
            budget=budget,
        )
    if not tabular_roc.empty:
        present = set(tabular_roc["dataset"])
        datasets = [
            dataset for dataset in config["tabular"]["datasets"] if dataset in present
        ]
        _plot_panels(
            tabular_roc,
            datasets=datasets,
            titles=TABULAR_TITLES,
            output_stem=figure_dir / "tabular_ours_loglogroc",
            floor=floor,
            figsize=(4.4 * len(datasets), 4.1),
        )


def export_all(config: Mapping[str, Any], output_root: Path) -> pd.DataFrame:
    assets = discover_assets(Path(output_root))
    if not assets:
        raise FileNotFoundError(f"No Ours score assets under {Path(output_root) / 'scores'}")
    metrics, image_roc, tabular_roc = export_tables(assets, Path(output_root))
    export_figures(config, image_roc, tabular_roc, Path(output_root))
    manifest = {
        "protocol_id": config["protocol_id"],
        "score_assets": len(assets),
        "image_cells": int(metrics["modality"].eq("image").sum()),
        "tabular_cells": int(metrics["modality"].eq("tabular").sum()),
        "score_files": [str(Path(asset["path"]).resolve()) for asset in assets],
        "metrics": "metrics/all_metrics.csv",
        "image_roc": "loglogroc_data/image_ours_exact_roc_points.csv.gz",
        "image_rank": "loglogroc_data/image_ours_rank_walk.csv.gz",
        "tabular_roc": "loglogroc_data/tabular_ours_exact_roc_points.csv.gz",
        "tabular_rank": "loglogroc_data/tabular_ours_rank_walk.csv.gz",
        "rank_sort": "descending stable mergesort",
        "roc": "sklearn.metrics.roc_curve(drop_intermediate=False)",
    }
    (Path(output_root) / "ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics
