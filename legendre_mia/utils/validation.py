from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PACKAGE_ROOT
from ..attacks.core import exact_report_metrics
from .reporting import discover_assets


MANIFEST_ROOT = PACKAGE_ROOT / "manifest"
METRIC_COLUMNS = (
    "AUC",
    "TPR@0FP",
    "TPR@1%FPR",
    "TPR@0.1%FPR",
    "TPR@0.01%FPR",
    "TPR@0.001%FPR",
)


def _key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return str(row["modality"]), str(row["dataset"]), str(row["budget"])


def selected_keys(
    image_datasets: Sequence[str],
    budgets: Sequence[str],
    tabular_datasets: Sequence[str],
) -> List[Tuple[str, str, str]]:
    keys = [
        ("image", str(dataset), str(budget))
        for dataset in image_datasets
        for budget in budgets
    ]
    keys.extend(("tabular", str(dataset), "fair256") for dataset in tabular_datasets)
    return keys


def _asset_metrics(asset: Mapping[str, Any]) -> Dict[str, float]:
    labels = np.asarray(asset["labels"], dtype=bool)
    scores = np.asarray(asset["scores"], dtype=np.float64)
    values = exact_report_metrics(labels, scores)
    return {name: float(values[name]) for name in METRIC_COLUMNS}


def verify_outputs(
    output_root: Path,
    *,
    required_keys: Iterable[Tuple[str, str, str]] | None = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    assets = discover_assets(output_root)
    asset_lookup = {_key(asset): asset for asset in assets}
    wanted = set(required_keys or asset_lookup)
    missing = sorted(wanted.difference(asset_lookup))
    unexpected = sorted(set(asset_lookup).difference(wanted))
    rows: List[Dict[str, Any]] = []

    for key in sorted(wanted.intersection(asset_lookup)):
        asset = asset_lookup[key]
        labels = np.asarray(asset["labels"], dtype=bool)
        scores = np.asarray(asset["scores"], dtype=np.float64)
        structure_ok = bool(
            labels.ndim == 1
            and scores.ndim == 1
            and labels.shape == scores.shape
            and labels.any()
            and (~labels).any()
            and np.isfinite(scores).all()
            and Path(asset["path"]).is_file()
        )
        row: Dict[str, Any] = {
            "modality": key[0],
            "dataset": key[1],
            "budget": key[2],
            "samples": int(labels.size),
            "members": int(labels.sum()),
            "nonmembers": int((~labels).sum()),
            "structure_ok": structure_ok,
            "score_array_sha256": asset["score_array_sha256"],
            "label_array_sha256": asset["label_array_sha256"],
            "ok": structure_ok,
        }
        row.update(_asset_metrics(asset))
        rows.append(row)

    table = pd.DataFrame(rows)
    verification_root = output_root / "verification"
    verification_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(verification_root / "verification_rows.csv", index=False)
    ok = bool(
        not missing
        and not unexpected
        and len(rows) == len(wanted)
        and (table["ok"].all() if not table.empty else not wanted)
    )
    report = {
        "mode": "structural-and-numerical",
        "output_root": str(output_root.resolve()),
        "required_cells": len(wanted),
        "verified_cells": len(rows),
        "missing_cells": [list(value) for value in missing],
        "unexpected_cells": [list(value) for value in unexpected],
        "ok": ok,
    }
    (verification_root / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not ok:
        failed = table.loc[~table["ok"], ["modality", "dataset", "budget"]]
        raise RuntimeError(
            "Output verification failed: "
            f"missing={missing}, unexpected={unexpected}, "
            f"failed={failed.to_dict('records')}"
        )
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest() -> Dict[str, Any]:
    manifest_path = MANIFEST_ROOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["vendored_sources"]:
        path = PACKAGE_ROOT / entry["vendored_path"]
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": entry["vendored_sha256"],
                "actual_sha256": actual,
                "ok": actual == entry["vendored_sha256"],
            }
        )
    report = {"ok": all(row["ok"] for row in rows), "rows": rows}
    if not report["ok"]:
        raise RuntimeError("Vendored source integrity verification failed")
    return report
