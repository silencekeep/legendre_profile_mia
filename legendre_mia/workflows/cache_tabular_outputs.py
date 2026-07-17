#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _member_mask(path: Path, sample_count: int) -> np.ndarray:
    split = json.loads(path.read_text(encoding="utf-8"))
    train_indices = np.asarray(split["clf_train_idx"], dtype=np.int64)
    if train_indices.size and (
        int(train_indices.min()) < 0 or int(train_indices.max()) >= sample_count
    ):
        raise ValueError(f"{path}: training index outside [0,{sample_count})")
    member = np.zeros(sample_count, dtype=bool)
    member[train_indices] = True
    return member


def _load_role(role_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal_path = role_dir / "signals.npz"
    split_path = role_dir / "data_split.json"
    if not signal_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Missing signals or split under {role_dir}")
    with np.load(signal_path) as values:
        labels = np.asarray(values["class_label"], dtype=np.int64)
        true_prob = np.asarray(values["true_prob"], dtype=np.float32)
        max_other = np.asarray(values["max_other_prob"], dtype=np.float32)
    member = _member_mask(split_path, len(labels))
    return labels, member, true_prob, max_other


def _write_role(
    path: Path,
    labels: np.ndarray,
    member: np.ndarray,
    true_prob: np.ndarray,
    max_other: np.ndarray,
) -> None:
    sample_count = len(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_idx=np.arange(sample_count, dtype=np.int64),
        class_label=labels.astype(np.int64),
        member=member.astype(bool),
        true_prob=true_prob.reshape(sample_count, 1).astype(np.float16),
        max_other_prob=max_other.reshape(sample_count, 1).astype(np.float16),
    )


def export_outputs(
    task_dir: Path,
    output_dir: Path,
    *,
    reference_count: int,
    expected_out: int,
) -> pd.DataFrame:
    models_dir = output_dir / "models"
    rows: List[Dict[str, object]] = []
    labels, target_member, target_prob, target_other = _load_role(
        task_dir / "roles" / "target"
    )
    _write_role(models_dir / "target.npz", labels, target_member, target_prob, target_other)
    rows.append(
        {
            "role": "target",
            "samples": len(labels),
            "members": int(target_member.sum()),
            "status": "ok",
        }
    )
    memberships = []
    for reference_id in range(int(reference_count)):
        role_name = f"shadow_{reference_id:03d}"
        role_dir = task_dir / "shadows" / role_name
        labels_i, member_i, prob_i, other_i = _load_role(role_dir)
        if not np.array_equal(labels_i, labels):
            raise ValueError(f"{role_dir}: labels differ from target labels")
        _write_role(models_dir / f"{role_name}.npz", labels_i, member_i, prob_i, other_i)
        memberships.append(member_i)
        rows.append(
            {
                "role": role_name,
                "samples": len(labels_i),
                "members": int(member_i.sum()),
                "status": "ok",
            }
        )
    membership = np.stack(memberships, axis=1)
    out_count = (~membership).sum(axis=1)
    if not np.all(out_count == int(expected_out)):
        raise RuntimeError(
            f"OUT-reference balance failed: min={out_count.min()} "
            f"max={out_count.max()} expected={expected_out}"
        )
    np.savez_compressed(
        output_dir / "membership_matrix.npz",
        reference_member=membership.astype(np.int8),
        out_count=out_count.astype(np.int16),
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tabular model outputs.")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-count", type=int, default=256)
    parser.add_argument("--expected-out", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = export_outputs(
        args.task_dir,
        args.output_dir,
        reference_count=args.reference_count,
        expected_out=args.expected_out,
    )
    report.to_csv(args.output_dir / "output_report.csv", index=False)
    manifest = {
        "task_dir": str(args.task_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "reference_count": args.reference_count,
        "expected_out": args.expected_out,
        "ok": bool(report["status"].eq("ok").all()),
    }
    (args.output_dir / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
