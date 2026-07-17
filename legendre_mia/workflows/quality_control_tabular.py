#!/usr/bin/env python3
"""Retrain tabular roles whose training accuracy is below the configured threshold."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from legendre_mia.models.tabular import (  # noqa: E402
    TABULAR_DATASETS,
    collect_model_pass_tabular,
    load_tabular_pool,
    save_torch_checkpoint,
    train_tabular_model_for_indices,
)


def role_dirs(task_dir: Path) -> List[Path]:
    roles = task_dir / "roles"
    out = []
    if (roles / "target").exists():
        out.append(roles / "target")
    out.extend(sorted((task_dir / "shadows").glob("shadow_*")))
    return out


def read_acc(role_dir: Path) -> float:
    path = role_dir / "metrics.json"
    if not path.exists():
        return float("-inf")
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get("acc", float("-inf")))
    except Exception:
        return float("-inf")


def save_signals(role_dir: Path, stats: Dict[str, np.ndarray]) -> None:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--task", choices=sorted(TABULAR_DATASETS), required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--view-type", default="identity")
    p.add_argument("--view-count", type=int, default=1)
    p.add_argument("--score-batch-size", type=int, default=4096)
    p.add_argument("--force", action="store_true", help="Retrain all roles, not only below-threshold roles.")
    p.add_argument(
        "--role-names",
        type=str,
        default="",
        help="Comma-separated role directory names to retrain, e.g. shadow_001,shadow_042. Applied before threshold filtering.",
    )
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional directory where existing role artifacts are copied before overwrite.",
    )
    p.add_argument(
        "--allow-remaining",
        action="store_true",
        help="Do not raise if roles remain below threshold; keep the report for inspection instead.",
    )
    p.add_argument("--early-stop-acc", type=float, default=None)
    p.add_argument("--early-stop-min-epochs", type=int, default=1)
    return p.parse_args()


def backup_role_artifacts(role_dir: Path, task_dir: Path, backup_dir: Path | None) -> str | None:
    if backup_dir is None:
        return None
    rel = role_dir.resolve().relative_to(task_dir.resolve())
    dst = backup_dir.resolve() / rel
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "model_state.pt",
        "model_state_meta.json",
        "signals.npz",
        "view_stats.npz",
        "metrics.json",
        "data_split.json",
    ]:
        src = role_dir / name
        if src.exists():
            shutil.copy2(src, dst / name)
    return str(dst)


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    spec = TABULAR_DATASETS[args.task]
    device = torch.device(args.device)
    x, y = load_tabular_pool(args.task, args.data_dir)
    default_epochs = int(spec["default_epochs"])
    control_epochs = max(default_epochs * 4, 120)
    lr = float(spec["default_lr"])
    optimizer_name = str(spec["default_optimizer"])
    weight_decay = float(spec["default_weight_decay"])

    rows: List[Dict[str, Any]] = []
    selected_role_names = {x.strip() for x in str(args.role_names).split(",") if x.strip()}
    all_role_dirs = role_dirs(task_dir)
    if selected_role_names:
        all_role_dirs = [r for r in all_role_dirs if r.name in selected_role_names]
        missing = sorted(selected_role_names.difference({r.name for r in all_role_dirs}))
        if missing:
            raise FileNotFoundError(f"{args.task}: requested role names not found: {missing[:10]}")
    bad_roles = [r for r in all_role_dirs if args.force or read_acc(r) < float(args.threshold)]
    print(
        json.dumps(
            {
                "task": args.task,
                "task_dir": str(task_dir),
                "threshold": float(args.threshold),
                "role_names": sorted(selected_role_names),
                "roles_to_update": len(bad_roles),
            },
            indent=2,
        ),
        flush=True,
    )

    for role_dir in bad_roles:
        before = read_acc(role_dir)
        split_path = role_dir / "data_split.json"
        if not split_path.exists():
            raise FileNotFoundError(f"{role_dir}: missing data_split.json")
        backup_path = backup_role_artifacts(role_dir, task_dir, args.backup_dir)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        role_name = str(split.get("role_name", role_dir.name))
        role_kind = str(split.get("role_kind", "shadow"))
        role_seed = int(split.get("seed", 0))
        train_idx = np.asarray(split["clf_train_idx"], dtype=np.int64)

        best_model = None
        best_metrics: Dict[str, float] = {"acc": float("-inf")}
        attempts = [
            {"seed": role_seed + 10007, "lr": lr, "epochs": control_epochs},
            {"seed": role_seed + 20011, "lr": lr * 0.5, "epochs": control_epochs},
            {"seed": role_seed + 30013, "lr": lr * 2.0, "epochs": control_epochs},
        ]
        for attempt_id, attempt in enumerate(attempts, start=1):
            model, metrics = train_tabular_model_for_indices(
                args.task,
                x,
                y,
                train_idx,
                epochs=int(attempt["epochs"]),
                batch_size=int(spec["default_batch_size"]),
                lr=float(attempt["lr"]),
                weight_decay=weight_decay,
                optimizer_name=optimizer_name,
                device=device,
                seed=int(attempt["seed"]),
                early_stop_acc=args.early_stop_acc,
                early_stop_min_epochs=int(args.early_stop_min_epochs),
            )
            metrics = dict(metrics)
            metrics["quality_control_attempt"] = float(attempt_id)
            metrics["quality_control_lr"] = float(attempt["lr"])
            metrics["quality_control_epochs"] = float(attempt["epochs"])
            if float(metrics.get("acc", -1.0)) > float(best_metrics.get("acc", -1.0)):
                if best_model is not None:
                    best_model.to("cpu")
                    del best_model
                best_model = model
                best_metrics = metrics
            else:
                model.to("cpu")
                del model
            if float(best_metrics.get("acc", -1.0)) >= float(args.threshold):
                break

        if best_model is None:
            raise RuntimeError(f"{role_dir}: failed to train a replacement model")

        stats = collect_model_pass_tabular(
            model=best_model,
            x=x,
            y=y,
            batch_size=int(args.score_batch_size),
            view_count=int(args.view_count),
            view_type=str(args.view_type),
            seed=role_seed + 404,
            device=device,
        )
        save_signals(role_dir, stats)
        best_metrics["quality_control_previous_acc"] = float(before)
        best_metrics["quality_control_threshold"] = float(args.threshold)
        (role_dir / "metrics.json").write_text(
            json.dumps(best_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        save_torch_checkpoint(
            output_dir=role_dir,
            model=best_model,
            model_name="mlp",
            task=args.task,
            role_name=role_name,
            role_kind=role_kind,
            extra_meta={"task_kind": "tabular", "quality_control": True},
        )
        after = float(best_metrics.get("acc", float("nan")))
        rows.append({"role_dir": str(role_dir), "before_acc": before, "after_acc": after, "backup_dir": backup_path})
        print(f"[quality] {args.task}/{role_name}: {before:.6f} -> {after:.6f}", flush=True)
        best_model.to("cpu")
        del best_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "task": args.task,
        "task_dir": str(task_dir),
        "threshold": float(args.threshold),
        "updated_roles": rows,
        "remaining_below_threshold": [
            {"role_dir": str(r), "acc": read_acc(r)}
            for r in role_dirs(task_dir)
            if read_acc(r) < float(args.threshold)
        ],
    }
    (task_dir / "tabular_accuracy_control_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if report["remaining_below_threshold"] and not args.allow_remaining:
        raise RuntimeError(f"{args.task}: roles still below threshold: {report['remaining_below_threshold'][:5]}")
    print(json.dumps({"task": args.task, "updated_roles": len(rows), "status": "ok"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
