from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

from .attacks.image import run_image_attacks
from .attacks.tabular import run_tabular_attacks
from .utils.config import DEFAULT_CONFIG, PACKAGE_ROOT, load_config, write_resolved_config
from .utils.reporting import export_all
from .utils.validation import selected_keys, verify_outputs, verify_source_manifest
from .workflows.orchestration import (
    StageRunner,
    cache_image_outputs,
    cache_tabular_outputs,
    doctor,
    quality_control_tabular_models,
    train_image_references,
    train_image_targets,
    train_tabular_models,
)


COMMANDS = (
    "doctor",
    "train-image-targets",
    "train-image-references",
    "cache-image",
    "train-tabular-models",
    "quality-control-tabular",
    "cache-tabular",
    "attack-image",
    "attack-tabular",
    "export",
    "verify",
    "models",
    "attack",
    "results",
    "all",
)

IMAGE_BASE_DATASETS = ("cifar10", "cifar100", "cinic10", "tinyimagenet")
IMAGE_AUGMENTATIONS = ("stdaug", "noaug")


def _csv(raw: str) -> List[str]:
    if str(raw).strip().lower() in {"", "none", "off"}:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _validate_subset(values: Sequence[str], allowed: Sequence[str], label: str) -> None:
    invalid = sorted(set(values).difference(allowed))
    if invalid:
        raise ValueError(f"Unsupported {label}: {invalid}; allowed={list(allowed)}")


def _positive(value: int | float, label: str) -> None:
    if float(value) <= 0:
        raise ValueError(f"{label} must be > 0; got {value}")


def normalize_image_datasets(
    values: Sequence[str],
    *,
    augmentation: str | None,
    default_augmentation: str,
) -> tuple[List[str], str]:
    parsed = []
    seen_augmentations = set()
    for raw in values:
        item = str(raw).strip().lower()
        item_augmentation = None
        for suffix in IMAGE_AUGMENTATIONS:
            marker = f"_{suffix}"
            if item.endswith(marker):
                item = item[: -len(marker)]
                item_augmentation = suffix
                break
        if item not in IMAGE_BASE_DATASETS:
            raise ValueError(
                f"Unsupported image dataset {raw!r}; allowed={list(IMAGE_BASE_DATASETS)}"
            )
        if augmentation is not None:
            item_augmentation = str(augmentation)
        elif item_augmentation is None:
            item_augmentation = str(default_augmentation)
        seen_augmentations.add(item_augmentation)
        parsed.append((item, item_augmentation))
    if len(seen_augmentations) > 1:
        raise ValueError(
            "One pipeline invocation supports one augmentation regime; "
            f"got {sorted(seen_augmentations)}"
        )
    selected_augmentation = next(
        iter(seen_augmentations), str(augmentation or default_augmentation)
    )
    return [f"{name}_{selected_augmentation}" for name, _ in parsed], selected_augmentation


def apply_image_overrides(
    config: Dict[str, Any],
    args: argparse.Namespace,
    *,
    image_datasets: Sequence[str],
    augmentation: str,
    budgets: Sequence[str],
) -> List[str]:
    image = config["image"]
    target = image["target_training"]
    reference = image["reference_training"]
    attack = image["attack"]
    selected_budgets = list(budgets)

    for tag in image_datasets:
        image["dataset_names"][tag] = tag.rsplit("_", 1)[0]
    image["datasets"] = list(image_datasets)
    target["augmentation"] = augmentation
    reference["augmentation"] = augmentation
    image["cache"]["unique_stdaug_views"] = augmentation == "stdaug"
    image["cache"]["sample_index_views"] = augmentation == "stdaug"

    architecture = str(args.architecture or target["architecture"])
    if architecture != "resnet18":
        raise ValueError("Only --architecture resnet18 is supported")
    target["architecture"] = architecture
    reference["architecture"] = architecture

    common_model_values = {
        "epochs": args.model_epochs,
        "lr": args.model_lr,
        "weight_decay": args.model_weight_decay,
        "batch_size": args.model_batch_size,
    }
    for key, value in common_model_values.items():
        if value is not None:
            target[key] = value
            reference[key] = value
    target_values = {
        "epochs": args.target_epochs,
        "lr": args.target_lr,
        "weight_decay": args.target_weight_decay,
        "batch_size": args.target_batch_size,
        "seed": args.target_seed,
    }
    reference_values = {
        "epochs": args.reference_epochs,
        "lr": args.reference_lr,
        "weight_decay": args.reference_weight_decay,
        "batch_size": args.reference_batch_size,
        "design_seed": args.reference_design_seed,
        "model_seed_base": args.reference_seed_base,
    }
    attack_values = {
        "epochs": args.attack_epochs,
        "learning_rate": args.attack_lr,
        "weight_decay": args.attack_weight_decay,
        "batch_size": args.attack_batch_size,
        "seed": args.attack_seed,
    }
    for section, values in ((target, target_values), (reference, reference_values), (attack, attack_values)):
        for key, value in values.items():
            if value is not None:
                section[key] = value

    custom_flags = (
        args.reference_models,
        args.pseudo_shadow_models,
        args.attack_views,
    )
    if any(value is not None for value in custom_flags) and "custom" not in selected_budgets:
        if selected_budgets == ["large", "small"]:
            selected_budgets = ["custom"]
        else:
            raise ValueError(
                "--reference-models/--shadow-models/--attack-views require "
                "--budgets custom"
            )
    if "custom" in selected_budgets:
        if selected_budgets != ["custom"]:
            raise ValueError("The custom budget cannot be mixed with small or large")
        reference_models = int(
            args.reference_models or image["budgets"]["large"]["reference_models"]
        )
        pseudo_shadows = int(args.pseudo_shadow_models or len(attack["pseudo_ids"]))
        attack_views = int(args.attack_views or image["budgets"]["large"]["views"])
        if reference_models < 4 or reference_models % 2 != 0:
            raise ValueError("--reference-models must be an even integer >= 4")
        if pseudo_shadows < 2 or pseudo_shadows > reference_models:
            raise ValueError(
                "--shadow-models must be in [2, reference-models]; these are "
                "the pseudo-shadow worlds used to train the attack readout"
            )
        if augmentation == "stdaug" and attack_views > 162:
            raise ValueError("stdaug supports at most 162 unique crop/flip views")
        _positive(attack_views, "--attack-views")
        image["budgets"]["custom"] = {
            "reference_models": reference_models,
            "out_references_per_sample": reference_models // 2,
            "views": attack_views,
        }
        attack["pseudo_ids"] = list(range(pseudo_shadows))
        attack["shadow_models"] = pseudo_shadows

    _validate_subset(selected_budgets, tuple(image["budgets"]), "image budgets")
    if selected_budgets:
        reference["num_references"] = max(
            int(image["budgets"][budget]["reference_models"])
            for budget in selected_budgets
        )
        image["cache"]["views"] = max(
            int(image["budgets"][budget]["views"])
            for budget in selected_budgets
        )

    for section_name, section, positive_keys in (
        ("target", target, ("epochs", "lr", "batch_size")),
        ("reference", reference, ("epochs", "lr", "batch_size")),
        ("attack", attack, ("epochs", "learning_rate", "batch_size")),
    ):
        for key in positive_keys:
            _positive(section[key], f"{section_name}.{key}")
        if float(section["weight_decay"]) < 0:
            raise ValueError(f"{section_name}.weight_decay must be >= 0")
    return selected_budgets


def run_signature(
    config: Mapping[str, Any],
    *,
    image_datasets: Sequence[str],
    tabular_datasets: Sequence[str],
    budgets: Sequence[str],
    devices: Sequence[str],
    attack_device: str,
) -> str:
    payload = {
        "config": config,
        "image_datasets": list(image_datasets),
        "tabular_datasets": list(tabular_datasets),
        "budgets": list(budgets),
        "devices": list(devices),
        "attack_device": str(attack_device),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _torchvision_version() -> str:
    try:
        import torchvision

        return str(torchvision.__version__)
    except Exception as error:  # pragma: no cover - diagnostic path
        return f"unavailable: {error!r}"


def _parser(
    *,
    fixed_command: str | None = None,
    prog: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Legendre membership inference: train models, export deterministic "
            "model outputs, fit K4 readouts, and report metrics and ROC data."
        )
    )
    if fixed_command is None:
        parser.add_argument("command", choices=COMMANDS)
    else:
        if fixed_command not in COMMANDS:
            raise ValueError(f"Unsupported fixed command: {fixed_command}")
        parser.set_defaults(command=fixed_command)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=PACKAGE_ROOT / "artifacts")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--image-datasets",
        "--datasets",
        dest="image_datasets",
        default="cifar10_stdaug,cifar100_stdaug,cinic10_stdaug,tinyimagenet_stdaug",
    )
    parser.add_argument("--tabular-datasets", default="purchase100,texas100,location30")
    parser.add_argument("--budgets", default="large,small")
    parser.add_argument("--devices", default="0,1", help="GPU ids, or cpu.")
    parser.add_argument("--attack-device", default="cpu")
    parser.add_argument("--python", default=sys.executable)
    image = parser.add_argument_group("image pipeline settings")
    image.add_argument("--augmentation", choices=IMAGE_AUGMENTATIONS, default=None)
    image.add_argument("--architecture", choices=("resnet18",), default=None)
    image.add_argument("--reference-models", type=int, default=None)
    image.add_argument(
        "--shadow-models",
        "--pseudo-shadow-models",
        dest="pseudo_shadow_models",
        type=int,
        default=None,
        help="Pseudo-shadow worlds used to fit the readout; selected from the reference pool.",
    )
    image.add_argument("--attack-views", type=int, default=None)
    image.add_argument("--model-epochs", type=int, default=None)
    image.add_argument("--model-lr", type=float, default=None)
    image.add_argument("--model-weight-decay", type=float, default=None)
    image.add_argument("--model-batch-size", type=int, default=None)
    image.add_argument("--target-epochs", type=int, default=None)
    image.add_argument("--target-lr", type=float, default=None)
    image.add_argument("--target-weight-decay", type=float, default=None)
    image.add_argument("--target-batch-size", type=int, default=None)
    image.add_argument("--target-seed", type=int, default=None)
    image.add_argument("--reference-epochs", type=int, default=None)
    image.add_argument("--reference-lr", type=float, default=None)
    image.add_argument("--reference-weight-decay", type=float, default=None)
    image.add_argument("--reference-batch-size", type=int, default=None)
    image.add_argument("--reference-design-seed", type=int, default=None)
    image.add_argument("--reference-seed-base", type=int, default=None)
    image.add_argument("--attack-epochs", type=int, default=None)
    image.add_argument("--attack-lr", type=float, default=None)
    image.add_argument("--attack-weight-decay", type=float, default=None)
    image.add_argument("--attack-batch-size", type=int, default=None)
    image.add_argument("--attack-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed stages instead of resuming from status/*.json.",
    )
    return parser


def _roots(config: Mapping[str, Any]) -> tuple[Dict[str, Path], Path]:
    image = {
        dataset: Path(config["paths"]["image_cache_root"])
        for dataset in config["image"]["datasets"]
    }
    return image, Path(config["paths"]["tabular_cache_root"])


def main(
    argv: Sequence[str] | None = None,
    *,
    fixed_command: str | None = None,
    prog: str | None = None,
) -> int:
    args = _parser(fixed_command=fixed_command, prog=prog).parse_args(argv)
    output_root = Path(
        args.output_root
        or PACKAGE_ROOT / "outputs" / "run"
    ).expanduser().resolve()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(
        Path(args.config), artifact_root=artifact_root, output_root=output_root
    )
    raw_image_datasets = _csv(args.image_datasets)
    image_datasets, augmentation = normalize_image_datasets(
        raw_image_datasets,
        augmentation=args.augmentation,
        default_augmentation=str(config["image"]["target_training"]["augmentation"]),
    )
    tabular_datasets = _csv(args.tabular_datasets)
    budgets = _csv(args.budgets)
    devices = _csv(args.devices)
    budgets = apply_image_overrides(
        config,
        args,
        image_datasets=image_datasets,
        augmentation=augmentation,
        budgets=budgets,
    )
    _validate_subset(tabular_datasets, config["tabular"]["datasets"], "tabular datasets")
    write_resolved_config(config, output_root / "resolved_config.json")
    (output_root / "environment.json").write_text(
        json.dumps(_environment(), indent=2, sort_keys=True), encoding="utf-8"
    )
    signature = run_signature(
        config,
        image_datasets=image_datasets,
        tabular_datasets=tabular_datasets,
        budgets=budgets,
        devices=devices,
        attack_device=args.attack_device,
    )
    runner = StageRunner(
        output_root,
        dry_run=args.dry_run,
        resume=not args.force,
        run_signature=signature,
    )
    image_cache_roots, tabular_cache_root = _roots(config)
    wanted = selected_keys(image_datasets, budgets, tabular_datasets)

    def run_doctor() -> None:
        report = doctor(
            config,
            image_enabled=bool(image_datasets),
            tabular_enabled=bool(tabular_datasets),
        )
        report["source_integrity"] = verify_source_manifest()
        (output_root / "doctor.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    def attack_image() -> None:
        if image_datasets:
            run_image_attacks(
                config,
                cache_roots={dataset: image_cache_roots[dataset] for dataset in image_datasets},
                output_root=output_root,
                datasets=image_datasets,
                budgets=budgets,
                device=args.attack_device,
            )

    def attack_tabular() -> None:
        if tabular_datasets:
            run_tabular_attacks(
                config,
                cache_root=tabular_cache_root,
                output_root=output_root,
                datasets=tabular_datasets,
                device=args.attack_device,
            )

    def export() -> None:
        export_all(config, output_root)

    def verify() -> None:
        verify_outputs(
            output_root,
            required_keys=wanted,
        )

    actions = {
        "doctor": lambda: runner.run_callable("doctor", run_doctor),
        "train-image-targets": lambda: train_image_targets(
            config,
            runner,
            datasets=image_datasets,
            devices=devices,
            python=args.python,
        ),
        "train-image-references": lambda: train_image_references(
            config,
            runner,
            datasets=image_datasets,
            devices=devices,
            python=args.python,
        ),
        "cache-image": lambda: cache_image_outputs(
            config,
            runner,
            datasets=image_datasets,
            devices=devices,
            python=args.python,
        ),
        "train-tabular-models": lambda: train_tabular_models(
            config,
            runner,
            datasets=tabular_datasets,
            device=devices[0] if devices else "cpu",
            python=args.python,
        ),
        "quality-control-tabular": lambda: quality_control_tabular_models(
            config,
            runner,
            datasets=tabular_datasets,
            devices=devices,
            python=args.python,
        ),
        "cache-tabular": lambda: cache_tabular_outputs(
            config, runner, datasets=tabular_datasets, python=args.python
        ),
        "attack-image": lambda: runner.run_callable("attack_image", attack_image),
        "attack-tabular": lambda: runner.run_callable("attack_tabular", attack_tabular),
        "export": lambda: runner.run_callable("export", export),
        "verify": lambda: runner.run_callable("verify", verify),
    }

    if args.command in actions:
        actions[args.command]()
        return 0

    def run_models() -> None:
        actions["doctor"]()
        if image_datasets:
            actions["train-image-targets"]()
            actions["train-image-references"]()
            actions["cache-image"]()
        if tabular_datasets:
            actions["train-tabular-models"]()
            actions["quality-control-tabular"]()
            actions["cache-tabular"]()

    def run_attacks() -> None:
        if image_datasets:
            actions["attack-image"]()
        if tabular_datasets:
            actions["attack-tabular"]()

    def run_results() -> None:
        actions["export"]()
        actions["verify"]()

    if args.command in {"models", "all"}:
        run_models()
    if args.command in {"attack", "all"}:
        run_attacks()
    if args.command in {"results", "all"}:
        run_results()
    return 0
