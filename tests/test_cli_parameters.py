from __future__ import annotations

from legendre_mia.cli import (
    _csv,
    _parser,
    apply_image_overrides,
    normalize_image_datasets,
)
from legendre_mia.utils.config import DEFAULT_CONFIG, load_config


def test_custom_image_budget_overrides_all_pipeline_counts(tmp_path) -> None:
    args = _parser().parse_args(
        [
            "attack-image",
            "--datasets",
            "cifar100",
            "--augmentation",
            "stdaug",
            "--budgets",
            "custom",
            "--reference-models",
            "32",
            "--shadow-models",
            "6",
            "--attack-views",
            "16",
            "--target-lr",
            "0.0007",
            "--reference-lr",
            "0.0005",
            "--attack-lr",
            "0.0008",
        ]
    )
    config = load_config(
        DEFAULT_CONFIG,
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
    )
    datasets, augmentation = normalize_image_datasets(
        _csv(args.image_datasets),
        augmentation=args.augmentation,
        default_augmentation="stdaug",
    )
    budgets = apply_image_overrides(
        config,
        args,
        image_datasets=datasets,
        augmentation=augmentation,
        budgets=_csv(args.budgets),
    )
    assert datasets == ["cifar100_stdaug"]
    assert budgets == ["custom"]
    assert config["image"]["reference_training"]["num_references"] == 32
    assert config["image"]["cache"]["views"] == 16
    assert config["image"]["budgets"]["custom"] == {
        "reference_models": 32,
        "out_references_per_sample": 16,
        "views": 16,
    }
    assert config["image"]["attack"]["pseudo_ids"] == list(range(6))
    assert config["image"]["attack"]["shadow_models"] == 6
    assert config["image"]["target_training"]["lr"] == 0.0007
    assert config["image"]["reference_training"]["lr"] == 0.0005
    assert config["image"]["attack"]["learning_rate"] == 0.0008


def test_noaug_dataset_tags_and_cache_flags(tmp_path) -> None:
    args = _parser().parse_args(
        [
            "cache-image",
            "--datasets",
            "cifar10,cinic10",
            "--augmentation",
            "noaug",
            "--budgets",
            "custom",
            "--reference-models",
            "16",
            "--shadow-models",
            "4",
            "--attack-views",
            "3",
        ]
    )
    config = load_config(
        DEFAULT_CONFIG,
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
    )
    datasets, augmentation = normalize_image_datasets(
        _csv(args.image_datasets),
        augmentation=args.augmentation,
        default_augmentation="stdaug",
    )
    apply_image_overrides(
        config,
        args,
        image_datasets=datasets,
        augmentation=augmentation,
        budgets=_csv(args.budgets),
    )
    assert datasets == ["cifar10_noaug", "cinic10_noaug"]
    assert config["image"]["target_training"]["augmentation"] == "noaug"
    assert config["image"]["reference_training"]["augmentation"] == "noaug"
    assert config["image"]["cache"]["unique_stdaug_views"] is False
    assert config["image"]["cache"]["sample_index_views"] is False
