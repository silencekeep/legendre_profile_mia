#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from legendre_mia.models import image as image_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an independent ResNet-18 target model.")
    parser.add_argument("--dataset", choices=image_models.DATASETS, required=True)
    parser.add_argument("--augmentation", choices=("stdaug", "noaug"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_models.set_seed(args.seed)
    dataset, num_classes = image_models.load_dataset(
        args.dataset, args.data_root, augmentation=args.augmentation
    )
    train_indices, heldout_indices = image_models.stratified_indices(
        dataset, train_fraction=0.5, seed=args.seed
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    heldout_loader = DataLoader(
        Subset(dataset, heldout_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor, classifier = image_models.build_model(num_classes)
    feature_extractor.to(device)
    classifier.to(device)
    optimizer = optim.Adam(
        [*feature_extractor.parameters(), *classifier.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[24], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    metrics = []

    for epoch in range(args.epochs):
        feature_extractor.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = classifier(feature_extractor(inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(inputs.size(0))
            train_correct += int(outputs.argmax(dim=1).eq(targets).sum().item())
            train_total += int(targets.size(0))

        feature_extractor.eval()
        classifier.eval()
        heldout_loss = 0.0
        heldout_correct = 0
        heldout_total = 0
        with torch.no_grad():
            for inputs, targets in heldout_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = classifier(feature_extractor(inputs))
                loss = criterion(outputs, targets)
                heldout_loss += float(loss.item()) * int(inputs.size(0))
                heldout_correct += int(outputs.argmax(dim=1).eq(targets).sum().item())
                heldout_total += int(targets.size(0))
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_total,
            "train_acc": train_correct / train_total,
            "val_loss": heldout_loss / heldout_total,
            "val_acc": heldout_correct / heldout_total,
        }
        metrics.append(row)
        print(
            f"Epoch {epoch + 1}: Train Loss {row['train_loss']:.4f}, "
            f"Train Acc {row['train_acc']:.4f}, Val Loss {row['val_loss']:.4f}, "
            f"Val Acc {row['val_acc']:.4f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(feature_extractor.state_dict(), args.output_dir / "FE.pth")
    torch.save(classifier.state_dict(), args.output_dir / "CF.pth")
    (args.output_dir / "data_split.json").write_text(
        json.dumps(
            {
                "clf_train_idx": train_indices,
                "clf_val_idx": heldout_indices,
                "arch": "resnet18",
                "dataset": args.dataset,
                "augmentation": args.augmentation,
                "seed": args.seed,
            }
        ),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
