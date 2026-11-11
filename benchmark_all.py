"""
Full Label-Efficiency Benchmarking Suite
========================================
Systematically benchmarks representation models across all labeled data fractions:
- 1% labels  (~1-2 images/class)
- 5% labels  (~2-3 images/class)
- 10% labels (~4 images/class)
- 25% labels (~9 images/class)
- 100% labels (~36 images/class)

Methods compared:
1. SimCLR (Linear Probe)
2. SimCLR (Full Fine-Tune)
3. Supervised (from Scratch)
4. Supervised (ImageNet Pretrained)

Produces:
- results/label_efficiency_results.csv
- results/label_efficiency_curve.png
- results/confusion_matrix.png
"""

import os
import sys
import argparse
from pathlib import Path
import csv
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.augmentations import get_eval_transform, NORM_MEAN, NORM_STD
from models.backbone import get_backbone
from finetune import DefectClassifier, get_train_transform, evaluate, compute_metrics
from utils.metrics import plot_confusion_matrix
from utils.visualize import plot_label_efficiency_curve


def train_and_eval_model(
    init_type: str,
    mode: str,
    split_name: str,
    checkpoint_path: str,
    data_dir: Path,
    epochs: int = 15,
    device: str = "cpu",
    batch_size: int = 16
):
    train_dir = data_dir / split_name
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    t_train = get_train_transform(200)
    t_eval = get_eval_transform(200)

    train_set = ImageFolder(str(train_dir), transform=t_train)
    val_set = ImageFolder(str(val_dir), transform=t_eval)
    test_set = ImageFolder(str(test_dir), transform=t_eval)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    freeze = (mode == "linear_probe")
    model = DefectClassifier(
        backbone_name="resnet18",
        num_classes=len(train_set.classes),
        init_type=init_type,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3 if freeze else 3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

        val_acc, _, _, _ = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    test_acc, _, preds, targets = evaluate(model, test_loader, device)
    metrics = compute_metrics(targets, preds)
    return test_acc * 100.0, metrics, preds, targets, train_set.classes


def run_full_benchmark(
    checkpoint_path: str = "checkpoints/simclr_resnet18_backbone.pth",
    data_dir_str: str = "data/splits",
    output_dir_str: str = "results",
    epochs_per_run: int = 12,
    device: str = "cpu"
):
    print("=" * 75)
    print("     FULL LABEL-EFFICIENCY & MULTI-MODEL BENCHMARKING SUITE")
    print("=" * 75)

    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("1%", "train_1pct"),
        ("5%", "train_5pct"),
        ("10%", "train_10pct"),
        ("25%", "train_25pct"),
        ("100%", "train_100pct")
    ]

    methods = [
        ("SimCLR (Linear Probe)", "simclr", "linear_probe"),
        ("SimCLR (Fine-Tuned)", "simclr", "full_finetune"),
        ("Supervised (Scratch)", "scratch", "full_finetune"),
        ("Supervised (ImageNet)", "imagenet", "full_finetune")
    ]

    records = []
    final_preds, final_targets, class_names = None, None, None

    for split_label, split_folder in splits:
        print(f"\n---> Evaluating Split: {split_label} ({split_folder})")
        for method_name, init_type, mode in methods:
            print(f"     [*] Running {method_name:<26} ... ", end="", flush=True)
            acc, m, preds, targets, classes = train_and_eval_model(
                init_type=init_type,
                mode=mode,
                split_name=split_folder,
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                epochs=epochs_per_run,
                device=device
            )
            print(f"Accuracy: {acc:.2f}% | F1: {m[2]*100:.2f}%")
            records.append({
                "Method": method_name,
                "Split": split_label,
                "Accuracy": round(acc, 2),
                "F1_Score": round(m[2] * 100, 2),
                "Precision": round(m[0] * 100, 2),
                "Recall": round(m[1] * 100, 2)
            })

            # Save 100% SimCLR Linear probe for Confusion Matrix
            if split_label == "100%" and method_name == "SimCLR (Linear Probe)":
                final_preds = preds
                final_targets = targets
                class_names = classes

    # Save to CSV
    csv_path = output_dir / "label_efficiency_results.csv"
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Full benchmark table saved to: {csv_path}")

    # Generate Label Efficiency Plot
    plot_path = output_dir / "label_efficiency_curve.png"
    plot_label_efficiency_curve(str(csv_path), str(plot_path))

    # Generate Confusion Matrix
    if final_preds is not None:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(final_targets, final_preds, class_names, str(cm_path))

    print("=" * 75)
    print("[+] ALL BENCHMARK PLOTS & TABLES EXPORTED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simclr_resnet18_backbone.pth")
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_full_benchmark(args.checkpoint, args.data_dir, args.output_dir, args.epochs, args.device)

