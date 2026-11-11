"""
Downstream Defect Classification: Linear Probing & Full Fine-Tuning
===================================================================
Benchmarking Visual Representations Across Label Scarcity Ratios:
- Evaluates:
  1. Scratch (Randomly Initialized ResNet)
  2. ImageNet-Pretrained Supervised ResNet
  3. SimCLR Pre-trained (Self-Supervised)
  4. BYOL Pre-trained (Self-Supervised)
- Across Labeled Data Splits:
  1%, 5%, 10%, 25%, 100%
- Modes:
  - linear_probe: Freeze backbone, train only the linear classification layer.
  - full_finetune: Fine-tune the entire model end-to-end.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import json
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from data.augmentations import get_eval_transform, NORM_MEAN, NORM_STD
from models.backbone import get_backbone


class DefectClassifier(nn.Module):
    """
    Downstream classification model combining an encoder backbone with a linear head.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 6,
        init_type: str = "scratch",
        checkpoint_path: str = None,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.init_type = init_type
        self.freeze_backbone = freeze_backbone
        
        # 1. Initialize backbone
        pretrained_imagenet = (init_type.lower() == "imagenet")
        self.encoder, self.feature_dim = get_backbone(backbone_name, pretrained=pretrained_imagenet)
        
        # Load SSL checkpoint if specified
        if init_type.lower() in ["simclr", "byol"] and checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                print(f"[+] Loading SSL weights from: {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                self.encoder.load_state_dict(state, strict=False)
            else:
                print(f"[!] Warning: Checkpoint {ckpt_path} not found! Falling back to scratch.")

        # 2. Freeze backbone if linear probing
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
                
        # 3. Linear classification head
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        logits = self.fc(feat)
        return logits


def get_train_transform(image_size: int = 200):
    """
    Standard supervised data augmentation for fine-tuning.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


def evaluate(model, loader, device):
    """
    Evaluates model on validation/test set.
    Returns: accuracy, loss, all_preds, all_labels
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return acc, avg_loss, all_preds, all_targets


def compute_metrics(targets, preds):
    """
    Computes precision, recall, and macro F1 score.
    """
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(targets, preds, average="macro", zero_division=0)
        rec = recall_score(targets, preds, average="macro", zero_division=0)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
    except ImportError:
        # Fallback pure Python computation
        classes = sorted(list(set(targets + preds)))
        f1_list = []
        for c in classes:
            tp = sum(1 for t, p in zip(targets, preds) if t == c and p == c)
            fp = sum(1 for t, p in zip(targets, preds) if t != c and p == c)
            fn = sum(1 for t, p in zip(targets, preds) if t == c and p != c)
            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_c = 2 * (p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0
            f1_list.append(f1_c)
        prec, rec = 0.0, 0.0
        f1 = sum(f1_list) / max(1, len(f1_list))
    return prec, rec, f1


def main():
    parser = argparse.ArgumentParser(description="Downstream Defect Classification Benchmarking")
    parser.add_argument("--init", type=str, default="simclr",
                        choices=["scratch", "imagenet", "simclr", "byol"],
                        help="Weight initialization source")
    parser.add_argument("--mode", type=str, default="linear_probe",
                        choices=["linear_probe", "full_finetune"],
                        help="Training mode: linear_probe (freeze backbone) or full_finetune")
    parser.add_argument("--split", type=str, default="train_10pct",
                        choices=["train_1pct", "train_5pct", "train_10pct", "train_25pct", "train_100pct"],
                        help="Labeled subset split to train on")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simclr_resnet18_backbone.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=200)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_root = Path(args.data_dir)
    train_dir = data_root / args.split
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    print("=" * 70)
    print("      DOWNSTREAM DEFECT CLASSIFICATION EXPERIMENT")
    print("=" * 70)
    print(f"[*] Initialization : {args.init.upper()}")
    print(f"[*] Training Mode  : {args.mode.upper()}")
    print(f"[*] Labeled Split  : {args.split}")
    print(f"[*] Backbone       : {args.backbone}")
    print(f"[*] Device         : {device}")
    print(f"[*] Epochs         : {args.epochs}")

    # Datasets
    t_train = get_train_transform(args.image_size)
    t_eval = get_eval_transform(args.image_size)

    train_set = ImageFolder(str(train_dir), transform=t_train)
    val_set = ImageFolder(str(val_dir), transform=t_eval)
    test_set = ImageFolder(str(test_dir), transform=t_eval)

    num_classes = len(train_set.classes)
    print(f"[+] Classes ({num_classes}): {train_set.classes}")
    print(f"[+] Labeled Train Images : {len(train_set)}")
    print(f"[+] Val Images           : {len(val_set)}")
    print(f"[+] Test Images          : {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    freeze = (args.mode == "linear_probe")
    model = DefectClassifier(
        backbone_name=args.backbone,
        num_classes=num_classes,
        init_type=args.init,
        checkpoint_path=args.checkpoint,
        freeze_backbone=freeze
    ).to(device)

    # Trainable parameters check
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[+] Trainable Parameters : {sum(p.numel() for p in trainable_params):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training Loop
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        val_acc, val_loss, _, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            train_loss = running_loss / max(1, len(train_set))
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    # Load best checkpoint for final test evaluation
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_acc, test_loss, preds, targets = evaluate(model, test_loader, device)
    prec, rec, f1 = compute_metrics(targets, preds)

    print("-" * 70)
    print("FINAL TEST EVALUATION RESULTS:")
    print(f"  Accuracy  : {test_acc * 100:.2f}%")
    print(f"  Macro-F1  : {f1 * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print("-" * 70)

    # Log to Results CSV
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / "classification_benchmarks.csv"
    
    file_exists = csv_file.exists()
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Initialization", "Mode", "Split", "Backbone", "Epochs", "Accuracy", "F1-Score", "Precision", "Recall"])
        writer.writerow([
            args.init,
            args.mode,
            args.split,
            args.backbone,
            args.epochs,
            f"{test_acc * 100:.2f}",
            f"{f1 * 100:.2f}",
            f"{prec * 100:.2f}",
            f"{rec * 100:.2f}"
        ])
    print(f"[+] Benchmark appended to: {csv_file}")


if __name__ == "__main__":
    main()

