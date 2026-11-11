"""
Self-Supervised Learning Pre-Training Pipeline
==============================================
Trains ResNet backbone on unlabeled industrial images using:
- Method 1: SimCLR (Contrastive Learning with NT-Xent loss)
- Method 2: BYOL (Non-contrastive bootstrap learning with EMA target)

Saves:
- Checkpoints to: checkpoints/{method}_{backbone}.pth
- Extracted backbone to: checkpoints/{method}_{backbone}_backbone.pth
- Training loss curves to: results/pretrain_{method}_loss.png and .csv
"""

import os
import sys
import time
import argparse
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.augmentations import (
    TwoCropsTransform,
    get_industrial_ssl_transform,
    UnlabeledIndustrialDataset
)
from models import SimCLR, NTXentLoss, BYOL


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Supervised Pre-Training on Industrial Images")
    parser.add_argument("--method", type=str, default="simclr", choices=["simclr", "byol"],
                        help="Self-supervised algorithm to use")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"],
                        help="Backbone architecture")
    parser.add_argument("--data_dir", type=str, default="data/splits/unlabeled_pretrain",
                        help="Directory containing unlabeled images")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Base learning rate (AdamW)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--proj_dim", type=int, default=128,
                        help="Dimension of projection space")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature parameter for SimCLR NT-Xent loss")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="EMA decay rate for BYOL target network")
    parser.add_argument("--image_size", type=int, default=200,
                        help="Image input resolution")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save metrics and plots")
    parser.add_argument("--device", type=str, default="auto",
                        help="Computation device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def plot_loss_curve(epoch_losses, output_path):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='#2563eb', linewidth=2)
        plt.title("Self-Supervised Pre-training Loss Convergence", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[!] Warning: Could not save loss curve plot ({e}). Loss values are saved in CSV.")


def main():
    args = parse_args()

    # 1. Reproducibility & Device setup
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("      SELF-SUPERVISED INDUSTRIAL DEFECT PRE-TRAINING")
    print("=" * 70)
    print(f"[*] Method       : {args.method.upper()}")
    print(f"[*] Backbone     : {args.backbone}")
    print(f"[*] Device       : {device}")
    print(f"[*] Epochs       : {args.epochs}")
    print(f"[*] Batch Size   : {args.batch_size}")
    print(f"[*] Learning Rate: {args.lr}")
    print(f"[*] Data Source  : {args.data_dir}")

    # Output directories
    save_dir = Path(args.save_dir)
    results_dir = Path(args.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataset & DataLoader
    ssl_transform = TwoCropsTransform(get_industrial_ssl_transform(args.image_size))
    dataset = UnlabeledIndustrialDataset(args.data_dir, transform=ssl_transform)
    print(f"[+] Loaded {len(dataset)} unlabeled industrial images for pre-training.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True if len(dataset) > args.batch_size else False,
        num_workers=0,  # 0 for maximum Windows compatibility
        pin_memory=(device.type == "cuda")
    )

    # 3. Model & Loss Selection
    if args.method == "simclr":
        model = SimCLR(backbone_name=args.backbone, proj_dim=args.proj_dim).to(device)
        criterion = NTXentLoss(temperature=args.temperature).to(device)
    elif args.method == "byol":
        model = BYOL(
            backbone_name=args.backbone,
            proj_dim=args.proj_dim,
            ema_decay=args.ema_decay
        ).to(device)
        criterion = None

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training Loop
    print("-" * 70)
    print("[*] Starting Pre-Training...")
    epoch_losses = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        for (v1, v2), _ in loader:
            v1 = v1.to(device)
            v2 = v2.to(device)

            optimizer.zero_grad()

            if args.method == "simclr":
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = criterion(z1, z2)
            elif args.method == "byol":
                loss, _ = model(v1, v2)

            loss.backward()
            optimizer.step()

            if args.method == "byol":
                model.update_target_network()

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(1, steps)
        epoch_losses.append(avg_loss)
        scheduler.step()

        # Display progress
        current_lr = scheduler.get_last_lr()[0]
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Elapsed: {elapsed:.1f}s")

    # 5. Save Checkpoints & Metrics
    print("-" * 70)
    ckpt_full_path = save_dir / f"{args.method}_{args.backbone}_full.pth"
    ckpt_backbone_path = save_dir / f"{args.method}_{args.backbone}_backbone.pth"

    # Save full SSL model (encoder + projector)
    torch.save(model.state_dict(), ckpt_full_path)
    
    # Save standalone backbone encoder (for downstream fine-tuning & evaluation)
    if args.method == "simclr":
        backbone_state = model.encoder.state_dict()
    elif args.method == "byol":
        backbone_state = model.online_encoder.state_dict()
    torch.save(backbone_state, ckpt_backbone_path)

    print(f"[+] Full SSL Checkpoint saved to: {ckpt_full_path}")
    print(f"[+] Feature Extractor saved to  : {ckpt_backbone_path}")

    # Save Loss CSV
    csv_path = results_dir / f"pretrain_{args.method}_loss.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for ep, l_val in enumerate(epoch_losses, 1):
            writer.writerow([ep, f"{l_val:.6f}"])
    print(f"[+] Loss history saved to      : {csv_path}")

    # Save Plot
    png_path = results_dir / f"pretrain_{args.method}_loss.png"
    plot_loss_curve(epoch_losses, png_path)
    print(f"[+] Loss plot saved to         : {png_path}")
    print("=" * 70)
    print("[+] Pre-training completed successfully!")


if __name__ == "__main__":
    main()

