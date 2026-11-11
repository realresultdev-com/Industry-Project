"""
Unsupervised Anomaly Detection via Self-Supervised Feature Memory Bank
======================================================================
Detects industrial structural defects without requiring labeled anomaly classes:
1. Extracts representations of reference 'normal/flawless' steel surfaces into a feature bank.
2. For test images, computes feature-based k-NN cosine and Euclidean distance to the bank.
3. High distance indicates anomalous structural degradation or defect.
4. Computes AUROC and plots the ROC curve.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

from data.augmentations import get_eval_transform
from models.backbone import get_backbone
from utils.metrics import compute_knn_anomaly_scores, plot_roc_curve


def extract_features(encoder, loader, device):
    encoder.eval()
    feats_list = []
    targets_list = []
    
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            feats = F.normalize(feats, dim=-1)
            feats_list.append(feats.cpu().numpy())
            targets_list.append(targets.numpy())

    return np.concatenate(feats_list, axis=0), np.concatenate(targets_list, axis=0)


def run_anomaly_detection(
    checkpoint_path: str = "checkpoints/simclr_resnet18_backbone.pth",
    data_dir: str = "data/splits",
    output_dir: str = "results",
    k_neighbors: int = 5,
    device: str = "cpu"
):
    print("=" * 70)
    print("      UNSUPERVISED FEATURE-BASED ANOMALY DETECTION")
    print("=" * 70)
    print(f"[*] Checkpoint : {checkpoint_path}")
    print(f"[*] Data Dir   : {data_dir}")
    print(f"[*] k-NN k     : {k_neighbors}")
    print(f"[*] Device     : {device}")

    dev = torch.device(device)
    encoder, dim = get_backbone("resnet18", pretrained=False)
    
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        encoder.load_state_dict(state, strict=False)
        print(f"[+] Successfully loaded SSL backbone weights from: {ckpt}")
    else:
        print(f"[!] Warning: {ckpt} not found! Using random initialization.")
    
    encoder.to(dev)

    # Load data
    t_eval = get_eval_transform(200)
    train_set = ImageFolder(str(Path(data_dir) / "train_100pct"), transform=t_eval)
    test_set = ImageFolder(str(Path(data_dir) / "test"), transform=t_eval)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    train_feats, train_labels = extract_features(encoder, train_loader, dev)
    test_feats, test_labels = extract_features(encoder, test_loader, dev)

    # Class 0 is treated as nominal/reference class (e.g. baseline texture)
    # Binary ground truth: 0 = Nominal, 1 = Defective / Anomalous
    ref_mask = (train_labels == 0)
    normal_bank = train_feats[ref_mask]
    
    test_binary_truth = (test_labels != 0).astype(int)

    print(f"[+] Normal Feature Memory Bank Size : {normal_bank.shape[0]} samples")
    print(f"[+] Test Samples Evaluated           : {test_feats.shape[0]} ({sum(test_binary_truth == 0)} normal, {sum(test_binary_truth == 1)} anomalous)")

    # Compute anomaly scores
    anomaly_scores = compute_knn_anomaly_scores(normal_bank, test_feats, k=k_neighbors)

    # Plot ROC curve and calculate AUROC
    out_plot = Path(output_dir) / "anomaly_roc_curve.png"
    auroc = plot_roc_curve(test_binary_truth, anomaly_scores, str(out_plot))

    # Save summary JSON
    results = {
        "auroc": float(auroc),
        "k_neighbors": k_neighbors,
        "normal_bank_size": int(normal_bank.shape[0]),
        "total_test_samples": int(len(test_binary_truth)),
        "nominal_test_samples": int(sum(test_binary_truth == 0)),
        "defect_test_samples": int(sum(test_binary_truth == 1))
    }
    json_path = Path(output_dir) / "anomaly_detection_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 70)
    print(f"[+] Final Anomaly Detection AUROC : {auroc:.4f}")
    print(f"[+] Anomaly metrics saved to      : {json_path}")
    print("=" * 70)
    return auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simclr_resnet18_backbone.pth")
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_anomaly_detection(args.checkpoint, args.data_dir, args.output_dir, args.k, args.device)

