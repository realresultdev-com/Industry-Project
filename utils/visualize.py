"""
Scientific Plotting & Visualization Suite
=========================================
Generates presentation-grade figures for project report and defense:
1. Label efficiency benchmark curves (1%, 5%, 10%, 25%, 100%)
2. Ablation study plots (Augmentations, Projection Dimension, Batch Size)
3. Pre-training loss comparison (SimCLR vs BYOL)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def plot_label_efficiency_curve(results_csv: str, output_path: str):
    """
    Renders publication-quality multi-method comparison across label efficiency splits.
    """
    df = pd.read_csv(results_csv)
    
    # Expected columns: Method, Split, Accuracy
    split_order = ["1%", "5%", "10%", "25%", "100%"]
    
    plt.figure(figsize=(9, 6))
    colors = {
        "SimCLR (Linear Probe)": "#2563eb",
        "SimCLR (Fine-Tuned)": "#1d4ed8",
        "BYOL (Linear Probe)": "#10b981",
        "Supervised (Scratch)": "#ef4444",
        "Supervised (ImageNet)": "#f59e0b"
    }
    markers = {
        "SimCLR (Linear Probe)": "o",
        "SimCLR (Fine-Tuned)": "s",
        "BYOL (Linear Probe)": "^",
        "Supervised (Scratch)": "x",
        "Supervised (ImageNet)": "d"
    }

    for method in df["Method"].unique():
        sub_df = df[df["Method"] == method]
        # Align by split order
        sub_df = sub_df.set_index("Split").reindex(split_order).reset_index()
        plt.plot(
            sub_df["Split"],
            sub_df["Accuracy"],
            marker=markers.get(method, "o"),
            color=colors.get(method, "#6b7280"),
            linewidth=2.5,
            markersize=8,
            label=method
        )

    plt.title("Label-Efficiency: Defect Classification Accuracy vs. Label Availability", fontsize=13, fontweight='bold')
    plt.xlabel("Percentage of Labeled Training Samples Available", fontsize=11, fontweight='bold')
    plt.ylabel("Test Classification Accuracy (%)", fontsize=11, fontweight='bold')
    plt.ylim([0, 105])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Label efficiency curve saved to: {output_path}")


def plot_ablation_study(ablation_csv: str, output_path: str):
    """
    Plots multi-panel ablation study showing impact of:
    - Domain Augmentations
    - Projection Head Dimension
    - Temperature / Loss Parameters
    """
    df = pd.read_csv(ablation_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Augmentation Ablation
    aug_df = df[df["Category"] == "Augmentation"]
    if not aug_df.empty:
        bars1 = axes[0].bar(aug_df["Setting"], aug_df["Accuracy"], color="#3b82f6", width=0.55)
        axes[0].set_title("Effect of Industrial Augmentations", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Linear Probe Accuracy (%)", fontsize=11)
        axes[0].set_ylim([0, 100])
        axes[0].grid(axis='y', linestyle='--', alpha=0.6)
        for bar in bars1:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Projection Dimension Ablation
    dim_df = df[df["Category"] == "Projection_Dim"]
    if not dim_df.empty:
        bars2 = axes[1].bar(dim_df["Setting"].astype(str), dim_df["Accuracy"], color="#10b981", width=0.55)
        axes[1].set_title("Effect of Projection Head Dimension (z)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Linear Probe Accuracy (%)", fontsize=11)
        axes[1].set_xlabel("Projection Dimension", fontsize=11)
        axes[1].set_ylim([0, 100])
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)
        for bar in bars2:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Ablation study plot saved to: {output_path}")

