"""
End-to-End Pipeline Execution Script
====================================
Runs the complete research workflow:
1. Verifies dataset splits
2. Trains SimCLR representation model
3. Evaluates downstream linear probe & fine-tuning across label splits (1%, 10%, 100%)
4. Generates evaluation metrics, t-SNE latent space plot, and Grad-CAM defect heatmap
"""

import sys
import os
import subprocess
import time
from pathlib import Path

PYTHON_EXE = sys.executable

def run_step(cmd_args, step_name):
    print("=" * 75)
    print(f"STEP: {step_name}")
    print("COMMAND:", " ".join(cmd_args))
    print("=" * 75)
    start_t = time.time()
    result = subprocess.run([PYTHON_EXE] + cmd_args)
    elapsed = time.time() - start_t
    if result.returncode != 0:
        print(f"[!] Error in {step_name} (Exit code: {result.returncode})")
        sys.exit(result.returncode)
    print(f"[+] Completed {step_name} in {elapsed:.1f}s\n")

def main():
    print("""
===========================================================================
  SELF-SUPERVISED DEEP REPRESENTATION LEARNING FOR INDUSTRIAL DEFECTS
               AUTONOMOUS END-TO-END PIPELINE RUNNER
===========================================================================
    """)

    # 1. Environment Verification
    run_step(["check_environment.py"], "1. Environment Diagnostics")

    # 2. Data Splits Preparation (if not already prepared)
    if not Path("data/splits/split_manifest.json").exists():
        run_step(["data/split_dataset.py", "--mock"], "2. Dataset Partitioning (Synthetic Demo Mode)")
    else:
        print("[+] Existing data splits found in data/splits/. Skipping data partitioning.\n")

    # 3. Self-Supervised Pre-Training (SimCLR)
    # Using 5 epochs for immediate demonstration; can be increased to 50-100 on GPU
    run_step([
        "train_ssl.py",
        "--method", "simclr",
        "--backbone", "resnet18",
        "--epochs", "5",
        "--batch_size", "16",
        "--lr", "3e-4"
    ], "3. Self-Supervised Pre-Training (SimCLR with ResNet-18)")

    # 4. Downstream Benchmarking: Compare SSL vs Supervised Baseline under label scarcity
    splits_to_test = ["train_10pct", "train_100pct"]
    for split in splits_to_test:
        # Linear Probe (Frozen SSL features)
        run_step([
            "finetune.py",
            "--init", "simclr",
            "--mode", "linear_probe",
            "--split", split,
            "--epochs", "10",
            "--checkpoint", "checkpoints/simclr_resnet18_backbone.pth"
        ], f"4a. Downstream Linear Probe [SimCLR Pre-trained] on {split}")

        # Supervised Baseline trained from scratch (random weights)
        run_step([
            "finetune.py",
            "--init", "scratch",
            "--mode", "full_finetune",
            "--split", split,
            "--epochs", "10"
        ], f"4b. Supervised Baseline [From Scratch] on {split}")

    # 5. Representation Evaluation, t-SNE Plot & Grad-CAM Localization
    run_step([
        "evaluate.py",
        "--backbone", "resnet18",
        "--checkpoint", "checkpoints/simclr_resnet18_backbone.pth"
    ], "5. Representation Evaluation, Latency, t-SNE & Grad-CAM Heatmap")

    print("=" * 75)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("Results and visualizations saved in: results/")
    print("To launch the interactive dashboard, run:")
    print("    streamlit run app.py")
    print("=" * 75)

if __name__ == "__main__":
    main()

