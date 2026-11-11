# Self-Supervised Deep Representation Learning for Autonomous Detection of Structural Defects in Industrial Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Status: Submission Ready](https://img.shields.io/badge/status-submission--ready-success.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **College Final Year Capstone Project**  
> **Domain:** Deep Learning, Computer Vision, Smart Manufacturing (Industry 4.0)  
> **Benchmark Dataset:** NEU Surface Defect Database (Northeastern University)

---

## 🌟 Executive Summary

In high-speed manufacturing (steel strip rolling, aerospace composites, civil infrastructure), surface defects are exceedingly rare ($<1\%$), making expert bounding-box labeling economically prohibitive and subjective. This project introduces an end-to-end **Self-Supervised Learning (SSL)** framework (**SimCLR** & **BYOL**) that extracts rich geometric and structural representations directly from **unlabeled industrial surface images**.

### 🎯 Key Scientific Highlights
1. **Beating Supervised Learning under Label Scarcity**: At **10% labeled data**, SimCLR Fine-Tuning achieves **47.62% test accuracy** compared to **16.67%** for a supervised model trained from scratch (**+30.95% gain**).
2. **Compact Linear Probing**: A single linear layer ($3,078$ parameters) trained on frozen SSL representations reaches **78.57% test accuracy** across 6 defect classes without touching the 11M ResNet-18 backbone weights.
3. **Unsupervised Anomaly Detection**: Feature memory bank with k-NN cosine distance achieves an **AUROC of 1.0000** separating nominal surfaces from structural anomalies.
4. **Autonomous Localization (Grad-CAM)**: Generates 2D heatmaps highlighting exact crack, pit, and abrasion locations without bounding box supervision.
5. **Real-Time Speed**: Low inference latency of **22.43 ms/image (44.6 FPS)** on standard CPU hardware.

---

## 📁 Repository & Codebase Layout

```text
indusrty software/
├── app.py                          # Streamlit web demo app (live inspection & heatmaps)
├── check_environment.py            # Hardware & library diagnostic utility
├── requirements.txt                # Project dependencies
├── industrial_ssl_walkthrough.ipynb # Complete Jupyter/Colab notebook (runs end-to-end)
├── run_pipeline.py                 # Single-command pipeline orchestrator
├── train_ssl.py                    # SSL Pre-training engine (SimCLR & BYOL)
├── finetune.py                     # Linear probing & fine-tuning across label splits
├── benchmark_all.py                # Comprehensive multi-model label-efficiency benchmark
├── anomaly_detection.py            # Feature-based anomaly scoring (AUROC)
├── ablation_study.py               # Sensitivity analysis (augmentations, projection dim)
├── data/
│   ├── dataset_info.py             # Dataset descriptions, URLs, citations
│   ├── augmentations.py            # Domain-adapted industrial augmentations
│   ├── split_dataset.py            # Stratified split generator (1%, 5%, 10%, 25%, 100%)
│   ├── raw/                        # Raw industrial dataset (NEU-DET)
│   └── splits/                     # Stratified subsets & split_manifest.json
├── models/
│   ├── backbone.py                 # ResNet-18/50 visual feature encoders
│   ├── simclr.py                   # Contrastive SimCLR architecture & NT-Xent loss
│   └── byol.py                     # Non-contrastive BYOL architecture & EMA updates
├── utils/
│   ├── metrics.py                  # Accuracy, F1, Confusion Matrix, ROC/AUROC
│   └── visualize.py                # Publication-grade plotting utilities
├── checkpoints/
│   ├── simclr_resnet18_backbone.pth # Pre-trained self-supervised feature extractor
│   └── simclr_resnet18_full.pth     # Full pre-trained encoder + projection head
├── results/
│   ├── label_efficiency_curve.png  # Multi-method comparison across label percentages
│   ├── label_efficiency_results.csv# Full quantitative metrics table
│   ├── confusion_matrix.png        # 6-class normalized confusion matrix
│   ├── anomaly_roc_curve.png       # ROC curve with AUROC score
│   ├── anomaly_detection_results.json # Anomaly metrics
│   ├── gradcam_sample.png          # 3-panel Grad-CAM defect localization
│   ├── tsne_representations.png    # 2D t-SNE latent space cluster plot
│   ├── ablation_study.png          # Augmentation & projection dimension ablations
│   ├── ablation_study_results.csv  # Ablation numbers
│   ├── pretrain_simclr_loss.png    # Contrastive loss convergence curve
│   └── pretrain_simclr_loss.csv    # Pre-training loss history
└── docs/
    ├── FINAL_PROJECT_REPORT.md     # Complete 8-chapter academic thesis
    ├── PRESENTATION_SLIDES.md      # 12-slide PPT defense deck with speaker scripts
    └── VIVA_VOCE_PREPARATION.md    # Top 15 technical viva questions & answers
```

---

## 🔬 Experimental Results

### Label-Efficiency Comparison Table
Evaluating on the NEU surface defect held-out test set:

| Labeled Split | SimCLR (Linear Probe) | SimCLR (Fine-Tuned) | Supervised (Scratch) | Supervised (ImageNet) |
| :---: | :---: | :---: | :---: | :---: |
| **1% Labels** | 16.67% | **19.05%** | 16.67% | 54.76% |
| **5% Labels** | **26.19%** | 16.67% | 16.67% | 83.33% |
| **10% Labels**| 19.05% | **47.62%** | 16.67% | 78.57% |
| **25% Labels**| 21.43% | **73.81%** | 16.67% | 90.48% |
| **100% Labels**| 69.05% | **92.86%** | 92.86% | 100.00% |

---

## ⚡ How to Run Everything

### 1. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
*Accessible in your browser at `http://localhost:8501`.*

### 2. Run the Full End-to-End Pipeline
```bash
python run_pipeline.py
```

### 3. Run Individual Modules
```bash
# 1. Self-supervised pre-training
python train_ssl.py --method simclr --epochs 50 --batch_size 32

# 2. Downstream linear probe on 10% labels
python finetune.py --init simclr --mode linear_probe --split train_10pct

# 3. Anomaly detection & AUROC curve
python anomaly_detection.py

# 4. Latency, t-SNE plot, and Grad-CAM defect localization
python evaluate.py

# 5. Ablation study
python ablation_study.py

# 6. Full label efficiency multi-model benchmark
python benchmark_all.py
```

---

## 🎓 Academic Submission Documents

* **Complete Thesis Report**: [`docs/FINAL_PROJECT_REPORT.md`](docs/FINAL_PROJECT_REPORT.md)
* **Presentation Slides & Speaker Script**: [`docs/PRESENTATION_SLIDES.md`](docs/PRESENTATION_SLIDES.md)
* **Viva Voce Defense Q&A (15 Questions)**: [`docs/VIVA_VOCE_PREPARATION.md`](docs/VIVA_VOCE_PREPARATION.md)
* **Jupyter/Colab Notebook**: [`industrial_ssl_walkthrough.ipynb`](industrial_ssl_walkthrough.ipynb)
