"""
Evaluation Metrics & Anomaly Scoring Utilities
==============================================
Provides functions for:
- Multi-class classification metrics (Accuracy, Macro-F1, Precision, Recall)
- Confusion Matrix computation and plotting
- Anomaly Detection scoring via k-NN Cosine / Mahalanobis distance
- Receiver Operating Characteristic (ROC) and Area Under ROC (AUROC)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)


def compute_classification_metrics(y_true, y_pred):
    """
    Computes accuracy, macro precision, recall, and F1 score.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str):
    """
    Renders and saves a normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Format ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Normalized Confusion Matrix (Defect Classification)",
        ylabel="True Defect Class",
        xlabel="Predicted Defect Class"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add numeric labels
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {output_path}")


def compute_knn_anomaly_scores(normal_features, test_features, k: int = 5):
    """
    Computes anomaly scores based on the mean cosine distance to the k-nearest normal samples.
    A higher distance indicates an anomaly (defect).
    """
    # Normalize features to unit sphere
    norm_ref = normal_features / (np.linalg.norm(normal_features, axis=1, keepdims=True) + 1e-8)
    norm_test = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix: (N_test, N_ref)
    sim_matrix = np.dot(norm_test, norm_ref.T)
    # Cosine distance = 1 - similarity
    dist_matrix = 1.0 - sim_matrix

    # Sort distances to find k-nearest neighbors
    dist_matrix.sort(axis=1)
    k_actual = min(k, dist_matrix.shape[1])
    anomaly_scores = dist_matrix[:, :k_actual].mean(axis=1)
    return anomaly_scores


def plot_roc_curve(y_true_binary, anomaly_scores, output_path: str):
    """
    Computes AUROC and plots the ROC curve for anomaly detection.
    """
    fpr, tpr, _ = roc_curve(y_true_binary, anomaly_scores)
    auroc = roc_auc_score(y_true_binary, anomaly_scores)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'SSL Anomaly Detector (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess (AUROC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) - Defect Anomaly Detection', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Anomaly ROC curve saved to: {output_path} (AUROC: {auroc:.4f})")
    return auroc

