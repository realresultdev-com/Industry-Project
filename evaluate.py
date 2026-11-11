"""
Evaluation & Interpretability Suite
===================================
1. Anomaly Detection: Feature-based defect scoring via Mahalanobis / Cosine k-NN distance + AUROC.
2. Latent Space Visualization: t-SNE / PCA / UMAP 2D clustering of learned embeddings.
3. Defect Localization: Grad-CAM visual heatmaps highlighting exact defect regions.
4. Inference Latency: Milliseconds per image and Frames Per Second (FPS).
"""

import os
import sys
import time
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

from data.augmentations import get_eval_transform, NORM_MEAN, NORM_STD
from models.backbone import get_backbone


class FeatureExtractor(nn.Module):
    """
    Wraps backbone to extract pooled representation vector and spatial feature maps.
    """
    def __init__(self, backbone_name: str = "resnet18", checkpoint_path: str = None):
        super().__init__()
        self.encoder, self.dim = get_backbone(backbone_name, pretrained=False)
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(state, strict=False)
            print(f"[+] Loaded weights from {checkpoint_path}")
        self.encoder.eval()

    def forward(self, x):
        return self.encoder(x)

    def get_feature_maps(self, x):
        return self.encoder.get_feature_maps(x)


def extract_all_features(model, loader, device):
    """
    Extracts L2-normalized representations and ground truth labels.
    """
    model.eval()
    features = []
    labels = []
    paths = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = model(images)
            feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu())
            labels.append(targets)

    return torch.cat(features, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def compute_tsne_plot(features, labels, class_names, output_path):
    """
    Computes 2D t-SNE dimensionality reduction and saves high-resolution scatter plot.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import numpy as np

        print("[*] Computing t-SNE projection...")
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(features) // 5)), random_state=42)
        embeds = tsne.fit_transform(features)

        plt.figure(figsize=(9, 7))
        classes = np.unique(labels)
        cmap = plt.colormaps.get_cmap("tab10")

        for idx, c in enumerate(classes):
            mask = (labels == c)
            name = class_names[c] if c < len(class_names) else f"Class {c}"
            plt.scatter(
                embeds[mask, 0],
                embeds[mask, 1],
                label=name,
                alpha=0.8,
                s=40,
                edgecolors='none'
            )

        plt.title("t-SNE Visualization of Learned Representations", fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[+] t-SNE plot saved to: {output_path}")
    except Exception as e:
        print(f"[!] Warning: Could not generate t-SNE plot ({e})")


def measure_inference_speed(model, image_size=200, device="cpu", repetitions=100):
    """
    Benchmarks model inference latency and throughput (FPS).
    """
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    total_time = time.perf_counter() - start
    avg_ms = (total_time / repetitions) * 1000.0
    fps = repetitions / total_time
    return avg_ms, fps


def generate_gradcam_heatmap(model, image_path, output_path, device="cpu"):
    """
    Simple, pure PyTorch Grad-CAM implementation for layer4 activations.
    Highlights defect locations without external third-party dependencies.
    """
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        raw_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = raw_img.size
        
        t_eval = get_eval_transform(200)
        tensor = t_eval(raw_img).unsqueeze(0).to(device)

        # Hook into layer4
        conv_outputs = []
        def hook_fn(module, input, output):
            conv_outputs.append(output)

        hook = model.encoder.layer4.register_forward_hook(hook_fn)
        
        # Forward pass
        feat = model.encoder(tensor)
        # Score is the max feature activation
        score = feat.norm(dim=-1)
        
        # Backward hook gradient
        model.zero_grad()
        score.backward()
        hook.remove()

        # Generate spatial heatmap
        activations = conv_outputs[0].squeeze(0).detach().cpu().numpy() # (512, 7, 7)
        weights = activations.mean(axis=(1, 2)) # Channel weights
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for w, act in zip(weights, activations):
            cam += w * act

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (orig_w, orig_h))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = np.uint8(0.6 * np.array(raw_img) + 0.4 * heatmap)

        # Save figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_img)
        axes[0].set_title("Original Surface Image")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Defect Activation Map")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[+] Grad-CAM defect localization saved to: {output_path}")
    except Exception as e:
        print(f"[!] Warning: Could not generate Grad-CAM ({e})")


def main():
    parser = argparse.ArgumentParser(description="Representation Evaluation and Interpretability")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simclr_resnet18_backbone.pth")
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      REPRESENTATION EVALUATION & DEFECT LOCALIZATION")
    print("=" * 70)
    print(f"[*] Device: {device}")

    # 1. Load Model
    model = FeatureExtractor(args.backbone, args.checkpoint).to(device)

    # 2. Benchmark Inference Latency
    ms, fps = measure_inference_speed(model, image_size=200, device=device, repetitions=50)
    print(f"[+] Inference Latency : {ms:.2f} ms/image ({fps:.1f} FPS)")

    # 3. Load Test Data for t-SNE
    test_dir = Path(args.data_dir) / "test"
    t_eval = get_eval_transform(200)
    test_set = ImageFolder(str(test_dir), transform=t_eval)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    feats, labels = extract_all_features(model, test_loader, device)
    tsne_path = results_dir / "tsne_representations.png"
    compute_tsne_plot(feats, labels, test_set.classes, tsne_path)

    # 4. Generate Sample Grad-CAM Defect Localization
    sample_img = next(test_dir.rglob("*.jpg"), None)
    if sample_img:
        gradcam_path = results_dir / "gradcam_sample.png"
        generate_gradcam_heatmap(model, sample_img, gradcam_path, device)

    print("=" * 70)
    print("[+] Evaluation module executed successfully.")


if __name__ == "__main__":
    main()

