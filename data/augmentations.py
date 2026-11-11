"""
Industrial Image Augmentations & Dataset Loaders for Self-Supervised Learning
=============================================================================
Provides domain-specific data augmentations for industrial defect inspection:
- Steel surfaces and industrial components have no single canonical orientation,
  making random 90-degree rotations, horizontal flips, and vertical flips highly valid.
- Steel is largely monochromatic/grayscale, so color jitter emphasizes brightness/contrast
  variations rather than unrealistic hue shifts.
- Gaussian blur and noise simulate sensor defocusing and lighting fluctuations on conveyor belts.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Standard ImageNet normalization parameters
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class TwoCropsTransform:
    """
    Applies two independent stochastic augmentations to the same input image.
    Produces the positive pair (x_i, x_j) required for contrastive/non-contrastive SSL.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_industrial_ssl_transform(image_size: int = 200) -> T.Compose:
    """
    Returns data augmentation pipeline specifically tailored for industrial defect surfaces.
    """
    return T.Compose([
        # 1. Random Resized Crop: captures defects at various scales and zooms
        T.RandomResizedCrop(
            size=image_size,
            scale=(0.4, 1.0),
            ratio=(0.75, 1.33)
        ),
        # 2. Geometric invariances: horizontal & vertical flips
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # 3. 90/180/270 degree rotation invariance (common in rolled steel)
        T.RandomRotation(degrees=(0, 360)),
        # 4. Photometric variations: brightness & contrast adjustments
        T.RandomApply([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.0)
        ], p=0.8),
        # 5. Sensor noise & defocus: Gaussian blur
        T.RandomApply([
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
        ], p=0.5),
        # 6. Grayscale conversion to handle monochromatic inspection
        T.RandomGrayscale(p=0.2),
        # 7. Convert to tensor & normalize
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


def get_eval_transform(image_size: int = 200) -> T.Compose:
    """
    Deterministic preprocessing for downstream evaluation (validation, testing, linear probing).
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


class UnlabeledIndustrialDataset(Dataset):
    """
    Loads raw industrial images from a single directory (without subdirectories or labels)
    for self-supervised pre-training.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        
        # Scan recursively if needed
        self.image_paths = sorted([
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in valid_exts
        ])
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image files found in directory: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, str(path)


if __name__ == "__main__":
    print("[+] Industrial data augmentations module loaded successfully.")
    t_ssl = get_industrial_ssl_transform(200)
    print("SSL Augmentation Pipeline:\n", t_ssl)

