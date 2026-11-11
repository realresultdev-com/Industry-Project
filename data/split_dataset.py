"""
Dataset Pre-processing and Stratified Split Script
=================================================
Prepares:
1. Unlabeled pre-training pool (for Self-Supervised Learning)
2. Stratified fine-tuning splits: 1%, 5%, 10%, 25%, 100%
3. Independent validation and test splits

Features automatic synthetic sample generation if raw dataset is not yet downloaded.
"""

import os
import sys
import shutil
import random
import json
import argparse
from pathlib import Path
from collections import defaultdict

def create_synthetic_neu_data(output_dir: Path, num_per_class: int = 50):
    """
    Generates synthetic sample industrial images if the user hasn't downloaded the raw dataset yet.
    Useful for testing pipeline logic and debugging offline with zero heavy external dependencies.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        print("[!] Pillow (PIL) is required to generate synthetic data.")
        return

    print(f"[*] Generating synthetic industrial defect dataset at: {output_dir}")
    classes = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled_in_scale", "scratches"
    ]
    
    for cls in classes:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_per_class):
            # Base steel surface texture (pure PIL implementation)
            img = Image.new("RGB", (200, 200), color=(128, 128, 128))
            draw = ImageDraw.Draw(img)
            
            # Subtle steel background noise
            for _ in range(300):
                nx, ny = random.randint(0, 199), random.randint(0, 199)
                shade = random.randint(110, 145)
                draw.point((nx, ny), fill=(shade, shade, shade))
            
            # Draw synthetic defect characteristics
            if cls == "scratches":
                for _ in range(random.randint(2, 5)):
                    x1, y1 = random.randint(10, 190), random.randint(10, 190)
                    draw.line([(x1, y1), (x1 + random.randint(-40, 40), y1 + random.randint(-40, 40))], fill=(30, 30, 30), width=2)
            elif cls == "patches":
                for _ in range(random.randint(1, 3)):
                    x, y = random.randint(20, 150), random.randint(20, 150)
                    draw.ellipse([x, y, x + random.randint(20, 40), y + random.randint(20, 40)], fill=(70, 70, 70))
            elif cls == "crazing":
                for _ in range(random.randint(6, 12)):
                    x, y = random.randint(10, 180), random.randint(10, 180)
                    draw.line([(x, y), (x + random.randint(-15, 15), y + random.randint(-15, 15))], fill=(40, 40, 40), width=1)
            elif cls == "pitted_surface":
                for _ in range(random.randint(20, 45)):
                    x, y = random.randint(10, 190), random.randint(10, 190)
                    draw.point((x, y), fill=(20, 20, 20))
            elif cls == "inclusion":
                x, y = random.randint(30, 140), random.randint(30, 140)
                draw.rectangle([x, y, x + random.randint(15, 35), y + random.randint(15, 35)], fill=(210, 210, 210))
            elif cls == "rolled_in_scale":
                for _ in range(random.randint(2, 4)):
                    x, y = random.randint(10, 160), random.randint(10, 160)
                    draw.line([(x, y), (x + random.randint(30, 60), y + random.randint(-10, 10))], fill=(45, 45, 45), width=3)
            
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            img.save(cls_dir / f"{cls}_{i+1:03d}.jpg")
            
    print(f"[+] Synthetic dataset created: {len(classes)} classes x {num_per_class} images = {len(classes) * num_per_class} images.")


def prepare_splits(
    raw_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Scans raw_dir for class folders, partitions into train/val/test,
    creates subset splits (1%, 5%, 10%, 25%, 100%), and prepares
    the unlabeled SSL pre-training pool.
    """
    random.seed(seed)
    
    # 1. Discover classes and images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    class_images = defaultdict(list)
    
    for entry in raw_dir.iterdir():
        if entry.is_dir():
            images = [p for p in entry.glob("*") if p.suffix.lower() in valid_exts]
            if images:
                class_images[entry.name] = sorted(images)
                
    if not class_images:
        print(f"[!] No class directories with valid images found in {raw_dir}")
        print("    Run with --mock to generate a synthetic test dataset first.")
        return

    classes = sorted(list(class_images.keys()))
    print(f"[+] Detected {len(classes)} classes: {classes}")
    for cls in classes:
        print(f"    - {cls:<20}: {len(class_images[cls])} images")
        
    # 2. Partition into train_full, val, test
    train_full_images = defaultdict(list)
    val_images = defaultdict(list)
    test_images = defaultdict(list)
    
    for cls, paths in class_images.items():
        shuffled = paths.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        
        test_images[cls] = shuffled[:n_test]
        val_images[cls] = shuffled[n_test : n_test + n_val]
        train_full_images[cls] = shuffled[n_test + n_val :]

    # 3. Create sub-fractions for label efficiency experiments
    # 1%, 5%, 10%, 25%, 100%
    ratios = {
        "train_100pct": 1.0,
        "train_25pct": 0.25,
        "train_10pct": 0.10,
        "train_5pct": 0.05,
        "train_1pct": 0.01
    }
    
    split_subsets = {}
    for split_name, ratio in ratios.items():
        split_subsets[split_name] = defaultdict(list)
        for cls, train_list in train_full_images.items():
            k = max(1, int(round(len(train_list) * ratio)))
            split_subsets[split_name][cls] = train_list[:k]

    # 4. Copy / Link files into target structure
    print("-" * 70)
    print(f"[*] Exporting partitioned splits to: {output_dir}")
    
    # Structure:
    # output_dir/
    #   ├── unlabeled_pretrain/  (all train images pooled, label folders removed for pure SSL)
    #   ├── val/
    #   ├── test/
    #   ├── train_100pct/
    #   ├── train_25pct/
    #   ├── train_10pct/
    #   ├── train_5pct/
    #   └── train_1pct/
    
    # Clear old destination if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper copy function
    def copy_class_dict(cls_dict, dest_root):
        for cls, items in cls_dict.items():
            cls_folder = dest_root / cls
            cls_folder.mkdir(parents=True, exist_ok=True)
            for src in items:
                dest = cls_folder / src.name
                shutil.copy2(src, dest)
                
    # Copy Validation and Test sets
    copy_class_dict(val_images, output_dir / "val")
    copy_class_dict(test_images, output_dir / "test")
    
    # Copy Labeled Subsets
    for split_name, cls_dict in split_subsets.items():
        copy_class_dict(cls_dict, output_dir / split_name)
        
    # Create Unlabeled SSL Pre-training Pool (Train + Val or Train full pooled)
    unlabeled_dir = output_dir / "unlabeled_pretrain" / "images"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    
    unlabeled_count = 0
    for cls, items in train_full_images.items():
        for src in items:
            # Flatten into unlabeled pool
            dest = unlabeled_dir / f"{cls}_{src.name}"
            shutil.copy2(src, dest)
            unlabeled_count += 1
            
    # 5. Build and Save Manifest
    manifest = {
        "classes": classes,
        "num_classes": len(classes),
        "splits": {
            "unlabeled_pretrain": unlabeled_count,
            "val": sum(len(v) for v in val_images.values()),
            "test": sum(len(v) for v in test_images.values()),
        }
    }
    for s_name, s_dict in split_subsets.items():
        manifest["splits"][s_name] = {cls: len(items) for cls, items in s_dict.items()}
        manifest["splits"][f"{s_name}_total"] = sum(len(v) for v in s_dict.values())
        
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    # 6. Display Statistics Summary Table
    print("=" * 70)
    print("SPLIT SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Split Name':<20} | {'Total Images':<15} | {'Samples Per Class'}")
    print("-" * 70)
    print(f"{'unlabeled_pretrain':<20} | {unlabeled_count:<15} | Pooled without labels")
    for s_name in ratios.keys():
        tot = manifest["splits"][f"{s_name}_total"]
        per_cls = manifest["splits"][s_name][classes[0]]
        print(f"{s_name:<20} | {tot:<15} | ~{per_cls} per class")
    print(f"{'val':<20} | {manifest['splits']['val']:<15} | ~{len(val_images[classes[0]])} per class")
    print(f"{'test':<20} | {manifest['splits']['test']:<15} | ~{len(test_images[classes[0]])} per class")
    print("=" * 70)
    print(f"[+] Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Industrial Defect Dataset for SSL.")
    parser.add_argument("--raw_dir", type=str, default="data/raw/NEU-DET", help="Path to raw dataset folder")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Destination path for splits")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic sample data if raw data is missing")
    
    args = parser.parse_args()
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    
    if args.mock or not raw_path.exists():
        if not raw_path.exists():
            print(f"[*] Raw data path {raw_path} not found.")
            print("[*] Generating synthetic sample images for pipeline testing...")
        create_synthetic_neu_data(raw_path, num_per_class=50)
        
    prepare_splits(raw_path, out_path, args.val_ratio, args.test_ratio, args.seed)
