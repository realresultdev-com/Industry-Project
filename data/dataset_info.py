"""
Dataset Information, Download Links, and Metadata Reference
============================================================
Details regarding datasets suitable for:
"Self-Supervised Deep Representation Learning for Autonomous Detection of Structural Defects in Industrial Systems"
"""

DATASETS = {
    "NEU-DET": {
        "name": "NEU Surface Defect Database (Northeastern University)",
        "recommended": True,
        "description": (
            "Six typical kinds of surface defects of hot-rolled steel strip: "
            "Crazing (Cr), Inclusion (In), Patches (Pa), Pitted Surface (PS), "
            "Rolled-in Scale (RS), and Scratches (Sc). Each class has 300 grayscale images."
        ),
        "total_images": 1800,
        "classes": [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled_in_scale",
            "scratches"
        ],
        "image_size": [200, 200],
        "kaggle_slug": "fantacher/neu-metal-surface-defects-data",
        "kaggle_direct_url": "https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data",
        "official_url": "http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html",
        "download_instructions": """
Method 1 (Kaggle CLI / kagglehub):
    pip install kagglehub
    import kagglehub
    path = kagglehub.dataset_download("fantacher/neu-metal-surface-defects-data")
    print("Downloaded to:", path)

Method 2 (Manual):
    Download zip from https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data
    Extract the folders into: data/raw/NEU-DET/
"""
    },
    "MVTec-AD": {
        "name": "MVTec Anomaly Detection Dataset",
        "recommended": False,
        "description": (
            "Comprehensive benchmark for unsupervised anomaly detection in industrial inspection. "
            "Contains 15 industrial categories (textures and objects) with normal training images "
            "and anomalous test images."
        ),
        "official_url": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "download_instructions": """
Download directly from https://www.mvtec.com/company/research/datasets/mvtec-ad
Select an industrial category (e.g. 'metal_nut', 'screw', 'grid', or 'tile').
"""
    },
    "Concrete-Crack": {
        "name": "Concrete Crack Images for Classification",
        "recommended": False,
        "description": (
            "40,000 images (227x227) divided into Positive (cracked) and Negative (non-cracked) "
            "subdirectories. Ideal for civil/structural defect inspection."
        ),
        "kaggle_direct_url": "https://www.kaggle.com/datasets/arunrk7/surface-crack-detection",
        "download_instructions": """
Download from Kaggle: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
Extract into data/raw/Concrete-Crack/
"""
    }
}

def print_dataset_guide():
    print("=" * 80)
    print("RECOMMENDED BENCHMARK DATASET: NEU Surface Defect Database")
    print("=" * 80)
    neu = DATASETS["NEU-DET"]
    print(f"Name         : {neu['name']}")
    print(f"Total Samples: {neu['total_images']} ({len(neu['classes'])} classes, 300 per class)")
    print(f"Classes      : {', '.join(neu['classes'])}")
    print(f"Why NEU-DET? :")
    print("  1. Lightweight (~35 MB), fast training on both CPU and Colab T4 GPU.")
    print("  2. Rich visual intra-class variations and subtle defect patterns.")
    print("  3. Real-world industrial steel manufacturing benchmark.")
    print("  4. Ideal for demonstrating SSL sample efficiency (1%, 5%, 10%, 25%, 100%).")
    print("-" * 80)
    print("DOWNLOAD INSTRUCTIONS:")
    print(neu['download_instructions'])
    print("=" * 80)

if __name__ == "__main__":
    print_dataset_guide()

