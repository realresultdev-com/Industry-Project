"""
Environment Diagnostic & Hardware Verification Script
=====================================================
Checks Python version, PyTorch installation, CUDA availability,
GPU memory, and core library readiness for the SSL project.
"""

import sys
import os
import platform

def check_env():
    print("=" * 70)
    print("      PROJECT ENVIRONMENT & HARDWARE VERIFICATION")
    print("=" * 70)
    
    # 1. OS & Python Check
    print(f"[*] Operating System : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"[*] Python Version   : {platform.python_version()} ({sys.executable})")
    
    # Check Python version >= 3.8
    if sys.version_info < (3, 8):
        print("[!] Warning: Python 3.8 or higher is strongly recommended.")
    else:
        print("[+] Python version is compatible.")

    # 2. PyTorch Check
    print("-" * 70)
    try:
        import torch
        print(f"[+] PyTorch Version  : {torch.__version__}")
        
        # Check CUDA
        cuda_avail = torch.cuda.is_available()
        print(f"[*] CUDA Available   : {cuda_avail}")
        
        if cuda_avail:
            device_count = torch.cuda.device_count()
            print(f"[+] Found {device_count} CUDA Device(s):")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"    - Device [{i}]: {device_name} ({mem_gb:.2f} GB VRAM)")
            device = "cuda:0"
        else:
            print("[-] No CUDA GPU detected by PyTorch.")
            print("    -> Local execution will use CPU.")
            print("    -> For GPU acceleration, either run on Google Colab / Kaggle, or install")
            print("       PyTorch with CUDA support: https://pytorch.org/get-started/locally/")
            device = "cpu"
            
        # Quick tensor compute test
        x = torch.randn(10, 10, device=device)
        y = torch.matmul(x, x)
        print(f"[+] Device test successful on: {device}")
        
    except ImportError:
        print("[!] PyTorch is NOT installed in this Python environment.")
        print("    Install via: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("    Or for CPU:  pip install torch torchvision")

    # 3. Check Other Required Libraries
    print("-" * 70)
    print("[*] Checking essential libraries:")
    libs = [
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("Pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("tqdm", "tqdm"),
        ("streamlit", "streamlit"),
    ]
    
    missing = []
    for pkg_name, import_name in libs:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [+] {pkg_name:<16} : {ver}")
        except ImportError:
            print(f"  [-] {pkg_name:<16} : MISSING")
            missing.append(pkg_name)
            
    print("-" * 70)
    if missing:
        print(f"[!] {len(missing)} package(s) missing. Run:")
        print(f"    pip install {' '.join(missing)}")
    else:
        print("[+] All core libraries are installed and ready!")
    print("=" * 70)

if __name__ == "__main__":
    check_env()

