"""
Copy CUDA DLLs to Python directory for TensorFlow GPU support.
Run this script AS ADMINISTRATOR.
"""

import os
import shutil
import sys

def is_admin():
    """Check if running as administrator"""
    try:
        return os.getuid() == 0
    except AttributeError:
        import ctypes
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False

def copy_dlls():
    """Copy CUDA DLLs to Python directory"""
    
    # Paths
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    python_dir = r"C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39"
    python_scripts = os.path.join(python_dir, "Scripts")
    
    print("=" * 80)
    print("CUDA DLL Copy Utility")
    print("=" * 80)
    
    # Check if running as admin
    if not is_admin():
        print("\n⚠️  WARNING: Not running as Administrator!")
        print("This script may fail to copy DLLs without admin privileges.")
        print("\nTo run as Administrator:")
        print("  1. Right-click on PowerShell")
        print("  2. Select 'Run as Administrator'")
        print("  3. Run: python copy_cuda_dlls.py")
        print("\nContinuing anyway...\n")
    
    # Check CUDA directory
    if not os.path.exists(cuda_bin):
        print(f"\n❌ ERROR: CUDA bin directory not found: {cuda_bin}")
        print("Please ensure CUDA Toolkit 12.8 is installed.")
        return False
    
    # Check Python directory
    if not os.path.exists(python_dir):
        print(f"\n❌ ERROR: Python directory not found: {python_dir}")
        return False
    
    # Find DLLs
    dlls_to_copy = []
    for filename in os.listdir(cuda_bin):
        if filename.endswith('.dll'):
            dlls_to_copy.append(filename)
    
    print(f"\nFound {len(dlls_to_copy)} DLLs in CUDA bin directory")
    
    # Copy to Python directory
    copied = 0
    failed = 0
    
    print(f"\nCopying DLLs to {python_dir}...")
    for dll in dlls_to_copy:
        src = os.path.join(cuda_bin, dll)
        dst = os.path.join(python_dir, dll)
        try:
            shutil.copy2(src, dst)
            copied += 1
            print(f"  ✓ {dll}")
        except PermissionError:
            print(f"  ⚠ {dll} - Permission denied (may already exist)")
            failed += 1
        except Exception as e:
            print(f"  ✗ {dll} - {e}")
            failed += 1
    
    # Copy to Scripts directory too
    print(f"\nCopying DLLs to {python_scripts}...")
    if not os.path.exists(python_scripts):
        os.makedirs(python_scripts)
    
    for dll in dlls_to_copy:
        src = os.path.join(cuda_bin, dll)
        dst = os.path.join(python_scripts, dll)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            failed += 1
    
    print(f"\n{'=' * 80}")
    print(f"COPY COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Copied: {copied} DLLs")
    print(f"  Failed: {failed} DLLs")
    
    if copied > 0:
        print(f"\n✓ DLLs copied successfully!")
        print(f"\nNext steps:")
        print(f"  1. Close all terminal windows")
        print(f"  2. Open a NEW terminal")
        print(f"  3. Run: python test_gpu_comprehensive.py")
        print(f"  4. If GPU is detected, run: python train.py")
    else:
        print(f"\n⚠ No DLLs were copied. Check permissions.")
    
    return copied > 0

if __name__ == "__main__":
    copy_dlls()
