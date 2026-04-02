"""
Test GPU detection and show detailed information
"""
import os
import sys

# Add NVIDIA DLL paths before importing TensorFlow
nvidia_paths = [
    r"C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39\lib\site-packages\nvidia\cudnn\bin",
    r"C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39\lib\site-packages\nvidia\cublas\bin",
    r"C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39\lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39\lib\site-packages\nvidia\cuda_nvcc\bin",
    r"C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39\lib\site-packages\nvidia\cuda_nvrtc\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"
]

# Add to PATH
current_path = os.environ.get('PATH', '')
for path in nvidia_paths:
    if os.path.exists(path) and path not in current_path:
        os.environ['PATH'] = path + ';' + os.environ.get('PATH', '')

# Set CUDA environment variables
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
os.environ['CUDA_HOME'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU Detection Test")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Python Version: {sys.version.split()[0]}")
print()

# Check physical devices
print("Physical Devices:")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(f"  {device}")
print()

# Check logical devices
print("Logical Devices:")
logical_devices = tf.config.list_logical_devices()
for device in logical_devices:
    print(f"  {device}")
print()

# Check GPUs specifically
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")
print()

if gpus:
    print("✅ SUCCESS! GPU is available!")
    
    # Test GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memory growth enabled")
    except Exception as e:
        print(f"⚠️ Memory growth error: {e}")
else:
    print("❌ No GPU detected")
    print()
    print("Troubleshooting steps:")
    print("1. Run the Administrator PowerShell commands from GPU_SETUP.md")
    print("2. Restart your computer")
    print("3. Run this test again")
    print()
    print("PATH contains NVIDIA:")
    print("  ", "Yes" if any('nvidia' in p.lower() for p in os.environ.get('PATH', '').split(';')) else "No")

print("=" * 60)
