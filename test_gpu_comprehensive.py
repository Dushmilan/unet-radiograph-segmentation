"""
Comprehensive GPU Diagnostic Test for TensorFlow on Windows
Tests all possible GPU detection methods and provides detailed troubleshooting.
"""

import os
import sys
import ctypes
import platform
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE GPU DIAGNOSTIC TEST")
print("=" * 80)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"Working Directory: {os.getcwd()}")
print()

# ============================================================================
# SECTION 1: Environment Variables
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: ENVIRONMENT VARIABLES")
print("=" * 80)

env_vars_to_check = ['CUDA_PATH', 'CUDA_PATH_V12_8', 'CUDA_PATH_V11_2', 'PATH']
for var in env_vars_to_check:
    value = os.environ.get(var, 'NOT SET')
    if var == 'PATH':
        cuda_paths = [p for p in value.split(';') if 'CUDA' in p or 'nvidia' in p.lower()]
        print(f"\n{var}:")
        if cuda_paths:
            for p in cuda_paths:
                print(f"  ✓ {p}")
        else:
            print(f"  ✗ No CUDA/NVIDIA paths found")
    else:
        status = "✓" if value != 'NOT SET' else "✗"
        print(f"{status} {var}: {value}")

# ============================================================================
# SECTION 2: CUDA Installation Check
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: CUDA INSTALLATION")
print("=" * 80)

cuda_locations = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2",
]

cuda_path_found = None
for loc in cuda_locations:
    if os.path.exists(loc):
        print(f"✓ CUDA found at: {loc}")
        cuda_path_found = loc
        
        # Check for key files
        bin_dir = os.path.join(loc, 'bin')
        if os.path.exists(bin_dir):
            print(f"  bin directory: ✓ exists")
            
            # Check for cuDNN DLLs
            cudnn_dlls = ['cudnn64_9.dll', 'cudnn64_8.dll', 'cudnn64_7.dll']
            for dll in cudnn_dlls:
                dll_path = os.path.join(bin_dir, dll)
                if os.path.exists(dll_path):
                    size_mb = os.path.getsize(dll_path) / (1024 * 1024)
                    print(f"    ✓ {dll} ({size_mb:.1f} MB)")
                else:
                    print(f"    ✗ {dll} (not found)")
        else:
            print(f"  bin directory: ✗ not found")
        break

if not cuda_path_found:
    print("✗ No CUDA installation found in standard locations")

# ============================================================================
# SECTION 3: NVIDIA Pip Packages
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: NVIDIA PIP PACKAGES")
print("=" * 80)

nvidia_packages = [
    'nvidia-cudnn-cu12',
    'nvidia-cudnn-cu11',
    'nvidia-cublas-cu12',
    'nvidia-cublas-cu11',
    'nvidia-cuda-runtime-cu12',
    'nvidia-cuda-runtime-cu11',
    'nvidia-cuda-nvcc-cu12',
    'nvidia-cuda-nvrtc-cu12',
]

import importlib.util
for pkg in nvidia_packages:
    spec = importlib.util.find_spec(pkg.replace('-', '_'))
    if spec and spec.origin:
        pkg_path = Path(spec.origin).parent
        print(f"✓ {pkg}: installed at {pkg_path}")
        
        # Check for bin directory with DLLs
        bin_path = pkg_path / 'bin'
        if bin_path.exists():
            dlls = list(bin_path.glob('*.dll'))
            if dlls:
                print(f"    Found {len(dlls)} DLLs in bin/")
    else:
        print(f"✗ {pkg}: not installed")

# ============================================================================
# SECTION 4: DLL Loading Test
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: CUDA DLL LOADING TEST")
print("=" * 80)

dlls_to_test = [
    ('CUDA Runtime', 'cudart64_12.dll', 'cudart64_110.dll', 'cudart64_101.dll'),
    ('cuDNN', 'cudnn64_9.dll', 'cudnn64_8.dll', 'cudnn64_7.dll'),
    ('cuBLAS', 'cublas64_12.dll', 'cublas64_11.dll', 'cublas64_10.dll'),
]

def try_load_dll(dll_name):
    """Try to load a DLL by name"""
    try:
        ctypes.CDLL(dll_name)
        return True, None
    except OSError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

for lib_name, *dll_names in dlls_to_test:
    print(f"\n{lib_name}:")
    loaded = False
    for dll in dll_names:
        success, error = try_load_dll(dll)
        if success:
            print(f"  ✓ {dll} - loaded successfully")
            loaded = True
            break
        else:
            print(f"  ✗ {dll} - {error[:60]}...")
    if not loaded:
        print(f"  ⚠ No version of {lib_name} could be loaded")

# ============================================================================
# SECTION 5: NVIDIA Driver Check
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: NVIDIA DRIVER")
print("=" * 80)

import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✓ nvidia-smi executed successfully")
        # Parse GPU info from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'GPU' in line or 'Name' in line or 'Driver' in line:
                print(f"  {line.strip()}")
    else:
        print("✗ nvidia-smi failed")
except FileNotFoundError:
    print("✗ nvidia-smi not found in PATH")
except subprocess.TimeoutExpired:
    print("✗ nvidia-smi timed out")
except Exception as e:
    print(f"✗ nvidia-smi error: {e}")

# ============================================================================
# SECTION 6: TensorFlow GPU Detection
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: TENSORFLOW GPU DETECTION")
print("=" * 80)

try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Get build info
    build_info = tf.sysconfig.get_build_info()
    print(f"\nTensorFlow Build Info:")
    for key, value in build_info.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # Physical devices
    print(f"\nPhysical Devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  {device}")
    
    # GPUs specifically
    print(f"\nGPU Detection:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ {len(gpus)} GPU(s) detected!")
        for i, gpu in enumerate(gpus):
            print(f"    GPU {i}: {gpu}")
            
            # Try to set memory growth
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"    ✓ Memory growth enabled for GPU {i}")
            except Exception as e:
                print(f"    ⚠ Memory growth error: {e}")
    else:
        print(f"  ✗ No GPUs detected")
        
        # Try to list logical devices
        print(f"\nLogical Devices:")
        logical_devices = tf.config.list_logical_devices()
        for device in logical_devices:
            print(f"  {device}")
    
    # Test GPU operation
    print(f"\nGPU Operation Test:")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"  ✓ Matrix multiplication on GPU successful")
            print(f"    Result: {c.numpy()}")
    except Exception as e:
        print(f"  ✗ GPU operation failed: {e}")
        
        # Fallback to CPU test
        print(f"\nCPU Fallback Test:")
        try:
            with tf.device('/CPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"  ✓ CPU operation successful")
        except Exception as e2:
            print(f"  ✗ CPU operation also failed: {e2}")

except ImportError as e:
    print(f"✗ TensorFlow not installed: {e}")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")

# ============================================================================
# SECTION 7: DirectML Test (Alternative GPU Backend)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: DIRECTML TEST (Alternative GPU Backend)")
print("=" * 80)

try:
    import tensorflow_directml_plugin as dml
    print("✓ DirectML plugin imported")
    
    # Try to list DirectML devices
    try:
        dml_devices = tf.config.list_physical_devices('DML')
        if dml_devices:
            print(f"  ✓ {len(dml_devices)} DirectML device(s) found")
            for i, dev in enumerate(dml_devices):
                print(f"    DML {i}: {dev}")
        else:
            print(f"  ✗ No DirectML devices found")
    except Exception as e:
        print(f"  ✗ DirectML device listing failed: {e}")
        
except ImportError:
    print("✗ DirectML plugin not installed")
except Exception as e:
    print(f"✗ DirectML error: {e}")

# ============================================================================
# SECTION 8: Summary and Recommendations
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

issues = []
recommendations = []

# Check TensorFlow version
try:
    import tensorflow as tf
    tf_version = tf.__version__
    major, minor = map(int, tf_version.split('.')[:2])
    
    if major == 2 and minor < 13:
        issues.append(f"TensorFlow {tf_version} may have limited GPU support")
        recommendations.append("Upgrade to TensorFlow 2.13.1 or newer")
    elif major == 2 and minor >= 16:
        issues.append(f"TensorFlow {tf_version} requires CUDA 12.x pip packages")
        recommendations.append("Ensure nvidia-cudnn-cu12 and related packages are installed")
except:
    issues.append("TensorFlow not properly installed")
    recommendations.append("Install TensorFlow: pip install tensorflow==2.13.1")

# Check for CUDA
if not cuda_path_found:
    issues.append("CUDA Toolkit not found")
    recommendations.append("Install CUDA Toolkit 12.x from NVIDIA")

# Check for cuDNN DLLs
cudnn_found = False
for loc in cuda_locations:
    bin_dir = os.path.join(loc, 'bin')
    if os.path.exists(bin_dir):
        for dll in ['cudnn64_9.dll', 'cudnn64_8.dll', 'cudnn64_7.dll']:
            if os.path.exists(os.path.join(bin_dir, dll)):
                cudnn_found = True
                break

if not cudnn_found:
    issues.append("cuDNN DLLs not found")
    recommendations.append("Copy cuDNN DLLs to CUDA bin directory or install nvidia-cudnn-cu12")

# Check GPU detection
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        issues.append("TensorFlow cannot detect GPU")
        recommendations.append("Add CUDA paths to system PATH and restart")
        recommendations.append("Run: [Environment]::SetEnvironmentVariable('PATH', ..., 'Machine') as Administrator")
except:
    pass

if issues:
    print(f"\n⚠️  ISSUES FOUND ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\n📋 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("\n✓ No issues found! GPU should be working correctly.")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
