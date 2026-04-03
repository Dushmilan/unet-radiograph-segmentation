"""
Configure CUDA DLL paths for TensorFlow GPU support on Windows.
This script must be imported before TensorFlow.
"""
import os
import sys

def configure_cuda_paths():
    """Add NVIDIA pip package DLL directories to the search path"""
    
    # Get Python site-packages directory
    site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
    
    # NVIDIA pip package DLL directories (CUDA 12)
    nvidia_dirs = [
        os.path.join(site_packages, 'nvidia', 'cudnn', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cublas', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cuda_runtime', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cuda_nvcc', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cuda_nvrtc', 'bin'),
        os.path.join(site_packages, 'nvidia', 'nccl', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cufft', 'bin'),
        os.path.join(site_packages, 'nvidia', 'curand', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cusolver', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cusparse', 'bin'),
    ]
    
    # Also add system CUDA installation
    system_cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    if os.path.exists(system_cuda):
        nvidia_dirs.append(system_cuda)
    
    # FIRST: Add to PATH before any DLL loading
    path_parts = []
    for path in nvidia_dirs:
        if os.path.exists(path) and path not in os.environ.get('PATH', ''):
            path_parts.append(path)
    
    if path_parts:
        os.environ['PATH'] = ';'.join(path_parts) + ';' + os.environ.get('PATH', '')
    
    # Set TensorFlow environment variables BEFORE import
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    os.environ['CUDA_HOME'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    
    # Add to DLL search path (Python 3.8+) - AFTER setting PATH
    for path in nvidia_dirs:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass  # Directory already added or not supported

# Configure paths immediately when this module is imported
configure_cuda_paths()
