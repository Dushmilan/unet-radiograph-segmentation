# GPU Setup Guide for Windows

## ⚠️ IMPORTANT: GPU-ONLY Training

This project requires a working GPU setup. The training script (`train.py`) will **exit immediately** if no GPU is detected.

---

## 🔧 Current Status (Your System)

```
✓ CUDA Toolkit: 12.8 installed
✓ CUDA_PATH: Set correctly
✓ PATH: Contains CUDA paths
✓ NVIDIA Driver: 595.97 (CUDA 13.2 compatible)
✓ GPU: NVIDIA GeForce RTX 4050 detected by nvidia-smi
✗ TensorFlow: Cannot load CUDA DLLs
```

**The Issue:** TensorFlow cannot load the CUDA DLLs at runtime, even though they exist on disk.

---

## ✅ Solution: Copy DLLs to Python Directory

The CUDA DLLs exist in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` but Python/TensorFlow cannot find them. 

### Step 1: Copy CUDA DLLs to Python Directory

Run this in **Administrator PowerShell**:

```powershell
# Copy CUDA DLLs to Python directory
$pythonDir = "C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39"
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

# Copy all CUDA DLLs
Copy-Item "$cudaBin\*.dll" "$pythonDir\" -Force

# Also copy to Scripts directory
Copy-Item "$cudaBin\*.dll" "$pythonDir\Scripts\" -Force

Write-Host "DLLs copied successfully!"
```

### Step 2: Verify DLL Loading

Run the comprehensive test:

```powershell
cd C:\Users\Dushmilan\Desktop\Project-Unet
python test_gpu_comprehensive.py
```

Look for:
- **SECTION 4: CUDA DLL LOADING TEST** - All should show ✓
- **SECTION 6: TENSORFLOW GPU DETECTION** - Should show GPUs detected

### Step 3: Test Training Script

```powershell
python train.py
```

If GPU is detected, training will start. If not, you'll see an error with troubleshooting steps.

---

## 🔍 Alternative: Add CUDA to DLL Search Path

If copying DLLs doesn't work, try adding CUDA to the DLL search path:

### Create a Startup Script

Create `set_cuda_path.py`:

```python
import os
import ctypes

# Add CUDA bin directory to DLL search path
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
os.add_dll_directory(cuda_bin)
os.environ['PATH'] = cuda_bin + ';' + os.environ['PATH']

# Now import TensorFlow
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

Run it:
```powershell
python set_cuda_path.py
```

---

## 📋 Full Administrator PowerShell Fix

Run ALL of these commands in an **Administrator PowerShell** window:

```powershell
# 1. Stop any Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# 2. Copy CUDA DLLs to Python directory
$pythonDir = "C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39"
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

if (Test-Path $cudaBin) {
    Copy-Item "$cudaBin\*.dll" "$pythonDir\" -Force
    Copy-Item "$cudaBin\*.dll" "$pythonDir\Scripts\" -Force
    Write-Host "✓ DLLs copied" -ForegroundColor Green
} else {
    Write-Host "✗ CUDA bin directory not found" -ForegroundColor Red
}

# 3. Verify PATH
$path = [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($path -like "*CUDA*") {
    Write-Host "✓ CUDA in system PATH" -ForegroundColor Green
} else {
    Write-Host "⚠ CUDA not in system PATH" -ForegroundColor Yellow
}

# 4. Test GPU
Write-Host "`nTesting GPU..." -ForegroundColor Cyan
& python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## � Troubleshooting

### "Could not find module cudart64_12.dll"

**Fix:** Copy the DLL manually:
```powershell
Copy-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll" "C:\Users\Dushmilan\AppData\Local\Programs\Python\Python39\"
```

### "No GPU detected" after copying DLLs

1. **Restart your computer** - DLL cache may need refresh
2. **Reinstall TensorFlow:**
   ```powershell
   pip uninstall tensorflow tensorflow-intel -y
   pip install tensorflow==2.13.1
   ```
3. **Run comprehensive test again:**
   ```powershell
   python test_gpu_comprehensive.py
   ```

### TensorFlow 2.13.1 still doesn't work

Try TensorFlow 2.11.0 which has better Windows GPU support:

```powershell
pip uninstall tensorflow tensorflow-intel -y
pip install tensorflow==2.11.0
```

Then update `requirements.txt`:
```
tensorflow==2.11.0
numpy>=1.22,<1.24
pandas>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.4.0
```

---

## 📊 Expected Results

When GPU is working correctly, `test_gpu_comprehensive.py` should show:

```
SECTION 4: CUDA DLL LOADING TEST
CUDA Runtime:
  ✓ cudart64_12.dll - loaded successfully

SECTION 6: TENSORFLOW GPU DETECTION
✓ TensorFlow version: 2.13.1
  ✓ is_cuda_build: True
GPUs detected: 1
  GPU 0: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
  ✓ Memory growth enabled for GPU 0
```

And `train.py` will start training:

```
================================================================================
GPU REQUIREMENT CHECK
================================================================================

✓ GPU(s) detected: 1
  GPU 0: NVIDIA GeForce RTX 4050

✓ Memory growth enabled for all GPUs

================================================================================
GPU CHECK PASSED - Starting training...
================================================================================
```

---

## 📞 Still Having Issues?

If GPU still isn't detected after following all steps:

1. Run the comprehensive diagnostic and save output:
   ```powershell
   python test_gpu_comprehensive.py > gpu_diagnostic.txt
   ```

2. Check the log for specific DLL loading errors

3. Try an older TensorFlow version (2.11.0)

4. Consider using WSL2 (Windows Subsystem for Linux) for better CUDA support
