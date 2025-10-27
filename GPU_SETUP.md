# GPU Setup for TensorFlow

This guide will help you set up TensorFlow with GPU support for faster training.

## Prerequisites

### Check if you have an NVIDIA GPU
```bash
nvidia-smi
```
If this command works and shows your GPU, you're ready to proceed.

## Installation Steps

### Option 1: Using Anaconda (Recommended)

If you have Anaconda/Miniconda installed:

```bash
# Create a new environment
conda create -n vbibridge python=3.10

# Activate the environment
conda activate vbibridge

# Install CUDA-enabled TensorFlow (conda handles CUDA/cuDNN automatically)
conda install tensorflow-gpu -c conda-forge

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Using pip (Windows)

For **TensorFlow 2.13+**, GPU support is included automatically:

```bash
# Update pip first
pip install --upgrade pip

# Install TensorFlow (GPU support included since 2.13+)
pip install tensorflow>=2.13.0

# Verify GPU is detected
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Option 2b: Manual CUDA Setup (Older TensorFlow versions)

If you need specific CUDA/cuDNN versions:

1. **Install CUDA Toolkit** (v11.8 recommended for TF 2.13+)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Install with default settings

2. **Install cuDNN** (v8.6 or later)
   - Download from: https://developer.nvidia.com/cudnn
   - Extract and copy files to CUDA installation directory

3. **Install TensorFlow**
   ```bash
   pip install tensorflow>=2.13.0
   ```

## Verify GPU Installation

Create and run this test script:

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\nGPU Devices:")
print(tf.config.list_physical_devices('GPU'))

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("\n✓ GPU is available and ready to use!")
    # Test GPU with a simple operation
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0]])
        b = tf.constant([[3.0], [4.0]])
        c = tf.matmul(a, b)
    print("✓ GPU computation test successful!")
else:
    print("\n⚠ GPU not detected. Will use CPU.")
```

Save this as `check_gpu.py` and run:
```bash
python check_gpu.py
```

## GPU Requirements

### Minimum Requirements:
- **NVIDIA GPU** with Compute Capability 3.5 or higher
- **4GB+ VRAM** (recommended: 8GB+)
- **CUDA Toolkit 11.8+** (for TF 2.13+)
- **cuDNN 8.6+**

### Recommended GPUs:
- GTX 1060 (6GB VRAM) or better
- RTX series (2060, 2070, 2080, 3060, 3070, 3080, etc.)
- Tesla/Quadro series

## Troubleshooting

### Issue: "Could not load dynamic library 'cudart64_*.dll'"
**Solution:** Install CUDA Toolkit. The DLL is part of CUDA.

### Issue: "Failed to get convolution algorithm"
**Solution:** This usually means not enough GPU memory. Try:
```python
# Add at the start of your training script
import tensorflow as tf
tf.config.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
```

### Issue: TensorFlow doesn't detect GPU
**Solution:**
1. Verify CUDA is installed: `nvcc --version`
2. Verify cuDNN is installed
3. Restart your terminal/IDE
4. Check GPU: `nvidia-smi`

### Issue: Out of memory errors
**Solution:** Reduce batch size in `train_cnn.py`:
- Change `batch_size=32` to `batch_size=16` or `8`
- Or reduce model size (fewer filters in Conv1D layers)

## Setting GPU Memory Growth

Add this to the top of your training script to prevent TensorFlow from allocating all GPU memory:

```python
import tensorflow as tf

# Configure GPU to use memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(e)
```

## Expected Performance Improvement

With GPU acceleration:
- **10-50x faster** training compared to CPU
- Example: 50 epochs ~20-30 minutes (GPU) vs 10-15 hours (CPU)
- Real-time depends on your GPU model

## Check GPU Usage During Training

While training, open another terminal and run:
```bash
watch -n 1 nvidia-smi
```

This shows real-time GPU utilization, memory usage, and temperature.

