# Training Instructions

## Current Status

✅ **All code and data files are ready on GitHub**
- 20 clipped FFT data files (Results/)
- Data loader script
- Training script with GPU detection
- Requirements file

## To Train the Model

### Option 1: Quick Start (Will use CPU)

```bash
# Navigate to project directory
cd C:\Users\Rakeh-PC\Downloads\VBI_BridgeProject

# Run training
python train_cnn.py
```

This will automatically:
- Detect if GPU is available
- Fall back to CPU if GPU not found
- Train the model
- Save results in `training_results/`

### Option 2: Enable GPU Support (Faster Training)

For your **GTX 1070** with CUDA 12.9, you can enable GPU by:

**Simple Method:**
```bash
# Install TensorFlow with CUDA support
pip install tensorflow-gpu==2.13.0

# Or downgrade to compatible version
pip install tensorflow==2.13.0 tensorflow-gpu==2.13.0
```

**Alternative: Use Conda (Easier)**
```bash
# If you have Anaconda/Miniconda
conda install tensorflow-gpu -c conda-forge
```

Then verify GPU is detected:
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Training Times (Expected)

- **CPU only**: ~10-15 hours for 50 epochs
- **GPU (GTX 1070)**: ~20-30 minutes for 50 epochs

## What to Expect

The training script will:

1. **Load data** (20 files, ~4000 samples in train + ~4000 in test)
2. **Build CNN model** (architecture with 3 conv blocks)
3. **Train** with early stopping (up to 50 epochs)
4. **Save** best model to `training_results/best_cnn_model.h5`
5. **Generate** plots (training history, confusion matrix)
6. **Print** evaluation metrics

## Output Files

After training completes:
```
training_results/
├── best_cnn_model.h5          # Best model checkpoint
├── cnn_model_TIMESTAMP.h5     # Final saved model  
├── CNN_TIMESTAMP_history.png  # Training curves
└── CNN_TIMESTAMP_confusion.png # Confusion matrix
```

## Troubleshooting

### "Out of memory" error
- Reduce batch size in `train_cnn.py`: change `batch_size=32` to `16` or `8`

### GPU not detected
- Your current setup will use CPU automatically
- For faster training, see "Option 2" above

### Import errors
```bash
pip install -r requirements.txt
```

### Want to resume training?
The script saves the best model automatically. You can:
1. Load the saved model
2. Continue training from that checkpoint

## Next Steps After Training

1. Load and test the model
2. Try different architectures
3. Fine-tune hyperparameters
4. Analyze results and confusion matrix
5. Share trained model with collaborator

