# VBI Bridge Project

## Overview
Vehicle-Bridge Interaction (VBI) analysis project for bridge damage detection using Convolutional Neural Networks (CNN). This project classifies bridge damage conditions (DC0-DC4) from vehicle-bridge interaction FFT spectra.

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/rakehsaleem/VBI_BridgeProject.git
cd VBI_BridgeProject
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the CNN
```bash
python train_cnn.py
```

The model will be saved in `training_results/` with:
- Best model checkpoint: `best_cnn_model.h5`
- Training history plots
- Confusion matrix
- Classification report

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture and data flow.

**Summary:**
- **Data**: 4 bridges (11m, 13m, 15m, 17m) × 5 damage conditions (DC0-DC4)
- **Input Shape**: (samples, 250, 2) → 250 frequency bins × 2 sensors
- **Output**: 5 classes (DC0-DC4 damage levels)
- **Training**: 3 bridges (11m, 13m, 17m) → 15,000 samples
- **Test**: 1 bridge (15m) → 5,000 samples

## Project Structure
- **train_cnn.py**: Main CNN training script with model saving and evaluation
- **cnn_data_loader.py**: Data loader that reshapes data and splits bridges for training/testing
- **clip.py**: Script to clip FFT data to first 250 frequency bins
- **Results/**: Contains 20 pre-processed clipped data files
- **requirements.txt**: Python dependencies
- **ARCHITECTURE.md**: Detailed architecture and data flow diagrams

## Dataset Details

### Training Data (3 bridges, 15 damage condition files):
- **11m Bridge**: Sim1 - DC0, DC1, DC2, DC3, DC4
- **13m Bridge**: Sim2 - DC0, DC1, DC2, DC3, DC4  
- **17m Bridge**: Sim4 - DC0, DC1, DC2, DC3, DC4

### Test Data (1 bridge, 5 damage condition files):
- **15m Bridge**: Sim3 - DC0, DC1, DC2, DC3, DC4

### Damage Conditions:
- **DC0**: No damage (baseline)
- **DC1**: Light damage
- **DC2**: Moderate damage
- **DC3**: Severe damage
- **DC4**: Critical damage

### Data Shape:
- Input: (samples, 250, 2) - 250 frequency bins × 2 sensors
- Output: (samples,) - Damage level labels (0-4)

**Data Format:**
- Each sample represents one vehicle passing over the bridge
- Contains readings from 2 sensors (sensor 1 and sensor 2)
- FFT normalized spectra with 250 frequency bins per sensor

## CNN Architecture

The model uses a 1D CNN with:
- **3 Convolutional blocks** with BatchNormalization and MaxPooling
- **32 → 64 → 128** filters
- **Dropout layers** (0.25-0.5) for regularization
- **Dense layers** (128 → 64 → 5 classes)
- **Early stopping** based on validation loss
- **Model checkpointing** saves best model during training

## Usage

### Train the Model
```bash
python train_cnn.py
```

This will run on CPU and:
1. Load the clipped data (20 files)
2. Build the CNN model
3. Train for up to 50 epochs with early stopping
4. Save the best model
5. Generate plots and evaluation metrics

### Load and Test Data Manually
```python
from cnn_data_loader import load_data_sets

X_train, Y_train, X_test, Y_test = load_data_sets()
print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
```

### Re-create Clipped Data (if needed)
```bash
python clip.py
```

## Dependencies
- numpy>=1.21.0
- scipy>=1.7.0
- tensorflow>=2.10.0
- matplotlib>=3.5.0
- scikit-learn>=1.0.0

## Output Files

After training, you'll find in `training_results/`:
- `best_cnn_model.h5` - Best model checkpoint (val_accuracy)
- `cnn_model_TIMESTAMP.h5` - Final saved model
- Training history plots
- Confusion matrix visualization
- Classification report in console

## Authors
- Rakeh Saleem (@rakehsaleem)
- [Your Collaborator Name]

