# VBI Bridge Project

## Overview
Vehicle-Bridge Interaction (VBI) analysis project for bridge damage detection using Convolutional Neural Networks (CNN).

## Project Structure
- **clip.py**: Script to clip FFT normalized data to first 250 frequency bins
- **cnn_data_loader.py**: Data loader for CNN training that automatically splits bridges (11m, 13m, 17m for training; 15m for testing)
- **Results/**: Contains Monte Carlo simulation results with 4 different bridges and 5 damage conditions each

## Data Processing Pipeline

### 1. Data Clipping
Use `clip.py` to clip aggregated FFT normalized data to 250 frequency bins:
```bash
python clip.py
```

### 2. Data Loading
Use `cnn_data_loader.py` to load and organize data for CNN training:
```bash
python cnn_data_loader.py
```

The loader automatically:
- Loads training data from bridges: 11m (Sim1), 13m (Sim2), 17m (Sim4)
- Loads test data from bridge: 15m (Sim3)
- Combines all 5 damage conditions (DC0-DC4) for each bridge
- Returns X_train, Y_train, X_test, Y_test ready for Keras/TensorFlow

## Data Structure
Each simulation contains 5 damage conditions:
- DC0: No damage
- DC1: Light damage
- DC2: Moderate damage
- DC3: Severe damage
- DC4: Critical damage

## Dependencies
- numpy
- scipy
- matplotlib (for visualization)
- tensorflow/keras (for CNN training)

## Authors
- [Your Name]
- [Collaborator Name]

