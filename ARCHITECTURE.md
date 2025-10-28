# VBI Bridge Project - Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VBI BRIDGE DAMAGE DETECTION                      │
│                       CNN CLASSIFICATION SYSTEM                      │
└─────────────────────────────────────────────────────────────────────┘

```

## Data Flow Pipeline

```
1. RAW SIMULATION DATA
   └─ Results/M01_BlankMonteCarlo_Temperature/Scenario1_4axlesload/
      ├─ Simulation01 (11m bridge)
      ├─ Simulation02 (13m bridge)  → TRAINING (3 bridges)
      ├─ Simulation03 (15m bridge)  → TEST (1 bridge)
      └─ Simulation04 (17m bridge)
         ├─ Sim*_DC0/ (No damage)
         ├─ Sim*_DC1/ (Light damage)
         ├─ Sim*_DC2/ (Moderate damage)
         ├─ Sim*_DC3/ (Severe damage)
         └─ Sim*_DC4/ (Critical damage)
            └─ aggregated_run_sol_data_FFT_normalized.mat
                └─ (2000 rows, 2700+ columns) FFT spectra
```

```
2. DATA CLIPPING (clip.py)
   └─ clipped_run_sol_data_FFT_normalized.mat
      └─ (2000 rows, 250 columns) First 250 frequency bins
          └─ Keep: aggregated_run_sol_data_FFT_normalized_Clipped.mat
```

```
3. DATA LOADING (cnn_data_loader.py)
   
   Input: (2000, 250) 
   ├─ Row 0, 2, 4, ... = Sensor 1
   ├─ Row 1, 3, 5, ... = Sensor 2
   └─ Each pair = One vehicle passing
   
   Reshape: (2000, 250) → (1000, 250, 2)
   └─ (samples, frequency_bins, sensors)
   
   Split:
   ├─ TRAIN: Simulation01, 02, 04 (3 bridges × 5 damage × 1000 samples = 15,000 samples)
   └─ TEST: Simulation03 (1 bridge × 5 damage × 1000 samples = 5,000 samples)
   
   Labels:
   └─ DC0=0, DC1=1, DC2=2, DC3=3, DC4=4
```

## CNN Architecture

```
INPUT: (None, 250, 2)
├─ Conv1D.Parameters: 32 filters, kernel=5, activation='relu'
├─ BatchNormalization()
├─ MaxPooling1D(pool_size=2) → (125, 32)
├─ Dropout(0.25)
│
├─ Conv1D.Parameters: 64 filters, kernel=5, activation='relu'
├─ BatchNormalization()
├─ MaxPooling1D(pool_size=2) → (62, 64)
├─ Dropout(0.25)
│
├─ Conv1D.Parameters: 128 filters, kernel=3, activation='relu'
├─ BatchNormalization()
├─ MaxPooling1D(pool_size=2) → (31, 128)
├─ Dropout(0.3)
│
├─ Flatten() → (3968)
├─ Dense(128, activation='relu')
├─ Dropout(0.5)
├─ Dense(64, activation='relu')
├─ Dropout(0.5)
│
└─ OUTPUT: Dense(5, activation='softmax') → [DC0, DC1, DC2, DC3, DC4]
```

## Training Pipeline

```
┌──────────────────────┐
│  cnn_data_loader.py  │
│  Load & reshape data │
└──────────┬───────────┘
           │ X_train: (15000, 250, 2)
           │ Y_train: (15000,) → One-hot → (15000, 5)
           │ X_test: (5000, 250, 2)
           │ Y_test: (5000,) → One-hot → (5000, 5)
           ▼
┌──────────────────────┐
│    train_cnn.py      │
│                      │
│ 1. Load data         │
│ 2. Build CNN model   │
│ 3. Train (50 epochs) │
│ 4. Evaluate          │
│ 5. Save model        │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  training_results/   │
│                      │
│ • best_cnn_model.h5  │
│ • history.png        │
│ • confusion.png      │
└──────────────────────┘
```

## File Structure

```
VBI_BridgeProject/
├── clip.py                    # Clip FFT data to 250 bins
├── cnn_data_loader.py         # Load & reshape data (alt rows = sensors)
├── train_cnn.py               # Train CNN model
├── pre-process.py             # Create aggregated FFT files
├── Results/                   # Simulation data
│   └── M01_BlankMonteCarlo_Temperature/
│       └── Scenario1_4axlesload/
│           ├── Simulation01/  (11m - TRAIN)
│           ├── Simulation02/  (13m - TRAIN)
│           ├── Simulation03/  (15m - TEST)
│           └── Simulation04/  (17m - TRAIN)
│               └── Sim*_DC*/aggregated_run_sol_data_FFT_normalized_Clipped.mat
├── training_results/          # Model outputs
│   ├── best_cnn_model.h5
│   ├── CNN_*_history.png
│   └── CNN_*_confusion.png
└── README.md
```

## Key Features

**Data Organization:**
- Alternate rows represent sensor 1 and sensor 2
- Each pair (sensor1, sensor2) = one vehicle passage
- 1000 vehicle passages per damage condition

**Model Architecture:**
- 1D CNN for sequence data (frequency bins)
- 2 input channels (dual sensor readings)
- 5 output classes (damage levels DC0-DC4)

**Training Strategy:**
- Train on 3 bridges (11m, 13m, 17m)
- Test on 1 bridge (15m)
- Early stopping, model checkpointing, learning rate reduction

