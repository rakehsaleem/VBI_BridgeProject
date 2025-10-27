import numpy as np
import scipy.io as sio
from pathlib import Path
import glob

# --- CONFIGURATION ---
config = {
    # Directory containing the data (should contain Simulations folders)
    'DATA_DIR': Path('Results/M01_BlankMonteCarlo_Temperature/Scenario1_4axlesload'),
    
    # Filename pattern for clipped aggregated files
    'CLIPPED_FILENAME': 'aggregated_run_sol_data_FFT_normalized_Clipped.mat',
    
    # Variable name inside the clipped .mat files
    'DATA_VARIABLE_NAME': 'Clipped_FFT_Spectra',
    
    # Bridge configurations
    # Training bridges: 11m (Sim1), 13m (Sim2), 17m (Sim4)
    # Test bridge: 15m (Sim3)
    'TRAIN_BRIDGES': ['Simulation01', 'Simulation02', 'Simulation04'],
    'TEST_BRIDGES': ['Simulation03'],
    
    # Damage conditions (DC0-DC4)
    'DAMAGE_CONDITIONS': ['DC0', 'DC1', 'DC2', 'DC3', 'DC4']
}


def load_bridge_data(data_dir, simulation_folder, damage_condition):
    """
    Load clipped data for a specific bridge and damage condition.
    
    Args:
        data_dir: Base directory (Scenario1_4axlesload)
        simulation_folder: Bridge folder (e.g., 'Simulation01')
        damage_condition: Damage level (e.g., 'DC0', 'DC1', etc.)
    
    Returns:
        tuple: (data_array, label) where label is the damage level (0-4)
    """
    # Construct path to the damage condition folder
    # Pattern: SimulationXX/SimX_XXm_DCY
    damage_folder_pattern = f"Sim*_{damage_condition}"
    search_path = data_dir / simulation_folder / damage_folder_pattern
    
    # Find matching folders
    matching_folders = list(search_path.parent.glob(damage_folder_pattern))
    
    if not matching_folders:
        raise FileNotFoundError(f"No damage folder found matching {damage_folder_pattern} in {simulation_folder}")
    
    damage_folder = matching_folders[0]
    
    # Load the clipped file
    clipped_file = damage_folder / config['CLIPPED_FILENAME']
    
    if not clipped_file.exists():
        raise FileNotFoundError(f"File not found: {clipped_file}")
    
    # Load data
    data = sio.loadmat(str(clipped_file))
    
    if config['DATA_VARIABLE_NAME'] not in data:
        raise KeyError(f"Variable '{config['DATA_VARIABLE_NAME']}' not found in {clipped_file}")
    
    X = data[config['DATA_VARIABLE_NAME']]
    
    # Reshape from (2000, 250) to (1000, 250, 2)
    # Rows alternate: sensor1, sensor2, sensor1, sensor2, ...
    # Extract sensor 1 data (even rows) and sensor 2 data (odd rows)
    sensor1 = X[::2, :]   # Even rows: 0, 2, 4, 6, ... (sensor 1)
    sensor2 = X[1::2, :]  # Odd rows: 1, 3, 5, 7, ... (sensor 2)
    
    # Stack to create (1000, 250, 2) shape where each sample has both sensors
    X_reshaped = np.stack([sensor1, sensor2], axis=2)  # (num_samples, 250, 2)
    
    # Extract damage level from damage condition (DC0 -> 0, DC1 -> 1, etc.)
    damage_level = int(damage_condition.replace('DC', ''))
    
    return X_reshaped, damage_level


def load_data_sets():
    """
    Load all training and test data from the bridge simulation folders.
    
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test) where Y arrays are flattened
    """
    data_dir = config['DATA_DIR']
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []
    
    # Load training data (bridges: 11m, 13m, 17m)
    print("Loading training data from bridges: 11m, 13m, 17m")
    for bridge in config['TRAIN_BRIDGES']:
        print(f"  Processing {bridge}...")
        for damage in config['DAMAGE_CONDITIONS']:
            try:
                X, Y = load_bridge_data(data_dir, bridge, damage)
                X_train_list.append(X)
                # Create labels array matching the number of samples in X
                Y_labels = np.full(X.shape[0], Y)  # One label per sample
                Y_train_list.append(Y_labels)
                print(f"    {damage}: Shape {X.shape}, Label: {Y}")
            except Exception as e:
                print(f"    Failed to load {damage}: {e}")
    
    # Load test data (bridge: 15m)
    print("\nLoading test data from bridge: 15m")
    for bridge in config['TEST_BRIDGES']:
        print(f"  Processing {bridge}...")
        for damage in config['DAMAGE_CONDITIONS']:
            try:
                X, Y = load_bridge_data(data_dir, bridge, damage)
                X_test_list.append(X)
                Y_labels = np.full(X.shape[0], Y)
                Y_test_list.append(Y_labels)
                print(f"    {damage}: Shape {X.shape}, Label: {Y}")
            except Exception as e:
                print(f"    Failed to load {damage}: {e}")
    
    # Combine all arrays
    if not X_train_list:
        raise ValueError("No training data loaded!")
    if not X_test_list:
        raise ValueError("No test data loaded!")
    
    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    
    X_test = np.vstack(X_test_list)
    Y_test = np.concatenate(Y_test_list)
    
    return X_train, Y_train, X_test, Y_test


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    try:
        print("=" * 60)
        print("CNN Data Loader - Bridge Damage Classification")
        print("=" * 60)
        print(f"\nData directory: {config['DATA_DIR']}")
        print(f"\nTraining bridges: {', '.join(config['TRAIN_BRIDGES'])}")
        print(f"Test bridges: {', '.join(config['TEST_BRIDGES'])}")
        print(f"Damage conditions: {', '.join(config['DAMAGE_CONDITIONS'])}")
        print("\n" + "=" * 60 + "\n")
        
        # Load all data
        X_train, Y_train, X_test, Y_test = load_data_sets()
        
        print("\n" + "=" * 60)
        print("Data Loading Summary")
        print("=" * 60)
        
        # Print shapes
        print(f"\nTraining Data:")
        print(f"  X_train.shape = {X_train.shape}")
        print(f"  Y_train.shape = {Y_train.shape}")
        print(f"  Total samples: {len(Y_train)}")
        
        print(f"\nTest Data:")
        print(f"  X_test.shape = {X_test.shape}")
        print(f"  Y_test.shape = {Y_test.shape}")
        print(f"  Total samples: {len(Y_test)}")
        
        # Verify label flattening (should be 1D)
        print(f"\nLabel Verification:")
        print(f"  Y_train is 1D: {Y_train.ndim == 1}")
        print(f"  Y_test is 1D: {Y_test.ndim == 1}")
        
        # Show label distribution
        print(f"\nTraining Label Distribution:")
        unique, counts = np.unique(Y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  DC{int(label)}: {count} samples")
        
        print(f"\nTest Label Distribution:")
        unique, counts = np.unique(Y_test, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  DC{int(label)}: {count} samples")
        
        # Data type information
        print(f"\nData Types:")
        print(f"  X_train.dtype = {X_train.dtype}")
        print(f"  Y_train.dtype = {Y_train.dtype}")
        print(f"  X_test.dtype = {X_test.dtype}")
        print(f"  Y_test.dtype = {Y_test.dtype}")
        
        print("\n" + "=" * 60)
        print("All data loaded successfully and ready for CNN training!")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print("Please ensure the data directory exists and contains the clipped .mat files.")
    except ValueError as e:
        print("Please check that all damage condition folders contain the required .mat files.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
