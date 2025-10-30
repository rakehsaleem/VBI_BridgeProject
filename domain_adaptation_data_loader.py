import numpy as np
import scipy.io as sio
from pathlib import Path
import glob
from typing import Tuple, Dict, List

# --- CONFIGURATION ---
config = {
    'DATA_DIR': Path('Results/M01_BlankMonteCarlo_Temperature/Scenario1_4axlesload'),
    
    # Filename pattern for clipped aggregated files
    'CLIPPED_FILENAME': 'aggregated_run_sol_data_FFT_normalized_Clipped.mat',
    
    # Variable name inside the clipped .mat files
    'DATA_VARIABLE_NAME': 'Clipped_FFT_Spectra',
    
    # Source domains: Fully labeled bridges
    'SOURCE_BRIDGES': ['Simulation01', 'Simulation02', 'Simulation04'],
    'SOURCE_BRIDGE_NAMES': {'Simulation01': '11m', 'Simulation02': '13m', 'Simulation04': '17m'},
    
    # Target domain: Partially labeled bridge
    'TARGET_BRIDGE': 'Simulation03',
    'TARGET_BRIDGE_NAME': '15m',
    
    # Damage conditions (DC0-DC4)
    'ALL_DAMAGE_CONDITIONS': ['DC0', 'DC1', 'DC2', 'DC3', 'DC4'],
    
    # Target domain: Which damage condition has labels?
    # Only samples with this label are labeled in target domain
    'TARGET_LABELED_DAMAGE': 'DC0',  # CHANGE THIS based on your data
    
    # All damage conditions present in target domain (labeled + unlabeled)
    'TARGET_DAMAGE_CONDITIONS': ['DC0', 'DC1', 'DC2', 'DC3', 'DC4']
}


def load_bridge_data(data_dir, simulation_folder, damage_condition):
    """
    Load clipped data for a specific bridge and damage condition.
    
    Args:
        data_dir: Base directory (Scenario1_4axlesload)
        simulation_folder: Bridge folder (e.g., 'Simulation01')
        damage_condition: Damage level (e.g., 'DC0', 'DC1', etc.)
    
    Returns:
        tuple: (data_array, label, bridge_id)
        - data_array: (num_samples, 250, 2) reshaped data
        - label: damage level (0-4)
        - bridge_id: integer bridge identifier
    """
    # Construct path to the damage condition folder
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
    sensor1 = X[::2, :]   # Even rows: 0, 2, 4, 6, ... (sensor 1)
    sensor2 = X[1::2, :]  # Odd rows: 1, 3, 5, 7, ... (sensor 2)
    
    # Stack to create (1000, 250, 2) shape
    X_reshaped = np.stack([sensor1, sensor2], axis=2)  # (num_samples, 250, 2)
    
    # Extract damage level from damage condition (DC0 -> 0, DC1 -> 1, etc.)
    damage_level = int(damage_condition.replace('DC', ''))
    
    # Bridge ID mapping
    bridge_id = None
    if simulation_folder in config['SOURCE_BRIDGES']:
        # Source bridges: 0, 1, 2
        bridge_id = config['SOURCE_BRIDGES'].index(simulation_folder)
    elif simulation_folder == config['TARGET_BRIDGE']:
        # Target bridge: 3
        bridge_id = 3
    
    return X_reshaped, damage_level, bridge_id


def load_domain_adaptation_data():
    """
    Load data for domain adaptation setup.
    
    Returns:
        dict containing:
        - source_X: (N_source, 250, 2) all source domain data
        - source_Y: (N_source,) source labels (all DC0-DC4)
        - source_bridge_id: (N_source,) bridge IDs (0=11m, 1=13m, 2=17m)
        
        - target_X_labeled: (N_target_labeled, 250, 2) labeled target data
        - target_Y_labeled: (N_target_labeled,) labeled target labels
        - target_X_unlabeled: (N_target_unlabeled, 250, 2) unlabeled target data
        - target_bridge_id: (N_target,) all target bridge IDs (all = 3)
        
        Statistics about the dataset
    """
    data_dir = config['DATA_DIR']
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Lists for source domain (fully labeled)
    source_X_list = []
    source_Y_list = []
    source_bridge_id_list = []
    
    # Lists for target domain
    target_X_labeled_list = []
    target_Y_labeled_list = []
    target_X_unlabeled_list = []
    target_bridge_id_list = []
    
    print("Loading source domains (fully labeled)...")
    for bridge in config['SOURCE_BRIDGES']:
        bridge_name = config['SOURCE_BRIDGE_NAMES'][bridge]
        print(f"  Processing {bridge} ({bridge_name})...")
        for damage in config['ALL_DAMAGE_CONDITIONS']:
            try:
                X, Y, bridge_id = load_bridge_data(data_dir, bridge, damage)
                # All source data is labeled
                source_X_list.append(X)
                Y_labels = np.full(X.shape[0], Y)
                source_Y_list.append(Y_labels)
                bridge_ids = np.full(X.shape[0], bridge_id)
                source_bridge_id_list.append(bridge_ids)
                print(f"    {damage}: Shape {X.shape}, Label: {Y}")
            except Exception as e:
                print(f"    Failed to load {damage}: {e}")
    
    print(f"\nLoading target domain: {config['TARGET_BRIDGE']} ({config['TARGET_BRIDGE_NAME']})...")
    print(f"  Labeled damage condition: {config['TARGET_LABELED_DAMAGE']}")
    print(f"  Unlabeled damage conditions: {[d for d in config['TARGET_DAMAGE_CONDITIONS'] if d != config['TARGET_LABELED_DAMAGE']]}")
    
    for damage in config['TARGET_DAMAGE_CONDITIONS']:
        try:
            X, Y, bridge_id = load_bridge_data(data_dir, config['TARGET_BRIDGE'], damage)
            
            if damage == config['TARGET_LABELED_DAMAGE']:
                # This damage condition is labeled
                target_X_labeled_list.append(X)
                Y_labels = np.full(X.shape[0], Y)
                target_Y_labeled_list.append(Y_labels)
                print(f"    [LABELED] {damage}: Shape {X.shape}, Label: {Y}")
            else:
                # This damage condition is unlabeled
                target_X_unlabeled_list.append(X)
                print(f"    [UNLABELED] {damage}: Shape {X.shape}")
            
            # Track bridge ID for all target samples
            bridge_ids = np.full(X.shape[0], bridge_id)
            target_bridge_id_list.append(bridge_ids)
            
        except Exception as e:
            print(f"    Failed to load {damage}: {e}")
    
    # Combine all arrays
    if not source_X_list:
        raise ValueError("No source data loaded!")
    
    source_X = np.vstack(source_X_list)
    source_Y = np.concatenate(source_Y_list)
    source_bridge_id = np.concatenate(source_bridge_id_list)
    
    if not target_X_labeled_list:
        raise ValueError("No labeled target data loaded!")
    
    target_X_labeled = np.vstack(target_X_labeled_list)
    target_Y_labeled = np.concatenate(target_Y_labeled_list)
    
    target_X_unlabeled = np.vstack(target_X_unlabeled_list) if target_X_unlabeled_list else np.array([]).reshape(0, 250, 2)
    target_bridge_id = np.concatenate(target_bridge_id_list)
    
    # Print statistics
    print("\n" + "="*70)
    print("Domain Adaptation Data Summary")
    print("="*70)
    print(f"\nSource Domains:")
    print(f"  Total samples: {len(source_Y)}")
    print(f"  Data shape: {source_X.shape}")
    print(f"  Bridge distribution:")
    for i, bridge_name in enumerate(['11m', '13m', '17m']):
        count = np.sum(source_bridge_id == i)
        print(f"    {bridge_name}: {count} samples")
    
    print(f"\nTarget Domain ({config['TARGET_BRIDGE_NAME']}):")
    print(f"  Labeled samples: {len(target_Y_labeled)}")
    print(f"  Unlabeled samples: {len(target_X_unlabeled)}")
    print(f"  Labeled data shape: {target_X_labeled.shape}")
    if len(target_X_unlabeled) > 0:
        print(f"  Unlabeled data shape: {target_X_unlabeled.shape}")
    
    print(f"\nLabel Distribution:")
    print(f"  Source - DC0: {np.sum(source_Y == 0)}, DC1: {np.sum(source_Y == 1)}, "
          f"DC2: {np.sum(source_Y == 2)}, DC3: {np.sum(source_Y == 3)}, DC4: {np.sum(source_Y == 4)}")
    print(f"  Target Labeled - DC{config['TARGET_LABELED_DAMAGE'][-1]}: {len(target_Y_labeled)}")
    
    return {
        'source_X': source_X.astype('float32'),
        'source_Y': source_Y,
        'source_bridge_id': source_bridge_id,
        
        'target_X_labeled': target_X_labeled.astype('float32'),
        'target_Y_labeled': target_Y_labeled,
        'target_X_unlabeled': target_X_unlabeled.astype('float32') if len(target_X_unlabeled) > 0 else None,
        'target_bridge_id': target_bridge_id
    }


if __name__ == '__main__':
    # Test the data loader
    data = load_domain_adaptation_data()
    
    print("\n" + "="*70)
    print("Data Loading Test Successful!")
    print("="*70)
    print(f"\nSource data shape: {data['source_X'].shape}")
    print(f"Source labels shape: {data['source_Y'].shape}")
    print(f"Target labeled shape: {data['target_X_labeled'].shape}")
    if data['target_X_unlabeled'] is not None:
        print(f"Target unlabeled shape: {data['target_X_unlabeled'].shape}")
    print("\nReady for domain adaptation training!")

