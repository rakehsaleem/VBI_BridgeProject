import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import glob
import sys 
from typing import List, Any, Dict

# --- CONFIGURATION: Define what to clip and where to find the files ---
config = {
    # !!! INPUT_ROOT_DIR: Set this to the parent folder containing all your subdirectories !!!
    'INPUT_ROOT_DIR': r"C:\Users\Rakeh-PC\Downloads\VBI_BridgeProject\Results\M01_BlankMonteCarlo_Temperature\Scenario1_4axlesload",
    
    # Target file name to process (only process these aggregated files)
    'TARGET_FILENAME': 'aggregated_run_sol_data_FFT_normalized.mat',
    
    # Variable name inside the aggregated files (these files already contain FFT spectra)
    'INPUT_VARIABLE_NAME': 'combined_A_FFT_spectra',

    # Clipping Parameters
    'TARGET_SAMPLES': 250,        # The number of frequency bins to keep (first 250)
    
    # Output Configuration
    'OUTPUT_FILENAME_SUFFIX': '_Clipped.mat',
    'OUTPUT_VARIABLE_NAME': 'Clipped_FFT_Spectra',  # The variable name inside the new .mat file
}

# --- HELPER FUNCTIONS (Kept robust for nested access) ---

def get_nested_data(data: Dict[str, Any], path: List[str]) -> np.ndarray:
    """Recursively navigates the nested structure of loaded MATLAB data to find the array."""
    current_data = data
    
    for i, key in enumerate(path):
        # Access Logic
        if isinstance(current_data, sio.matlab.mio5_params.mat_struct):
            if not hasattr(current_data, key): raise KeyError(f"Key '{key}' not found.")
            current_data = getattr(current_data, key)
        elif isinstance(current_data, dict):
            if key not in current_data: raise KeyError(f"Key '{key}' not found.")
            current_data = current_data[key]
        elif isinstance(current_data, np.ndarray) and current_data.dtype.fields is not None:
            if key not in current_data.dtype.names: raise KeyError(f"Key '{key}' not found.")
            current_data = current_data[key]
        else:
            if i == len(path) - 1 and isinstance(current_data, np.ndarray): break
            raise TypeError(f"Data structure mismatch at level {i+1} ('{key}').")
        
        # Unwrapping Logic
        if isinstance(current_data, np.ndarray) and (current_data.shape == (1, 1) or current_data.ndim == 0):
             current_data = current_data.item()

    if not isinstance(current_data, np.ndarray):
        raise TypeError(f"Final object extracted is not a NumPy array ({type(current_data).__name__}).")

    return current_data


def batch_clip_and_save():
    """Main function to find aggregated FFT files, clip the spectra, and save the new .mat files."""
    root_path = Path(config['INPUT_ROOT_DIR'])
    
    if not root_path.is_dir():
        print(f"Error: The specified root directory was not found: {root_path}")
        sys.exit(1)

    # Search recursively in all subdirectories for the specific aggregated files
    target_filename = config['TARGET_FILENAME']
    search_pattern = str(root_path / '**' / target_filename)
    mat_files = glob.glob(search_pattern, recursive=True)
    
    if not mat_files:
        print(f"Error: No '{target_filename}' files found recursively under '{config['INPUT_ROOT_DIR']}'. Aborting.")
        return

    print("--- STARTING BATCH FFT SPECTRA CLIPPING ---")
    print(f"Found {len(mat_files)} aggregated FFT files across all subdirectories.")
    print(f"Targeting variable: {config['INPUT_VARIABLE_NAME']}")
    print(f"Clipping to: {config['TARGET_SAMPLES']} frequency bins.")
    print(f"Processing 4 different bridges (Simulation01-04) with 5 damage conditions each.\n")
    
    processed_count = 0
    
    for filepath_str in mat_files:
        filepath = Path(filepath_str)
        filename = filepath.name
        
        try:
            # Load the data
            data = sio.loadmat(filepath_str, struct_as_record=False, squeeze_me=True)
            
            # Extract the FFT spectra array
            if config['INPUT_VARIABLE_NAME'] not in data:
                print(f"Warning: {filename} does not contain variable '{config['INPUT_VARIABLE_NAME']}'. Skipping.")
                continue
                
            original_array = data[config['INPUT_VARIABLE_NAME']]
            
            # Validation: Check if array has enough samples
            if original_array.shape[1] < config['TARGET_SAMPLES']:
                print(f"Warning: {filename} has only {original_array.shape[1]} frequency bins. Cannot clip to {config['TARGET_SAMPLES']}. Skipping.")
                continue
            
            # Clip the data: keep all rows (all Monte Carlo runs), but clip columns (frequency bins)
            clipped_array = original_array[:, :config['TARGET_SAMPLES']]
            
            # Construct the new filename (e.g., 'aggregated_run_sol_data_FFT_normalized_Clipped.mat')
            base_name = filepath.stem
            output_filename = base_name + config['OUTPUT_FILENAME_SUFFIX']
            
            # Save in the same directory as the original file
            output_path = filepath.parent / output_filename
            
            # Save the clipped data with the specified variable name
            save_data = {config['OUTPUT_VARIABLE_NAME']: clipped_array}
            sio.savemat(str(output_path), save_data)
            
            print(f"✓ Clipped & Saved: {filepath.parent.name}/{output_filename}")
            print(f"  Shape: {original_array.shape} -> {clipped_array.shape}")
            processed_count += 1
            
        except KeyError as e:
            print(f"Error: {filename} missing expected structure. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\n--- BATCH CLIPPING COMPLETE ---")
    print(f"Successfully processed and saved {processed_count} new clipped files.")

# --- EXECUTION BLOCK ---

def main():
    batch_clip_and_save() 
    
    # Pause the console for viewing output
    input("Press Enter to close this window...")

if __name__ == '__main__':
    main()