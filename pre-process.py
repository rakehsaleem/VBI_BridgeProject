import numpy as np
import scipy.io as sio
from scipy.fft import fft
import os
from pathlib import Path
import glob
import sys 

INPUT_DIR = r"C:\Users\Rakeh-PC\Downloads\ZProject\VBI\Results\M01_BlankMonteCarlo_Temperature\Scenario1_4axlesload\Simulation04\Sim4_17m_DC2"

# The complete path of nested keys leading to the final data array 'A'
NESTED_KEY_PATH = ['Run', 'Sol', 'Veh', 'A']

# The slice you want to extract from the final array.
# Extracts the 1st row (index 0) AND the last row (index -1). This results in a (2, N) array.
FINAL_ARRAY_SLICE = [[0, -1], slice(None)] 

# The name of the final output .mat file
OUTPUT_FILENAME = 'aggregated_run_sol_data_FFT_normalized.mat' 

# The name of the combined variable inside the output .mat file
AGGREGATED_VARIABLE_NAME = 'combined_A_FFT_spectra' 

# We use 'MAX' to pad to the longest signal found so all FFT inputs are the same length.
TARGET_FFT_LENGTH = 'MAX' 

# Set this flag to True (or False) to toggle detailed print statements during processing.
DEBUG_MODE = False 

# --- CORE FUNCTION: FFT PROCESSING ---
def process_to_fft(arr, max_len):
    """
    Pads the input array to max_len (using 0.0), and returns the normalized 
    magnitude spectrum (single-sided FFT).
    """
    
    # 1. Pad the array to a uniform length (max_len) using 0.0
    padded_arr = pad_array(arr, max_len, fill_value=0.0)
    
    processed_rows = []
    
    # Iterate over each row extracted (e.g., the 1st and last row)
    for row in padded_arr:
        # Calculate FFT for the row
        fft_values = fft(row)
        
        # Calculate the magnitude spectrum (amplitude)
        magnitude = np.abs(fft_values)
        
        # Truncate to the single-sided spectrum (first half)
        N = len(row)
        single_sided_magnitude = magnitude[:N // 2]
        
        # Normalize by length (standard practice for comparative spectra)
        normalized_magnitude = single_sided_magnitude / N
        
        # Reshape back to (1, N//2)
        processed_rows.append(normalized_magnitude[np.newaxis, :])
    
    # Stack the processed rows back into a (2, N//2) array for the file
    return np.vstack(processed_rows)


def pad_array(arr, target_length, fill_value=0.0): 
    """Pads a 1D or 2D array to the target length along the last dimension (columns) with a specified fill value."""
    current_length = arr.shape[-1]
    
    if current_length == target_length:
        return arr
    
    padding_needed = target_length - current_length
    
    if arr.ndim == 1:
        padding = np.full(padding_needed, fill_value, dtype=arr.dtype)
        return np.concatenate([arr, padding])
    elif arr.ndim == 2:
        # For a 2D array (R, N), pad along axis 1
        padding = np.full((arr.shape[0], padding_needed), fill_value, dtype=arr.dtype)
        return np.concatenate([arr, padding], axis=1)
    else:
        raise ValueError(f"Array has unexpected dimensions ({arr.ndim}). Cannot pad.")


def aggregate_mat_data():
    """
    Scans the input directory for .mat files, extracts the specified rows, 
    converts them to FFT spectra, and aggregates the results.
    """
    input_path = Path(INPUT_DIR)
    
    if not input_path.is_dir():
        print(f"Error: The specified input directory was not found: {input_path}")
        sys.exit(1)

    search_pattern = input_path / '*.mat'
    mat_files = glob.glob(str(search_pattern))
    
    if not mat_files:
        print(f"Error: No .mat files found in '{INPUT_DIR}'. Aborting.")
        return

    print(f"Found {len(mat_files)} .mat files to process.")
    print(f"Targeting data path: {' -> '.join(NESTED_KEY_PATH)}")
    
    raw_extracted_data = []
    processed_files = [] 
    max_length = 0
    
    # --- Phase 1: Extraction and Max Length Determination (on RAW DATA) ---
    for filepath in mat_files:
        filename = os.path.basename(filepath)
        
        try:
            data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
            current_data = data
            
            is_valid = True
            for i, key in enumerate(NESTED_KEY_PATH):
                
                # --- ACCESS LOGIC ---
                if isinstance(current_data, sio.matlab.mio5_params.mat_struct):
                    if not hasattr(current_data, key):
                        is_valid = False
                        break
                    current_data = getattr(current_data, key)
                elif isinstance(current_data, dict):
                    if key not in current_data:
                        is_valid = False
                        break
                    current_data = current_data[key]
                elif isinstance(current_data, np.ndarray) and current_data.dtype.fields is not None:
                    if key not in current_data.dtype.names:
                         is_valid = False
                         break
                    current_data = current_data[key]
                else:
                    is_valid = False
                    break
                
                # --- UNWRAPPING LOGIC ---
                if i < len(NESTED_KEY_PATH) - 1:
                    if isinstance(current_data, np.ndarray) and current_data.shape == (1, 1):
                        current_data = current_data.item()
                    elif isinstance(current_data, np.ndarray) and current_data.ndim == 0:
                         current_data = current_data.item()
            
            if is_valid:
                final_array = current_data

                # Final cleaning step
                if isinstance(final_array, np.ndarray) and final_array.shape == (1, 1):
                    final_array = final_array.item()

                if not isinstance(final_array, np.ndarray):
                    raise ValueError(f"Final object is not a NumPy array ({type(final_array).__name__}).")

                # Apply final slice
                if FINAL_ARRAY_SLICE is not None:
                    final_slice_tuple = tuple(FINAL_ARRAY_SLICE)
                    
                    # Ensure the array has at least 2 rows for the slice [0, -1] to work
                    if final_array.shape[0] < 2: 
                        print(f"Warning: File {filename} has {final_array.shape[0]} row(s). Skipping first/last row extraction.")
                        continue 
                        
                    extracted_data = final_array[final_slice_tuple]
                else:
                    extracted_data = final_array

                # Store raw data and update max length
                raw_extracted_data.append(extracted_data)
                processed_files.append(filename)
                max_length = max(max_length, extracted_data.shape[1])
            
        except Exception as e:
            print(f"Error processing file {filename} during data extraction: {e}")

    # --- Phase 2: FFT Processing and Combination ---
    if not raw_extracted_data:
        print("No data was successfully extracted. Aborting save operation.")
        return

    print(f"\nSuccessfully extracted {len(raw_extracted_data)} raw data segments.")
    print(f"Maximum segment length found: {max_length} columns.")

    target_fft_output_length = max_length // 2
    
    fft_processed_data = []
    
    for i, arr in enumerate(raw_extracted_data):
        # Process the raw data segment into its FFT spectrum
        fft_spectrum = process_to_fft(arr, max_length)
        fft_processed_data.append(fft_spectrum)
        
        # Display padding information
        if arr.shape[1] < max_length:
            print(f"  Padded file '{processed_files[i]}' from {arr.shape[1]} to {max_length} samples before FFT.")


    # 3. Combine Data and Save
    try:
        combined_array = np.vstack(fft_processed_data)
        
        output_path = Path(INPUT_DIR) / OUTPUT_FILENAME
        save_data = {AGGREGATED_VARIABLE_NAME: combined_array}
        
        # Save to the same directory as the input files
        sio.savemat(str(output_path), save_data)
        
        print("\n--- RESULTS ---")
        print(f"Data for **FFT Magnitude Spectra** successfully aggregated and saved to: {output_path}")
        print(f"Output columns (frequency bins): {target_fft_output_length}")
        print(f"Combined shape: {combined_array.shape}")
        print(f"The variable name in the new file is: '{AGGREGATED_VARIABLE_NAME}'")

    except Exception as e:
        print(f"\nError saving the output file: {e}")


if __name__ == '__main__':
    
    print("Starting deep nested data aggregation and FFT normalization process...")
    aggregate_mat_data() 
    print("\nProcess finished.")