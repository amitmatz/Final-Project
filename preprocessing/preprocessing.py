import os
import numpy as np
import pandas as pd
import json
import h5py
from defines import BASE_DATA_PATH, PROCESSED_DATA_DIR, PATIENTS_CONFIG_PATH


def load_mat_file(filepath):
    """
    Loads LFP signals and builds synthetic time vector with 1ms steps.
    Ensures time and signal arrays match in length.
    """
    with h5py.File(filepath, 'r') as f:
        lfp_tt = f['LFP_tt']
        values = np.array(lfp_tt).squeeze()

        num_samples = len(values)
        times = np.arange(0, num_samples) * 0.001  # 1kHz

    print(f"[DEBUG] num_samples (based on values): {num_samples}")
    return times, values




def load_labels(filepath):
    """
    Loads label timestamps from a tab-delimited text file.
    Expected format: start_time, end_time, label.
    """
    return pd.read_csv(filepath, delimiter='\t', header=None, names=['start', 'end', 'label'])


def normalize_label(label):
    """
    Maps various label variants to unified categories.    Ensures consistent naming across training samples.
    """
    if label in ['Haarye', 'Arye']:
        return 'Arye'
    elif label == 'Ahav':
        return 'Ahav'
    elif label == 'Tut':
        return 'Tut'
    else:
        return label  # Unrecognized labels are passed through


def process_patient(patient_name, patient_info):
    """
    Processes one patient:
    - Loads LFP signals from .mat files
    - Loads label timings from text
    - Applies offset to align labels with neural data
    - Extracts labeled signal segments
    - Saves the processed data as .npy
    """
    OFFSET = 558.14  # seconds

    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info['lfp_folder'])
    labels_path = os.path.join(BASE_DATA_PATH, patient_info['labels_file'])

    print(f"Processing {patient_name}...")
    print(f"Looking for LFP signals at: {lfp_dir}")
    print(f"Exists? {os.path.exists(lfp_dir)}")

    all_times, all_values = [], []

    # Load all LFP files
    for file in sorted(os.listdir(lfp_dir)):
        if file.endswith('.mat'):
            times, values = load_mat_file(os.path.join(lfp_dir, file))
            all_times.append(times)
            all_values.append(values)

    # Concatenate and proceed
    all_times = np.concatenate(all_times)
    all_values = np.concatenate(all_values)

    print("✅ Example all_times (in seconds):", all_times[:5])
    print(f"LFP time range: {all_times[0]} -> {all_times[-1]}")

    labels_df = load_labels(labels_path)
    print(f"First label (raw): {labels_df.iloc[0]}")
    print(f"First label (with offset): {labels_df.iloc[0]['start'] + OFFSET} -> {labels_df.iloc[0]['end'] + OFFSET}")

    data_struct = []

    # Extract signal segments
    for idx, row in labels_df.iterrows():
        start = row['start'] + OFFSET
        end = row['end'] + OFFSET
        label = normalize_label(row['label'])

        indices = np.where((all_times >= start) & (all_times <= end))[0]
        if len(indices) > 0:
            segment = all_values[indices]
            data_struct.append({"signal": segment, "label": label})
        else:
            print(f"Warning: No signal found for label {label} ({start}-{end})")

    # Save to file
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_data.npy"), data_struct)
    print(f"✅ Saved processed data for {patient_name}. Total segments: {len(data_struct)}")



def process_all_patients():
    """
    Loads configuration from JSON file and processes each patient listed.
    """
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    for patient_name, patient_info in config['patients'].items():
        process_patient(patient_name, patient_info)
