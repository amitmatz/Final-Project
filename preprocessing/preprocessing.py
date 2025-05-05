# preprocessing/preprocessing.py

import os
import numpy as np
import pandas as pd
import json
import h5py
from defines import BASE_DATA_PATH, PROCESSED_DATA_DIR, PATIENTS_CONFIG_PATH


def load_mat_file(filepath):
    """
    Load LFP time vector and signals from MATLAB v7.3 file (microseconds).
    """
    with h5py.File(filepath, 'r') as f:
        refs = f['#refs#']
        row_times = refs['f']['rowTimes']
        origin = np.array(row_times['origin']).flatten()[0]
        step_size = np.array(row_times['stepSize']).flatten()[0]
        num_samples = int(np.array(refs['f']['numRows'])[0][0])

        times_microsec = np.arange(0, num_samples) * step_size + origin

        data_ref = refs['f']['data'][0][0]
        lfp_values = np.array(f[data_ref]).flatten()

        print(f"[DEBUG] Loaded {len(times_microsec)} times and {len(lfp_values)} values from {os.path.basename(filepath)}")

        return times_microsec, lfp_values


def load_labels(filepath):
    """
    Load labels from text file (seconds units).
    """
    return pd.read_csv(filepath, delimiter='\t', header=None, names=['start', 'end', 'label'])


def load_start_time_and_sampling_rate(offset_file_path):
    """
    Load recording start time and sampling rate from sound_w_times.mat
    """
    with h5py.File(offset_file_path, 'r') as f:
        start_time = np.array(f['new_start_end_times_micsec'])[0][0]
        sampling_rate = np.array(f['FS'])[0][0]
        print(f"[DEBUG] Loaded start time: {start_time} µsec, FS: {sampling_rate} Hz")
        return start_time, sampling_rate


def process_patient(patient_name, patient_info):
    """
    Process patient data: load signals, adjust label times, save segments.
    """
    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info['lfp_folder'])
    labels_path = os.path.join(BASE_DATA_PATH, patient_info['labels_file'])
    offset_path = os.path.join(BASE_DATA_PATH, patient_info['offset_file'])

    print(f"\n----- Processing {patient_name} -----")

    # Load labels (seconds)
    labels_df = load_labels(labels_path)
    print(f"[DEBUG] Loaded {len(labels_df)} labels")
    print("[DEBUG] First 3 labels BEFORE adjustment:")
    print(labels_df.head(3))

    # Convert labels to microseconds
    labels_df['start'] = labels_df['start'] * 1e6
    labels_df['end'] = labels_df['end'] * 1e6

    # Load start time and sampling rate
    start_time_microsec, sampling_rate_hz = load_start_time_and_sampling_rate(offset_path)

    # Shift labels
    labels_df['start'] = labels_df['start'] + start_time_microsec
    labels_df['end'] = labels_df['end'] + start_time_microsec

    print(f"\n[DEBUG] Labels converted to microseconds and offset applied: {start_time_microsec} µsec")
    print("[DEBUG] First 3 labels AFTER adjustment:")
    print(labels_df.head(3))

    # Match labels to LFP signals per electrode
    data_struct = []

    for file in sorted(os.listdir(lfp_dir)):
        if file.endswith('.mat'):
            electrode_name = os.path.splitext(file)[0]
            file_path = os.path.join(lfp_dir, file)
            times, values = load_mat_file(file_path)

            for idx, row in labels_df.iterrows():
                start, end, label = row['start'], row['end'], row['label']
                indices = np.where((times >= start) & (times <= end))[0]
                if len(indices) > 0:
                    segment = values[indices]
                    data_struct.append({
                        "signal": segment,
                        "label": label,
                        "electrode": electrode_name
                    })
                else:
                    print(f"⚠️ Warning: No signal found for label '{label}' on electrode '{electrode_name}' ({start}-{end})")

    # Save result
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_data.npy")
    np.save(output_path, data_struct)
    print(f"\n✅ Saved processed data for {patient_name}. Total segments: {len(data_struct)}")


def process_all_patients():
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    for patient_name, patient_info in config['patients'].items():
        process_patient(patient_name, patient_info)
