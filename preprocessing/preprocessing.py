import os
import numpy as np
import pandas as pd
import json
import h5py
from defines import BASE_DATA_PATH, PROCESSED_DATA_DIR, PATIENTS_CONFIG_PATH

# הקלטה התחילה בזמן מוחלט זה (ביחידות שניות)
LFP_START_TIME = 558138174.511444
SAMPLING_RATE = 1000                  # 1 kHz (כל 1ms)
SAMPLE_STEP = 1.0 / SAMPLING_RATE     # 0.001 שניות בין דגימות


def load_mat_file(filepath):
    """
    Loads LFP signals and builds a real-world time vector using known start time and sampling rate.
    """
    with h5py.File(filepath, 'r') as f:
        lfp_data = f['LFP_tt']
        values = np.array(lfp_data).squeeze()

        num_samples = len(values)
        times = np.arange(0, num_samples) * SAMPLE_STEP + LFP_START_TIME

    return times, values


def load_labels(filepath):
    """
    Loads label timestamps from a tab-delimited file (start, end, label).
    Assumes label times are relative to the beginning of the recording and need offset correction.
    """
    return pd.read_csv(filepath, delimiter='\t', header=None, names=['start', 'end', 'label'])


def normalize_label(label):
    """
    Normalizes label strings to consistent values for classification.
    """
    if label in ['Haarye', 'Arye']:
        return 'Arye'
    elif label == 'Ahav':
        return 'Ahav'
    elif label == 'Tut':
        return 'Tut'
    else:
        return label


def process_patient(patient_name, patient_info):
    """
    Processes one patient's data:
    - Loads LFP signals and constructs absolute timestamps
    - Loads label file and applies offset to align with LFP times
    - Extracts signal segments corresponding to each label
    - Saves the processed dataset as .npy
    """
    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info['lfp_folder'])
    labels_path = os.path.join(BASE_DATA_PATH, patient_info['labels_file'])

    print(f"Processing {patient_name}...")
    print(f"Looking for LFP signals at: {lfp_dir}")
    print(f"Exists? {os.path.exists(lfp_dir)}")

    all_times, all_values = [], []

    for file in sorted(os.listdir(lfp_dir)):
        if file.endswith('.mat'):
            times, values = load_mat_file(os.path.join(lfp_dir, file))
            all_times.append(times)
            all_values.append(values)

    all_times = np.concatenate(all_times)
    all_values = np.concatenate(all_values)

    print("✅ Example all_times (absolute):", all_times[:5])
    print(f"LFP time range: {all_times[0]} -> {all_times[-1]}")

    labels_df = load_labels(labels_path)

    # הוספת offset מוחלט לתוויות
    labels_df['start'] += LFP_START_TIME
    labels_df['end'] += LFP_START_TIME

    print(f"First label (absolute): {labels_df.iloc[0]['start']} -> {labels_df.iloc[0]['end']}")

    data_struct = []

    for idx, row in labels_df.iterrows():
        start = row['start']
        end = row['end']
        label = normalize_label(row['label'])

        indices = np.where((all_times >= start) & (all_times <= end))[0]
        if len(indices) > 0:
            segment = all_values[indices]
            data_struct.append({"signal": segment, "label": label})
        else:
            print(f"⚠️ Warning: No signal found for label {label} ({start}-{end})")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_data.npy")
    np.save(output_path, data_struct)
    print(f"✅ Saved processed data for {patient_name}. Total segments: {len(data_struct)}")


def process_all_patients():
    """
    Reads the patient configuration JSON and processes all defined patients.
    """
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    for patient_name, patient_info in config['patients'].items():
        process_patient(patient_name, patient_info)
