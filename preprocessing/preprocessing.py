# preprocessing/preprocessing.py

import os
import numpy as np
import scipy.io
import pandas as pd
import json
import h5py
from defines import BASE_DATA_PATH, PROCESSED_DATA_DIR, PATIENTS_CONFIG_PATH


def load_mat_file(filepath):
    """
    Load time vector and LFP signal from MATLAB v7.3 file using h5py with proper dereferencing.
    """
    with h5py.File(filepath, 'r') as f:
        refs = f['#refs#']

        # שליפת מידע על הזמן
        row_times = refs['f']['rowTimes']
        origin = np.array(row_times['origin']).flatten()[0] / 1e6  # המרה לשניות
        step_size = np.array(row_times['stepSize']).flatten()[0] / 1e6  # המרה לשניות
        num_samples = int(np.array(refs['f']['numRows'])[0][0])

        times_seconds = np.arange(0, num_samples) * step_size + origin

        # שליפת האותות עם dereference
        data_ref = refs['f']['data'][0][0]
        lfp_values = np.array(f[data_ref]).flatten()

        print(f"[DEBUG] Loaded {len(times_seconds)} times and {len(lfp_values)} values from {os.path.basename(filepath)}")

        return times_seconds, lfp_values



def load_labels(filepath):
    return pd.read_csv(filepath, delimiter='\t', header=None, names=['start', 'end', 'label'])


def load_time_offset(offset_file_path):
    """
    Load time offset from a MATLAB v7.3 .mat file using h5py.
    """
    with h5py.File(offset_file_path, 'r') as f:
        offset_array = np.array(f['new_start_end_times_micsec'])
        offset_microsec = offset_array[0][0]   # לוקחים את הזמן התחלה
        offset_sec = offset_microsec / 1e6     # המרה לשניות
        print(f"Loaded offset: {offset_sec} seconds")
        return offset_sec


def process_patient(patient_name, patient_info):
    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info['lfp_folder'])
    labels_path = os.path.join(BASE_DATA_PATH, patient_info['labels_file'])

    print(f"\n----- Processing {patient_name} -----")

    # --- שלב 1: טוענים את ה-Labels ---
    labels_df = load_labels(labels_path)
    print(f"\n[DEBUG] Loaded {len(labels_df)} labels")
    print("[DEBUG] First 3 labels BEFORE offset adjustment:")
    print(labels_df.head(3))

    # --- שלב 2: טוענים את ה-Offset ---
    offset_file = r'G:\My Drive\FinalProject\Data\Patient_01\ZIP_files_and_backup_data_Patient_01\pt2_LFPs_sound\sound_w_times.mat'
    offset = load_time_offset(offset_file)

    # --- שלב 3: מתאימים את ה-Labels עם offset ---
    labels_df['start'] = labels_df['start'] - offset
    labels_df['end'] = labels_df['end'] - offset

    print(f"\n[DEBUG] Offset applied: {offset} seconds")
    print("[DEBUG] First 3 labels AFTER offset adjustment:")
    print(labels_df.head(3))

    # --- שלב 4: טוענים את כל קבצי ה-LFP ---
    all_times, all_values = [], []
    for file in sorted(os.listdir(lfp_dir)):
        if file.endswith('.mat'):
            times, values = load_mat_file(os.path.join(lfp_dir, file))
            all_times.append(times)
            all_values.append(values)

    all_times = np.concatenate(all_times)
    all_values = np.concatenate(all_values)

    print(f"\n[DEBUG] LFP times range: {all_times.min()} to {all_times.max()}")
    print(f"[DEBUG] Total LFP points: {len(all_times)}")

    # --- שלב 5: חיתוך לפי ה-Labels ---
    data_struct = []
    for idx, row in labels_df.iterrows():
        start, end, label = row['start'], row['end'], row['label']
        indices = np.where((all_times >= start) & (all_times <= end))[0]
        if len(indices) > 0:
            segment = all_values[indices]
            data_struct.append({"signal": segment, "label": label})
        else:
            print(f"Warning: No signal found for label '{label}' ({start}-{end})")

    # --- שלב 6: שמירת התוצאה ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_data.npy"), data_struct)
    print(f"\n✅ Saved processed data for {patient_name}. Total segments: {len(data_struct)}")


def process_all_patients():
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    for patient_name, patient_info in config['patients'].items():
        process_patient(patient_name, patient_info)
