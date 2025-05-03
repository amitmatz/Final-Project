# preprocessing/preprocessing.py

import os
import numpy as np
import pandas as pd
import json
import h5py
from defines import BASE_DATA_PATH, PROCESSED_DATA_DIR, PATIENTS_CONFIG_PATH


# =============================================================
# טוען קובץ LFP
# =============================================================

def load_mat_file(filepath):
    """
    Load LFP signal and construct time vector manually.
    החילוץ הוא ישירות מתוך השדה 'g' בקובץ ה-MATLAB, עם קצב דגימה קבוע של 48000Hz.
    """
    SAMPLE_RATE = 48000  # ידוע שהקלטות הוקלטו ב-48000Hz

    with h5py.File(filepath, 'r') as f:
        refs = f['#refs#']

        # טוען את האות
        signal = np.array(refs['g']).flatten()

        # בונה את הזמנים בעצמי לפי קצב הדגימה
        times_seconds = np.arange(len(signal)) / SAMPLE_RATE

        print(f"[DEBUG] Loaded {len(times_seconds)} times and {len(signal)} values from {os.path.basename(filepath)}")
        print(f"[DEBUG] LFP times range: {times_seconds.min()} to {times_seconds.max()} seconds")

        return times_seconds, signal


# =============================================================
# טוען קובץ תוויות (Labels)
# =============================================================

def load_labels(filepath):
    """
    Load label segments from a tab-separated text file.
    """
    return pd.read_csv(filepath, delimiter='\t', header=None, names=['start', 'end', 'label'])


# =============================================================
# טוען את ה-Offset
# =============================================================

def load_time_offset(offset_file_path):
    """
    Load time offset from sound_w_times.mat.
    """
    with h5py.File(offset_file_path, 'r') as f:
        offset_array = np.array(f['new_start_end_times_micsec'])
        offset_microsec = offset_array[0][0]
        offset_sec = offset_microsec / 1e6
        print(f"Loaded offset: {offset_sec} seconds")
        return offset_sec


# =============================================================
# עיבוד חולה אחד
# =============================================================

def process_patient(patient_name, patient_info):
    """
    Process LFP signals and label segments for a specific patient.
    """
    # נתיב לתיקיית קבצי ה-LFP המתוקנים
    lfp_dir = r'G:\My Drive\FinalProject\Data\Patient_01\ZIP_files_and_backup_data_Patient_01\pt2_LFPs_sound\LFPs'
    labels_path = os.path.join(BASE_DATA_PATH, patient_info['labels_file'])

    print(f"\n----- Processing {patient_name} -----")

    # --- שלב 1: טעינת התוויות ---
    labels_df = load_labels(labels_path)
    print(f"\n[DEBUG] Loaded {len(labels_df)} labels")
    print("[DEBUG] First 3 labels BEFORE offset adjustment:")
    print(labels_df.head(3))

    # --- שלב 2: טעינת ה-Offset ---
    offset_file = r'G:\My Drive\FinalProject\Data\Patient_01\ZIP_files_and_backup_data_Patient_01\pt2_LFPs_sound\sound_w_times.mat'
    offset = load_time_offset(offset_file)

    # --- שלב 3: התאמת התוויות עם Offset ---
    labels_df['start'] = labels_df['start'] - offset
    labels_df['end'] = labels_df['end'] - offset

    print(f"\n[DEBUG] Offset applied: {offset} seconds")
    print("[DEBUG] First 3 labels AFTER offset adjustment:")
    print(labels_df.head(3))

    # --- שלב 4: טעינת כל קבצי ה-LFP ---
    all_times = None
    all_channels = []

    for file in sorted(os.listdir(lfp_dir)):
        if file.endswith('.mat'):
            times, values = load_mat_file(os.path.join(lfp_dir, file))
            if all_times is None:
                all_times = times
            all_channels.append(values)

    all_channels = np.stack(all_channels)  # shape: [num_channels, num_samples]

    print(f"\n[DEBUG] Loaded {len(all_channels)} channels, shape={all_channels.shape}")
    print(f"[DEBUG] Global LFP times range: {all_times.min()} to {all_times.max()} seconds")
    print(f"[DEBUG] Total LFP timepoints: {len(all_times)}")

    # --- שלב 5: חיתוך האות לפי התוויות ---
    data_struct = []
    for idx, row in labels_df.iterrows():
        start, end, label = row['start'], row['end'], row['label']
        indices = np.where((all_times >= start) & (all_times <= end))[0]
        if len(indices) > 0:
            segment = all_channels[:, indices]  # shape: [num_channels, num_samples]
            data_struct.append({"signals": segment, "label": label})
        else:
            print(f"Warning: No signal found for label '{label}' ({start}-{end})")

    # --- שלב 6: שמירת התוצאה ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_data.npy"), data_struct)
    print(f"\n✅ Saved processed data for {patient_name}. Total segments: {len(data_struct)}")


# =============================================================
# עיבוד כל החולים בקובץ קונפיג
# =============================================================

def process_all_patients():
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    for patient_name, patient_info in config['patients'].items():
        process_patient(patient_name, patient_info)
