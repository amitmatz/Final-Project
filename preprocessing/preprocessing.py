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
    Load Time and LFP values from a MATLAB v7.3 file (LFP_tt structure) using h5py.
    """
    with h5py.File(filepath, 'r') as f:
        lfp_tt = f['LFP_tt']

        refs = f['#refs#']

        # ניגש לשדות הרלוונטיים
        try:
            origin = refs['f']['rowTimes']['origin'][()]
            step_size = refs['f']['rowTimes']['stepSize'][()]
            num_samples = refs['f']['numRows'][()]
        except KeyError as e:
            raise Exception(f"Couldn't find expected fields inside LFP_tt: {e}")

        # ננקה את הנתונים כדי לקבל ערכים בודדים
        origin = np.array(origin).squeeze()
        if origin.size > 1:
            origin = origin[0]

        step_size = np.array(step_size).squeeze()
        if step_size.size > 1:
            step_size = step_size[0]

        num_samples = np.array(num_samples).squeeze()
        if num_samples.size > 1:
            num_samples = num_samples[0]

        # בונים את מערך הזמנים
        times = np.arange(0, num_samples) * step_size + origin

        # טוענים את האותות עצמם מתוך LFP_tt
        values = np.array(lfp_tt).squeeze()

    return times, values

# def load_mat_file(filepath): #todo amit: delete if unused
#     """
#     Load LFP data from a .mat file saved in MATLAB v7.3 format using h5py.
#     """
#     with h5py.File(filepath, 'r') as f:
#         # חשוב לבדוק איך בדיוק בנויים השדות בקובץ!
#         timestamps = np.array(f['Time']).squeeze()
#         values = np.array(f['LFP']).squeeze()
#     return timestamps, values

def load_labels(filepath):
    return pd.read_csv(filepath, delimiter='\t', header=None, names=['start', 'end', 'label'])


def process_patient(patient_name, patient_info):
    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info['lfp_folder'])
    labels_path = os.path.join(BASE_DATA_PATH, patient_info['labels_file'])

    print(f"Processing {patient_name}...")
    print(f"Looking for LFP signals at: {lfp_dir}") # todo amit: delete prints
    print(f"Exists? {os.path.exists(lfp_dir)}")     # todo amit: delete prints

    # Load all LFP files
    all_times, all_values = [], []

    for file in sorted(os.listdir(lfp_dir)):
        if file.endswith('.mat'):
            times, values = load_mat_file(os.path.join(lfp_dir, file))
            all_times.append(times)
            all_values.append(values)
    all_times = np.concatenate(all_times)
    all_values = np.concatenate(all_values)

    labels_df = load_labels(labels_path)

    data_struct = []
    for idx, row in labels_df.iterrows():
        start, end, label = row['start'], row['end'], row['label']
        indices = np.where((all_times >= start) & (all_times <= end))[0]
        if len(indices) > 0:
            segment = all_values[indices]
            data_struct.append({"signal": segment, "label": label})
        else:
            print(f"Warning: No signal found for label {label} ({start}-{end})")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_data.npy"), data_struct)
    print(f"Saved processed data for {patient_name}.")


def process_all_patients():
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    for patient_name, patient_info in config['patients'].items():
        process_patient(patient_name, patient_info)
