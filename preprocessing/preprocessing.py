import os
import json
import numpy as np
import pandas as pd
import h5py
from defines import BASE_DATA_PATH, PATIENTS_CONFIG_PATH, PROCESSED_DATA_DIR

def load_offset(mat_path):
    with h5py.File(mat_path, 'r') as f:
        offset_array = f['new_start_end_times_micsec'][:]
        start_micro = offset_array[0][0]
        offset_sec = start_micro / 1e6
        # [DEBUG] הדפסה לצורך איתור תזוזה בין אודיו ל־LFP
        # print(f"[DEBUG] Offset (from audio→LFP): {offset_sec:.6f} s")
        return offset_sec

def load_labels(label_path):
    df = pd.read_csv(label_path, delimiter="\t", names=["start", "end", "label"])
    # [DEBUG] הדפסה למספר תוויות שנטענו
    # print(f"[DEBUG] Loaded {len(df)} labels")
    return df

def export_lfp_csvs(lfp_folder, out_dir):
    for fname in sorted(os.listdir(lfp_folder)):
        if not fname.endswith(".mat"):
            continue
        path = os.path.join(lfp_folder, fname)
        with h5py.File(path, 'r') as f:
            signal = np.array(f["#refs#/g"]).flatten()
            sr = float(f["#refs#/f/rowTimes"]["sampleRate"][()])
            times = np.arange(len(signal)) / sr
            df = pd.DataFrame({"time": times, "signal": signal})
            out_path = os.path.join(out_dir, fname.replace(".mat", ".csv"))

            # [DEBUG] בדיקות מבנה הדאטה
            # print(f"[DEBUG] Attempting to export {fname} to {out_path}")
            # print(f"[DEBUG] DataFrame shape: {df.shape}, types:\n{df.dtypes}")
            # print(f"[DEBUG] First 5 rows:\n{df.head()}")

            df.to_csv(out_path, index=False)
            print(f"[EXPORT] {fname}: {len(signal)} samples → {times[-1]:.3f}s → {out_path}")

def process_patient(patient_name, patient_info):
    print(f"===== Processing {patient_name} =====")

    label_path = os.path.join(BASE_DATA_PATH, patient_info["labels_file"])
    offset_path = os.path.join(BASE_DATA_PATH, patient_info["offset_file"])
    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info["lfp_folder"])
    csv_out_dir = os.path.join(PROCESSED_DATA_DIR, "csvs")
    os.makedirs(csv_out_dir, exist_ok=True)

    offset = load_offset(offset_path)

    df_labels = load_labels(label_path)
    df_labels["start_adj"] = df_labels["start"] + offset
    df_labels["end_adj"] = df_labels["end"] + offset

    # [DEBUG] בדיקת תוויות מתוזמנות
    # print(f"[DEBUG] Labels with positive adjusted times: {(df_labels['start_adj'] >= 0).sum()}")
    # print(f"[DEBUG] Labels with negative adjusted times: {(df_labels['start_adj'] < 0).sum()}")
    # print("[DEBUG] First 10 adjusted labels:")
    # print(df_labels[["start_adj", "end_adj", "label"]].head(10))

    export_lfp_csvs(lfp_dir, csv_out_dir)

def process_all_patients():
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    for name, info in config["patients"].items():
        process_patient(name, info)
