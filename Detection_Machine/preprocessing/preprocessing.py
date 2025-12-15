import os
import json
import numpy as np
import pandas as pd
import h5py

from defines import BASE_DATA_PATH, PATIENTS_CONFIG_PATH, PROCESSED_DATA_DIR

# Debugging and feature toggles
SKIP_LFP_EXPORT = False        # If True, skip exporting LFP to CSV (assume already done)
SKIP_DETECTION_EXPORT = False  # If True, skip creating detection windows (assume already done)


def load_offset(mat_path):
    with h5py.File(mat_path, 'r') as f:
        offset_array = f['new_start_end_times_micsec'][:]
        start_micro = offset_array[0][0]
        offset_sec = start_micro / 1e6
        return offset_sec


def load_labels(label_path):
    df = pd.read_csv(label_path, delimiter="\t", names=["start", "end", "label"])
    return df


def export_lfp_csvs(lfp_folder, out_dir):
    os.makedirs(out_dir, exist_ok=True)
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
        df.to_csv(out_path, index=False)
        print(f"[EXPORT] {fname}: {len(signal)} samples @ {sr:.2f} Hz → {times[-1]:.3f}s → {out_path}")


def export_detection_windows(lfp_dir, df_labels, out_path, fixed_length=300):
    """
    NOTE: detection windows are still saved at PROCESSED_DATA_DIR/<patient>_detection_data.npy
    (not inside the per-patient subdir) to stay compatible with the classification code.
    """
    print(f"[INFO] Creating labeled detection windows from CSV files...")
    all_segments = []
    skipped_segments = 0

    for fname in sorted(os.listdir(lfp_dir)):
        if not fname.endswith(".csv"):
            continue
        ch_name = fname.replace(".csv", "")
        path = os.path.join(lfp_dir, fname)
        df = pd.read_csv(path)

        for _, row in df_labels.iterrows():
            label_str = row["label"]
            start, end = row["start_adj"], row["end_adj"]
            is_speech = label_str.lower() != "none"

            seg = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
            signal = seg["signal"].values
            if len(signal) < fixed_length:
                skipped_segments += 1
                continue

            for i in range(0, len(signal) - fixed_length + 1, fixed_length):
                chunk = signal[i:i + fixed_length]
                all_segments.append({
                    "channel": ch_name,
                    "signals": chunk,
                    "label": label_str,
                    "is_speech": is_speech
                })

    np.save(out_path, all_segments)
    print(f"[INFO] Saved {len(all_segments)} segments, skipped {skipped_segments} (too short).")
    print(f"✅ Detection dataset saved to: {out_path}")


def export_labels_to_csv(df_labels, patient_name: str):
    """
    Save labels_aligned.csv under a per-patient directory:
        PROCESSED_DATA_DIR/<patient_name>/labels_aligned.csv
    """
    patient_dir = os.path.join(PROCESSED_DATA_DIR, patient_name)
    os.makedirs(patient_dir, exist_ok=True)
    out_path = os.path.join(patient_dir, "labels_aligned.csv")
    df_labels.to_csv(out_path, index=False)
    print(f"[EXPORT] Labels saved to: {out_path}")


def process_patient(patient_name, patient_info):
    print(f"===== Processing {patient_name} =====")

    label_path = os.path.join(BASE_DATA_PATH, patient_info["labels_file"])
    offset_path = os.path.join(BASE_DATA_PATH, patient_info["offset_file"])
    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info["lfp_folder"])

    # Per-patient processed directory:
    #   PROCESSED_DATA_DIR/<patient_name>/csvs
    patient_dir = os.path.join(PROCESSED_DATA_DIR, patient_name)
    csv_out_dir = os.path.join(patient_dir, "csvs")
    os.makedirs(csv_out_dir, exist_ok=True)

    # Load labels and align to absolute time using offset
    offset = load_offset(offset_path)
    df_labels = load_labels(label_path)
    df_labels["start_adj"] = df_labels["start"] + offset
    df_labels["end_adj"] = df_labels["end"] + offset

    # Optionally merge NoSpeech segments from a detection labels file, if exists
    detection_file = os.path.join(
        BASE_DATA_PATH,
        patient_name,
        "Labels",
        "Detection_Labels",
        f"{patient_name.replace('Patient_', 'Patient')}_Labels_Detection 1.txt"
    )
    if os.path.exists(detection_file):
        df_det = load_labels(detection_file)
        df_det["start_adj"] = df_det["start"] + offset
        df_det["end_adj"] = df_det["end"] + offset
        df_none = df_det[df_det["label"].str.lower() == "nospeech"].copy()
        df_none["label"] = "None"
        df_labels = pd.concat([df_labels, df_none], ignore_index=True)
        df_labels = df_labels.sort_values("start_adj").reset_index(drop=True)
        print(f"[INFO] Merged detection labels: added {len(df_none)} 'None' segments")

    # Export LFP CSVs (per patient)
    if not SKIP_LFP_EXPORT:
        export_lfp_csvs(lfp_dir, csv_out_dir)
    else:
        print("[INFO] SKIP_LFP_EXPORT=True: skipping LFP export to CSV")

    # Export labels CSV (per patient)
    export_labels_to_csv(df_labels, patient_name)

    # Export detection windows (path kept compatible with classification step)
    if not SKIP_DETECTION_EXPORT:
        # Note: stays at PROCESSED_DATA_DIR, not per-patient subdir, to avoid
        # breaking existing classification code that looks there.
        detection_out_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_detection_data.npy")
        export_detection_windows(csv_out_dir, df_labels, detection_out_path)
    else:
        print("[INFO] SKIP_DETECTION_EXPORT=True: skipping detection window creation")


def process_all_patients(patient_id: str | None = None):
    """
    If patient_id is None: process all patients from the config.
    Otherwise: process only the given patient_id.
    """
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    patients = config.get("patients", {})

    if patient_id is None:
        for name, info in patients.items():
            process_patient(name, info)
    else:
        if patient_id not in patients:
            raise KeyError(
                f"Patient '{patient_id}' not found in config file: {PATIENTS_CONFIG_PATH}"
            )
        process_patient(patient_id, patients[patient_id])

# if __name__ == '__main__':
#     process_all_patients()
