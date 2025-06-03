import os
import h5py
import pandas as pd
import numpy as np

# נתיבים לקבצים
BASE_DATA_PATH = r'G:/My Drive/FinalProject/Data/'
LABELS_FILE = os.path.join(BASE_DATA_PATH, "Patient_01/Labels.txt")
OFFSET_FILE = os.path.join(BASE_DATA_PATH, "Patient_01/sound_w_times.mat")
LFP_DIR = os.path.join(BASE_DATA_PATH, "Patient_01/ZIP_files_and_backup_data_Patient_01/pt2_LFPs_sound/LFPs")
CSV_OUT_DIR = os.path.join("processed_data", "csvs")
os.makedirs(CSV_OUT_DIR, exist_ok=True)

def load_offset(offset_path):
    """
    Load offset (start time of audio relative to LFP).
    """
    with h5py.File(offset_path, 'r') as f:
        offset_array = f['new_start_end_times_micsec'][:]
        start_micro = offset_array[0][0]
        offset_sec = start_micro / 1e6
        print(f"[DEBUG] Offset (from audio→LFP): {offset_sec:.6f} s")
        return offset_sec

def load_labels(label_path, offset):
    """
    Load labels file and apply offset to times.
    """
    df = pd.read_csv(label_path, delimiter='\t', header=None, names=['start', 'end', 'label'])
    df['start_adj'] = df['start'] + offset
    df['end_adj'] = df['end'] + offset
    print(f"[DEBUG] Loaded {len(df)} labels")
    print(f"[DEBUG] Labels with positive adjusted times: {(df['start_adj'] >= 0).sum()}")
    print(f"[DEBUG] Labels with negative adjusted times: {(df['start_adj'] < 0).sum()}")
    print("[DEBUG] First 10 adjusted labels:")
    print(df[['start_adj', 'end_adj', 'label']].head(10))
    return df

def export_lfp_csvs(lfp_folder, out_dir):
    """
    Export LFP signals to CSVs with time (in seconds) and signal.
    """
    for fname in sorted(os.listdir(lfp_folder)):
        if not fname.endswith(".mat"):
            continue
        path = os.path.join(lfp_folder, fname)
        with h5py.File(path, 'r') as f:
            signal = np.array(f["#refs#/g"]).flatten()
            sr = float(f["#refs#/f/rowTimes"]["sampleRate"][()].item())
            times = np.arange(len(signal)) / sr
            df = pd.DataFrame({"time": times, "signal": signal})
            out_path = os.path.join(out_dir, fname.replace(".mat", ".csv"))
            df.to_csv(out_path, index=False)
            print(f"[EXPORT] {fname}: {len(signal)} samples → 0→{times[-1]:.3f}s → {out_path}")

def main():
    print("===== Processing Patient_01 =====")

    # שלב 1: טעינת offset
    offset = load_offset(OFFSET_FILE)

    # שלב 2: טעינת תוויות + החלת offset
    df_labels = load_labels(LABELS_FILE, offset)

    # שלב 3: ייצוא אותות LFP ל-CSV (כבוי כברירת מחדל)
    export_lfp_csvs(LFP_DIR, CSV_OUT_DIR)

if __name__ == "__main__":
    main()
