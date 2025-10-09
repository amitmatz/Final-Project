# preprocessing/preprocessing.py
import os
import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Optional, List, Dict, Any
from defines import BASE_DATA_PATH, PATIENTS_CONFIG_PATH, PROCESSED_DATA_DIR

# ============================
# Debugging and feature toggles
# ============================
SKIP_LFP_EXPORT = False        # If True: skip LFP→CSV export (assumes CSVs already exist)
SKIP_CLASSIFICATION_EXPORT = False  # If True: skip classification NPY export

# Make sure processed dir exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ---------- I/O helpers ----------

def load_offset(mat_path: Path) -> float:
    """
    Load offset (seconds) from a MAT file.
    Expects dataset 'new_start_end_times_micsec' with first element = start time in microseconds.
    """
    with h5py.File(str(mat_path), 'r') as f:
        offset_array = f['new_start_end_times_micsec'][:]
        start_micro = offset_array[0][0]
        offset_sec = start_micro / 1e6
        return float(offset_sec)

def load_labels(label_path: Path) -> pd.DataFrame:
    """
    Load labels text file with 3 tab-separated columns: start, end, label.
    """
    df = pd.read_csv(str(label_path), delimiter="\t", names=["start", "end", "label"], encoding="utf-8")
    return df

def export_lfp_csvs(lfp_folder: Path, out_dir: Path) -> None:
    """
    Export all .mat LFP channels to per-channel CSV files with columns: time, signal.
    Assumes MAT structure contains '#refs#/g' (signal) and '#refs#/f/rowTimes/sampleRate' (Hz).
    """
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(lfp_folder)):
        if not fname.endswith(".mat"):
            continue
        path = lfp_folder / fname
        with h5py.File(str(path), 'r') as f:
            signal = np.array(f["#refs#/g"]).flatten()
            sr = float(f["#refs#/f/rowTimes"]["sampleRate"][()])
            times = np.arange(len(signal)) / sr
        df = pd.DataFrame({"time": times, "signal": signal})
        out_path = out_dir / fname.replace(".mat", ".csv")
        df.to_csv(out_path, index=False)
        print(f"[EXPORT] {fname}: {len(signal)} samples @ {sr:.2f} Hz → {times[-1]:.3f}s → {out_path}")

# ---------- Label normalization ----------

def normalize_label(s: str) -> str:
    """
    Return one of: 'HAARYE' / 'TUT' / 'OTHER'.
    Any label other than exact 'HAARYE' or 'TUT' becomes 'OTHER'.
    """
    up = (s or "").strip().upper()
    if up == "HAARYE":
        return "HAARYE"
    if up == "TUT":
        return "TUT"
    return "OTHER"

# ---------- Multi-channel helpers ----------

def _load_all_channel_csvs(lfp_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all channel CSVs from a directory into a dict: {channel_name: DataFrame[time, signal]}.
    Assumes all channels share identical length and time grid.
    """
    ch2df: Dict[str, pd.DataFrame] = {}
    for fname in sorted(os.listdir(lfp_dir)):
        if fname.endswith(".csv"):
            ch = fname[:-4]
            df = pd.read_csv(lfp_dir / fname)
            if "time" not in df.columns or "signal" not in df.columns:
                raise ValueError(f"{fname} must contain 'time' and 'signal' columns")
            ch2df[ch] = df
    if not ch2df:
        raise RuntimeError(f"No CSV files found in {lfp_dir}")
    L = len(next(iter(ch2df.values())))
    if not all(len(df) == L for df in ch2df.values()):
        raise ValueError("All channels must share the same length/time grid")
    return ch2df

def _slice_window_multich(ch2df: Dict[str, pd.DataFrame],
                          t_center: float,
                          win_samples: int,
                          sr_float: float) -> Optional[np.ndarray]:
    """
    Return a window [C, T] centered at t_center; if out-of-bounds, return None.
    """
    idx_center = int(round(t_center * sr_float))
    start = idx_center - win_samples // 2
    end = start + win_samples
    L = len(next(iter(ch2df.values())))
    if start < 0 or end > L:
        return None
    mats: List[np.ndarray] = []
    for df in ch2df.values():
        sig = df["signal"].values
        mats.append(sig[start:end][None, :])  # [1, T]
    return np.concatenate(mats, axis=0)  # [C, T]

# ---------- Robust offset resolution ----------

def _resolve_offset(patient_info: Dict[str, Any], lfp_dir: Path) -> float:
    """
    Resolve the recording start offset (seconds) with the following priority:
    1) If 'offset_seconds' is provided in JSON, use it directly.
    2) Else, try 'offset_file' (relative to BASE_DATA_PATH).
    3) Else, try common fallback filenames inside lfp_dir.
    Raise FileNotFoundError with a helpful message if nothing is found.
    """
    # 1) direct numeric offset in JSON
    if "offset_seconds" in patient_info:
        try:
            return float(patient_info["offset_seconds"])
        except Exception as e:
            raise ValueError(f"Invalid 'offset_seconds' value: {patient_info['offset_seconds']} ({e})")

    candidates: List[Path] = []

    # 2) explicit offset_file in JSON
    off_rel = patient_info.get("offset_file")
    if off_rel:
        candidates.append(BASE_DATA_PATH / off_rel)

    # 3) fallbacks under lfp_dir
    candidates.extend([
        lfp_dir / "sound_w_times.mat",
        lfp_dir / "sound_with_times.mat",
        lfp_dir / "sound_w_times_microsec.mat",
    ])

    for p in candidates:
        if p.exists():
            return load_offset(p)

    paths_list = " | ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Offset file not found. Looked for: "
        + paths_list
        + ". Fix 'offset_file' in the JSON or add 'offset_seconds'."
    )

# ---------- Create OTHER from gaps ----------

def export_classification_windows(
    lfp_dir: Path,
    df_labels: pd.DataFrame,
    out_path: Path,
    sample_rate_hz: float,
    window_ms: int,
    other_stride_ms: int = 500,
    other_margin_ms: int = 250,
    other_ratio: float = 1.0,
) -> None:
    """
    Save an NPY file with dict items:
      - 'signals': np.ndarray[C, T]
      - 'label'  : 'HAARYE' / 'TUT' / 'OTHER'
    """
    ch2df = _load_all_channel_csvs(lfp_dir)
    sr = float(sample_rate_hz)
    win_samples = int(round(sr * (window_ms / 1000.0)))
    stride_other = int(round(sr * (other_stride_ms / 1000.0)))
    margin = int(round(sr * (other_margin_ms / 1000.0)))

    # Common time vector (shared across channels)
    time_vec = next(iter(ch2df.values()))["time"].values
    L = len(time_vec)

    items: List[Dict[str, Any]] = []

    # Positive samples: HAARYE/TUT
    pos_mask = np.zeros(L, dtype=bool)  # marks occupied regions (including margins)
    pos_rows = df_labels[df_labels["label"].isin(["HAARYE", "TUT"])]

    for _, row in pos_rows.iterrows():
        t_center = 0.5 * (float(row["start_adj"]) + float(row["end_adj"]))
        x = _slice_window_multich(ch2df, t_center, win_samples, sr)
        if x is not None:
            items.append({"signals": x.astype(np.float32), "label": str(row["label"])})
        # Update occupied mask with margins
        i0 = max(0, int(np.floor(float(row["start_adj"]) * sr)) - margin)
        i1 = min(L, int(np.ceil (float(row["end_adj"])   * sr)) + margin)
        if i1 > i0:
            pos_mask[i0:i1] = True

    num_pos = len(items)

    # OTHER windows from free regions
    free = ~pos_mask
    edges = np.diff(np.concatenate(([0], free.astype(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]

    other_centers: List[int] = []
    for s, e in zip(starts, ends):
        run_len = e - s
        if run_len < win_samples:
            continue
        c_start = s + win_samples // 2
        c_end   = e - win_samples // 2
        for c in range(c_start, c_end + 1, stride_other):
            other_centers.append(c)

    # Limit OTHER count by ratio
    target_other = int(max(1, other_ratio * num_pos)) if num_pos > 0 else 0
    if target_other > 0 and len(other_centers) > target_other:
        idx = np.linspace(0, len(other_centers) - 1, target_other).astype(int)
        other_centers = [other_centers[i] for i in idx]

    # Build items
    for c in other_centers:
        t_center = c / sr
        x = _slice_window_multich(ch2df, t_center, win_samples, sr)
        if x is not None:
            items.append({"signals": x.astype(np.float32), "label": "OTHER"})

    np.save(out_path, items)
    print(f"[INFO] CLASSIFICATION: saved {len(items)} windows "
          f"(pos={num_pos}, other={len(items)-num_pos}), win={win_samples} samples.")
    print(f"✅ Saved to: {out_path}")

# ---------- Orchestration ----------

def export_labels_to_csv(df_labels: pd.DataFrame) -> None:
    out_path = PROCESSED_DATA_DIR / "labels_aligned.csv"
    df_labels.to_csv(out_path, index=False)
    print(f"[EXPORT] Labels saved to: {out_path}")

def process_patient(patient_name: str, patient_info: Dict[str, Any]) -> None:
    print(f"===== Processing {patient_name} =====")

    label_path  = BASE_DATA_PATH / patient_info["labels_file"]
    lfp_dir     = BASE_DATA_PATH / patient_info["lfp_folder"]

    # Pre-flight checks with informative errors
    if not label_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {label_path}. "
            f"Fix 'labels_file' in the JSON (relative to BASE_DATA_PATH)."
        )
    if not lfp_dir.exists():
        raise FileNotFoundError(
            f"LFP folder not found: {lfp_dir}. "
            f"Fix 'lfp_folder' in the JSON (relative to BASE_DATA_PATH)."
        )

    csv_out_dir = PROCESSED_DATA_DIR / "csvs"
    os.makedirs(csv_out_dir, exist_ok=True)

    # 1) Resolve offset and load labels
    offset = _resolve_offset(patient_info, lfp_dir)
    df_labels = load_labels(label_path)
    df_labels["start_adj"] = df_labels["start"] + offset
    df_labels["end_adj"]   = df_labels["end"]   + offset

    # 2) Normalize labels to HAARYE / TUT / OTHER
    df_labels["label"] = df_labels["label"].astype(str).apply(normalize_label)
    df_labels = df_labels.sort_values("start_adj").reset_index(drop=True)

    # 3) Export LFP channel CSVs (optional)
    if not SKIP_LFP_EXPORT:
        export_lfp_csvs(lfp_dir, csv_out_dir)
    else:
        print('[INFO] SKIP_LFP_EXPORT=True: skipping LFP→CSV export')
    export_labels_to_csv(df_labels)

    # 4) Export multi-channel classification windows + OTHER from gaps (optional)
    if not SKIP_CLASSIFICATION_EXPORT:
        classification_out_path = PROCESSED_DATA_DIR / f"{patient_name}_classification_data.npy"

        # Optional params from config (supports 'other_ratio' and 'target_other_pos_ratio')
        other_stride_ms = int(patient_info.get("other_stride_ms", 500))
        other_margin_ms = int(patient_info.get("other_margin_ms", 250))
        other_ratio     = float(patient_info.get("other_ratio",
                                  patient_info.get("target_other_pos_ratio", 1.0)))

        export_classification_windows(
            lfp_dir=csv_out_dir,
            df_labels=df_labels,
            out_path=classification_out_path,
            sample_rate_hz=patient_info.get("sample_rate", 2000),
            window_ms=patient_info.get("window_size", 500),
            other_stride_ms=other_stride_ms,
            other_margin_ms=other_margin_ms,
            other_ratio=other_ratio,
        )
    else:
        print('[INFO] SKIP_CLASSIFICATION_EXPORT=True: skipping classification window export')

def process_all_patients() -> None:
    with open(PATIENTS_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    for name, info in config["patients"].items():
        process_patient(name, info)

# if __name__ == '__main__':
#     process_all_patients()
