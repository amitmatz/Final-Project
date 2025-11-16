# -*- coding: utf-8 -*-
# preprocessing/preprocessing.py
# End-to-end preprocessing for LFP classification (paths via defines.py):
# - Robust path resolution using defines.DATA_BASES_FOR_SEARCH
# - v7.3 MAT reading via h5py (and CSV export per channel)
# - Per-patient CSV folder to avoid cross-patient mix-ups
# - Flexible label loader (supports "start end label" and "trial_id label onset")
# - Trial-based 5/5/5 VAL/TEST split (by label), indices saved to *_splits.json
# - Output: list-of-dicts {signals [T,C], label (str), trial_id}

import os
import json
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import h5py

# ---------------------------------------------------------------------
# Ensure project root (where defines.py lives) is on sys.path
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # one level up from 'preprocessing' folder
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defines import (
    DATA_BASES_FOR_SEARCH,
    PATIENTS_CONFIG_PATH,
    PROCESSED_DATA_DIR,
)

# ---------------- Logging ----------------
def LOG_INFO(msg: str):  print(f"[INFO] {msg}")
def LOG_DEBUG(msg: str): print(f"[DEBUG] {msg}")
def LOG_WARN(msg: str):  print(f"[WARN] {msg}")
def LOG_ERR(msg: str):   print(f"[ERROR] {msg}")

def _norm(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)

# ---------------- Labels map ----------------
def _norm_label(s: str) -> str:
    up = (s or "").strip().upper()
    if up.startswith("HAARY"): return "HAARYE"
    if up in {"TUT", "TUT.", "TUT,"}: return "TUT"
    if up in {"האריה"}: return "HAARYE"
    if up in {"תות"}:   return "TUT"
    return "OTHER"

# ---------------- Config ----------------
@dataclass
class PreprocConfig:
    sample_rate: float = 2000.0
    window_size_ms: int = 500
    other_stride_ms: int = 500
    other_margin_ms: int = 250
    other_ratio: float = 1.0
    per_class_test: int = 5
    per_class_val: int = 5
    seed: int = 42
    # CSV export
    do_export_csv: bool = True
    csv_root_dirname: str = "csvs"  # under PROCESSED_DATA_DIR

# ---------------- Helpers: labels ----------------
def _is_float(x: str) -> bool:
    try:
        float(x); return True
    except Exception:
        return False

def load_labels_file(labels_path: Path) -> List[Dict[str, Any]]:
    """
    Supports BOTH:
      A) start<TAB>end<TAB>label
      B) trial_id<TAB>label<TAB>onset_sec
    Returns: list of {trial_id, label, onset_sec}
    """
    events: List[Dict[str, Any]] = []
    if not labels_path.exists():
        LOG_WARN("No labels file found; falling back to OTHER-only windows.")
        return events

    with open(labels_path, "r", encoding="utf-8") as f:
        line_idx = 0
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.replace(",", "\t").split("\t") if p.strip()]
            if len(parts) < 3:
                continue

            # A) start, end, label
            if _is_float(parts[0]) and _is_float(parts[1]) and not _is_float(parts[2]):
                start_sec = float(parts[0])
                end_sec = float(parts[1])
                label = _norm_label(parts[2])
                onset_sec = 0.5 * (start_sec + end_sec)
                events.append(
                    {"trial_id": f"trial_{line_idx}", "label": label, "onset_sec": onset_sec}
                )
                line_idx += 1
                continue

            # B) trial_id, label, onset_sec
            if (not _is_float(parts[0])) and _is_float(parts[2]):
                trial_id = parts[0]
                label = _norm_label(parts[1])
                onset_sec = float(parts[2])
                events.append({"trial_id": trial_id, "label": label, "onset_sec": onset_sec})
                line_idx += 1
                continue

            # Fallback: if first two look numeric, treat as start/end
            if _is_float(parts[0]) and _is_float(parts[1]):
                start_sec = float(parts[0])
                end_sec = float(parts[1])
                label = _norm_label(parts[2])
                onset_sec = 0.5 * (start_sec + end_sec)
                events.append(
                    {"trial_id": f"trial_{line_idx}", "label": label, "onset_sec": onset_sec}
                )
                line_idx += 1
                continue

            LOG_WARN(f"Skipped label line (unrecognized format): {s}")

    LOG_INFO(f"Loaded {len(events)} label events")
    return events

# ---------------- Helpers: path resolution ----------------
def _search_under_bases(rel: Path) -> Optional[Path]:
    LOG_INFO("Resolving data paths (searching DATA_BASES_FOR_SEARCH):")
    for base in DATA_BASES_FOR_SEARCH:
        LOG_DEBUG(f"  Base: {base}")
        cand = (base / rel)
        ok = cand.exists()
        LOG_DEBUG(f"    -> {_norm(cand)} [{'OK' if ok else 'missing'}]")
        if ok:
            return cand
    return None

def resolve_patient_paths(patient_id: str, patients_config_path: Path) -> Tuple[Path, Path, Optional[Path]]:
    with open(patients_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    patients = cfg.get("patients", {})
    if patient_id not in patients:
        raise KeyError(f"Patient '{patient_id}' not in patients_config.json")

    pinfo = patients[patient_id]
    labels_rel = Path(pinfo["labels_file"])
    lfp_rel    = Path(pinfo["lfp_folder"])
    offset_rel = Path(pinfo.get("offset_file", "")) if pinfo.get("offset_file") else None

    labels = _search_under_bases(labels_rel)
    lfp    = _search_under_bases(lfp_rel)
    offset = _search_under_bases(offset_rel) if offset_rel else None

    if labels is None:
        raise FileNotFoundError(
            "Labels file not found.\n"
            f"  -> Expected relative: {labels_rel}\n"
            "Fix 'labels_file' in patients_config.json."
        )
    if lfp is None or not lfp.is_dir():
        raise FileNotFoundError(
            "LFP folder not found.\n"
            f"  -> Expected relative: {lfp_rel}\n"
            "Fix 'lfp_folder' in patients_config.json."
        )

    LOG_DEBUG(f"Resolved labels: {_norm(labels)}")
    LOG_DEBUG(f"Resolved LFP dir: {_norm(lfp)}")
    if offset:
        LOG_DEBUG(f"Resolved offset: {_norm(offset)}")
    return lfp, labels, offset

# ---------------- LFP readers / exporters ----------------
def _clean_dir(dir_path: Path, pattern: str = "*.csv") -> None:
    """Remove existing files (e.g., from previous runs) to avoid cross-patient mixing."""
    if dir_path.exists():
        for p in dir_path.glob(pattern):
            try:
                p.unlink()
            except Exception:
                pass
    else:
        dir_path.mkdir(parents=True, exist_ok=True)

def export_lfp_csvs(lfp_folder: Path, out_dir: Path) -> None:
    """
    Export v7.3 MAT channels to per-channel CSVs with columns: time, signal.
    MAT structure:
      - signal: '#refs#/g'
      - sample rate (Hz): '#refs#/f/rowTimes/sampleRate'
    """
    _clean_dir(out_dir, "*.csv")
    mat_files = sorted([p for p in lfp_folder.iterdir() if p.suffix.lower() == ".mat"])
    for fname in mat_files:
        with h5py.File(str(fname), 'r') as f:
            if "#refs#/g" not in f:
                raise RuntimeError(f"{fname.name}: '#refs#/g' not found (unexpected MAT structure)")
            signal = np.array(f["#refs#/g"]).astype(np.float64).flatten()
            sr_ds = f.get("#refs#/f/rowTimes/sampleRate")
            if sr_ds is None:
                raise RuntimeError(f"{fname.name}: sampleRate not found at '#refs#/f/rowTimes/sampleRate'")
            sr = float(sr_ds[()])
            times = np.arange(len(signal)) / sr
            df = pd.DataFrame({"time": times, "signal": signal})
            out_path = out_dir / fname.name.replace(".mat", ".csv")
            df.to_csv(out_path, index=False)
            dur = times[-1] if len(times) > 0 else 0.0
            print(f"[EXPORT] {fname.name}: {len(signal)} samples @ {sr:.2f} Hz → {dur:.3f}s → {out_path}")

def _load_all_channel_csvs(lfp_csv_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSVs in a directory and enforce a common length/time grid:
      - Trim all to min length (Lmin)
      - Use first channel's time as master, overwrite others if needed
    """
    csv_files = sorted([p for p in lfp_csv_dir.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {lfp_csv_dir}")

    ch2df: Dict[str, pd.DataFrame] = {}
    lengths: List[int] = []
    for p in csv_files:
        df = pd.read_csv(p)
        if "time" not in df.columns or "signal" not in df.columns:
            raise ValueError(f"{p.name} must contain 'time' and 'signal' columns")
        df = df[["time", "signal"]].copy()
        ch2df[p.stem] = df
        lengths.append(len(df))

    Lmin = int(min(lengths))
    if len(set(lengths)) > 1:
        LOG_WARN(f"Channel CSVs have different lengths; trimming all to Lmin={Lmin} samples.")

    first_ch = sorted(ch2df.keys())[0]
    master_time = ch2df[first_ch]["time"].values[:Lmin]
    tol = 1e-6

    for ch, df in ch2df.items():
        if len(df) != Lmin:
            df = df.iloc[:Lmin].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        t = df["time"].values
        if len(t) != Lmin or np.max(np.abs(t - master_time[:len(t)])) > tol:
            df["time"] = master_time
        ch2df[ch] = df

    return ch2df

# ---------------- Offset resolution ----------------
def _resolve_offset_seconds(offset_file: Optional[Path]) -> float:
    if not offset_file or not offset_file.exists():
        raise FileNotFoundError("Offset file not found (provide 'offset_file' in patients_config.json).")
    with h5py.File(str(offset_file), 'r') as f:
        if "new_start_end_times_micsec" not in f:
            raise RuntimeError("Offset MAT missing 'new_start_end_times_micsec'")
        arr = np.array(f["new_start_end_times_micsec"])
        start_micro = float(arr.flat[0])
        return start_micro / 1e6

# ---------------- Window extraction ----------------
def _slice_window_multich(ch2df: Dict[str, pd.DataFrame],
                          t_center: float,
                          win_samples: int,
                          sr_float: float) -> Optional[np.ndarray]:
    idx_center = int(round(t_center * sr_float))
    start = idx_center - win_samples // 2
    end = start + win_samples
    L = len(next(iter(ch2df.values())))
    if start < 0 or end > L:
        return None
    mats: List[np.ndarray] = []
    # keep deterministic channel order
    for ch in sorted(ch2df.keys()):
        sig = ch2df[ch]["signal"].values
        mats.append(sig[start:end][:, None])  # [T,1]
    return np.concatenate(mats, axis=1)  # [T,C]

# ---------------- Trial split ----------------
def stratified_trials_split(events: List[Dict[str, Any]],
                            per_class_test: int,
                            per_class_val: int,
                            seed: int) -> Tuple[List[str], List[str], List[str]]:
    by_label: Dict[str, List[str]] = {"HAARYE": [], "TUT": [], "OTHER": []}
    for ev in events:
        lbl = ev["label"]
        if lbl in by_label:
            by_label[lbl].append(ev["trial_id"])
    rng = np.random.default_rng(seed)
    for k in by_label:
        rng.shuffle(by_label[k])

    test_ids, val_ids, train_ids = [], [], []
    for _, ids in by_label.items():
        t_take = min(per_class_test, len(ids))
        v_take = min(per_class_val, max(0, len(ids) - t_take))
        test_ids.extend(ids[:t_take])
        start = t_take
        end = t_take + v_take
        val_ids.extend(ids[start:end])
        train_ids.extend(ids[end:])
    return train_ids, val_ids, test_ids

# ---------------- Main core API ----------------
def process_patient(patient_id: str,
                    patients_config_path: str = str(PATIENTS_CONFIG_PATH),
                    cfg: Optional[PreprocConfig] = None) -> Tuple[str, str]:
    """
    Returns (npy_path, splits_json_path)
    """
    if cfg is None:
        cfg = PreprocConfig()

    LOG_INFO("Running preprocessing ...")
    lfp_dir, labels_path, offset_path = resolve_patient_paths(patient_id, Path(patients_config_path))

    # Per-patient CSV dir to avoid cross-patient mixing
    csv_root = PROCESSED_DATA_DIR / cfg.csv_root_dirname
    lfp_csv_dir = csv_root / patient_id

    if cfg.do_export_csv:
        LOG_INFO("SKIP_LFP_EXPORT setting is controlled inside preprocessing.py")
        export_lfp_csvs(lfp_dir, lfp_csv_dir)
    else:
        LOG_INFO("Skipping LFP->CSV export (do_export_csv=False)")
        lfp_csv_dir.mkdir(parents=True, exist_ok=True)

    # Labels + optional offset shift
    events = load_labels_file(labels_path)
    if offset_path is not None and len(events) > 0:
        try:
            off_sec = _resolve_offset_seconds(offset_path)
        except Exception as e:
            LOG_WARN(f"Offset read failed ({e}); using 0.0")
            off_sec = 0.0
        new_events = []
        for ev in events:
            onset = float(ev["onset_sec"])
            if onset < 1e6:  # heuristic: not an absolute epoch
                onset = onset + off_sec
            new_events.append({"trial_id": ev["trial_id"], "label": ev["label"], "onset_sec": onset})
        events = new_events

    # Build windows
    ch2df = _load_all_channel_csvs(lfp_csv_dir)
    sr = float(cfg.sample_rate)
    win_samples = int(round(sr * (cfg.window_size_ms / 1000.0)))
    L = len(next(iter(ch2df.values())))

    items: List[Dict[str, Any]] = []
    pos_mask = np.zeros(L, dtype=bool)
    # positive windows + occupancy mask
    for ev in events:
        if ev["label"] not in {"HAARYE", "TUT"}:
            continue
        x = _slice_window_multich(ch2df, ev["onset_sec"], win_samples, sr)
        if x is None:
            continue
        items.append({"signals": x.astype(np.float32), "label": ev["label"], "trial_id": ev["trial_id"]})

        margin = int(round(sr * (cfg.other_margin_ms / 1000.0)))
        idx_center = int(round(ev["onset_sec"] * sr))
        i0 = max(0, idx_center - win_samples // 2 - margin)
        i1 = min(L, idx_center + win_samples // 2 + margin)
        if i1 > i0:
            pos_mask[i0:i1] = True

    num_pos = sum(1 for it in items if it["label"] in {"HAARYE", "TUT"})

    # OTHER windows (limited by ratio), only if we have positives
    if num_pos > 0 and cfg.other_ratio > 0:
        free = ~pos_mask
        edges = np.diff(np.concatenate(([0], free.astype(np.int8), [0])))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]

        centers: List[int] = []
        for s, e in zip(starts, ends):
            run_len = e - s
            if run_len < win_samples:
                continue
            c_start = s + win_samples // 2
            c_end   = e - win_samples // 2
            step    = int(round(sr * (cfg.other_stride_ms / 1000.0)))
            for c in range(c_start, c_end + 1, max(step, 1)):
                centers.append(c)

        target_other = int(max(1, cfg.other_ratio * num_pos))
        if len(centers) > target_other:
            idx = np.linspace(0, len(centers) - 1, target_other).astype(int)
            centers = [centers[i] for i in idx]

        for c in centers:
            t_center = c / sr
            x = _slice_window_multich(ch2df, t_center, win_samples, sr)
            if x is None:
                continue
            items.append({"signals": x.astype(np.float32), "label": "OTHER", "trial_id": f"OTHER_{c}"})

    # Trial-based splits
    train_ids, val_ids, test_ids = stratified_trials_split(
        [{"trial_id": it["trial_id"], "label": it["label"]} for it in items],
        cfg.per_class_test, cfg.per_class_val, cfg.seed
    )

    idx_train, idx_val, idx_test = [], [], []
    for i, it in enumerate(items):
        tid = it["trial_id"]
        if tid in test_ids:
            idx_test.append(i)
        elif tid in val_ids:
            idx_val.append(i)
        else:
            idx_train.append(i)

    # Save outputs
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = PROCESSED_DATA_DIR / f"{patient_id}_classification_data.npy"
    split_path = PROCESSED_DATA_DIR / f"{patient_id}_splits.json"

    np.save(npy_path, np.array(items, dtype=object), allow_pickle=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "patient_id": patient_id,
            "train_idx": idx_train,
            "val_idx": idx_val,
            "test_idx": idx_test
        }, f, ensure_ascii=False, indent=2)

    LOG_INFO(f"Saved {len(items)} windows → {npy_path}")
    LOG_INFO(f"Saved splits → {split_path}")
    return str(npy_path), str(split_path)

# ---------------- CLI entry point (for main.py subprocess) ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing for LFP classification")
    parser.add_argument("--patient_id", required=True, help="Patient ID, e.g. Patient_03")
    parser.add_argument("--out_data", type=str, default=None,
                        help="Output .npy path (main.py passes this, but by default we use PROCESSED_DATA_DIR)")
    parser.add_argument("--out_splits", type=str, default=None,
                        help="Output .json splits path (main.py passes this)")
    parser.add_argument("--force", action="store_true", help="Currently unused; kept for compatibility")

    args = parser.parse_args()

    # Run core pipeline
    npy_path, split_path = process_patient(args.patient_id)

    # If main.py passed explicit paths and they differ, copy to them
    if args.out_data is not None:
        out_data_path = Path(args.out_data)
        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(npy_path).resolve() != out_data_path.resolve():
            shutil.copy2(npy_path, out_data_path)
    if args.out_splits is not None:
        out_splits_path = Path(args.out_splits)
        out_splits_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(split_path).resolve() != out_splits_path.resolve():
            shutil.copy2(split_path, out_splits_path)
