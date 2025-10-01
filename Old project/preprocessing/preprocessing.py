#!/usr/bin/env python3
"""
Preprocessing pipeline for LFP ➜ LSTM-ready datasets (classification-oriented)
==============================================================================

Update (crop by last label end)
------------------------------
• Added optional `--crop_to_last_label_end` (default: True):
  The LFP recording is cropped so that its **end time equals the end of the last label included**.
  If a label has `offset_sec`, we use it; otherwise we use its `onset_sec`.

What this does
--------------
• Builds (N, T, F) sequences for **downstream classification** only.
• Does **not** classify or map labels; attaches original string labels (`y_str`).
• Exports `labels_vocab.json` with the unique label strings seen (for convenience only).
• Zero leakage across train/val/test (time-contiguous blocks).
• Features: `raw` / `bandpower` / `stft`.
• Saves `.npz` files + `metadata.json` (now includes `cutoff_end_sec` when cropping is enabled).

Label file assumptions
----------------------
• CSV with event onsets (and optional offsets): headers `onset_sec,label[,offset_sec]`.
• No label mutation in preprocessing (mapping to IDs happens later in dataset/training).

Usage
-----
python preprocessing_lstm.py \
  --lfp /path/to/patientX_lfp.npy \
  --fs 1000 \
  --labels /path/to/patientX_words.csv \
  --outdir ./PROCESSED_DATA_DIR/patientX \
  --seq_ms 600 --hop_ms 50 --feature raw \
  --crop_to_last_label_end

Output
------
• outdir/
    metadata.json
    labels_vocab.json
    train.npz  (X, y_str, idx, t0_sec)
    val.npz
    test.npz

Where:
  X:     float32, shape (N, T, F)
  y_str: object array of Python strings (exact labels from file)
  idx:   sample indices
  t0_sec: onset time per sample

Notes
-----
• For PyTorch LSTM expect (batch, T, F).
• Cropping ensures windows near the end of recording do not extend past the last label end; if needed,
  windows are shifted left internally to keep exact length.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.signal import welch, stft

# ------------------------------
# Dataclasses & Config
# ------------------------------

@dataclass
class SplitRatios:
    train: float
    val: float
    test: float

    def as_list(self):
        return [self.train, self.val, self.test]

    def validate(self):
        s = self.train + self.val + self.test
        if not math.isclose(s, 1.0, rel_tol=1e-3):
            raise ValueError(f"Splits must sum to 1.0 (got {s})")

@dataclass
class PreproConfig:
    fs: int
    seq_ms: int
    hop_ms: int
    feature: str = "raw"  # raw|bandpower|stft
    band_defs: Optional[Dict[str, Tuple[float, float]]] = None
    zscore_per: str = "channel"  # channel|sequence|global

    def seq_len(self) -> int:
        return int(round(self.seq_ms * 1e-3 * self.fs))

    def hop_len(self) -> int:
        return int(round(self.hop_ms * 1e-3 * self.fs))

# Default LFP bands (Hz)
DEFAULT_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "low_gamma": (30, 70),
    "high_gamma": (70, 150),
}

# ------------------------------
# Loading helpers
# ------------------------------

def load_lfp(lfp_path: str) -> np.ndarray:
    """Load LFP as np.ndarray (T, C) or (C, T) and return (T, C)."""
    arr = np.load(lfp_path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D LFP (T,C) or (C,T); got shape {arr.shape}")
    if arr.shape[0] < arr.shape[1]:  # likely (C, T)
        arr = arr.T
    return arr.astype(np.float32)

# Classification labels: events CSV (onset_sec,label[,offset_sec])

def load_labels_classification(csv_path: str) -> List[Tuple[float, str, Optional[float]]]:
    import csv
    events = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            onset = float(row["onset_sec"]) ; lab = row["label"].strip()
            off = float(row["offset_sec"]) if "offset_sec" in row and row["offset_sec"] else None
            events.append((onset, lab, off))
    return sorted(events, key=lambda x: x[0])

# ------------------------------
# Feature extraction
# ------------------------------

def zscore(x: np.ndarray, axis: Optional[int]) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True) + 1e-8
    return (x - m) / s


def features_raw(window_tc: np.ndarray, cfg: PreproConfig) -> np.ndarray:
    if cfg.zscore_per == "channel":
        return zscore(window_tc, axis=0)
    elif cfg.zscore_per == "sequence":
        return zscore(window_tc, axis=None)
    elif cfg.zscore_per == "global":
        return zscore(window_tc, axis=0)  # placeholder
    else:
        return window_tc


def features_bandpower(window_tc: np.ndarray, fs: int, bands: Dict[str, Tuple[float, float]]) -> np.ndarray:
    T, C = window_tc.shape
    frame = max(64, int(0.2 * fs))  # 200 ms
    step = max(16, int(0.05 * fs))  # 50 ms
    feats = []
    for start in range(0, max(1, T - frame + 1), step):
        seg = window_tc[start:start+frame, :]
        bp = []
        for ch in range(C):
            f, Pxx = welch(seg[:, ch], fs=fs, nperseg=min(frame, 256))
            for lo, hi in bands.values():
                mask = (f >= lo) & (f < hi)
                bp.append(np.trapz(Pxx[mask], f[mask]))
        feats.append(np.array(bp, dtype=np.float32))
    F = np.stack(feats, axis=0)  # (n_frames, C*B)
    F = zscore(F, axis=0)
    reps = math.ceil(T / F.shape[0])
    F_up = np.repeat(F, reps=reps, axis=0)[:T]
    return F_up


def features_stft(window_tc: np.ndarray, fs: int) -> np.ndarray:
    T, C = window_tc.shape
    nperseg = max(64, int(0.064 * fs))
    noverlap = int(0.75 * nperseg)
    feats = []
    for ch in range(C):
        f, t, Z = stft(window_tc[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
        mag = np.abs(Z).astype(np.float32)
        nb = 16
        splits = np.array_split(mag, nb, axis=0)
        comp = np.stack([s.mean(axis=0) for s in splits], axis=0)  # (nb, frames)
        comp = (comp - comp.mean(axis=1, keepdims=True)) / (comp.std(axis=1, keepdims=True) + 1e-8)
        feats.append(comp.T)  # (frames, nb)
    F = np.concatenate(feats, axis=1)
    reps = math.ceil(T / F.shape[0])
    F_up = np.repeat(F, reps=reps, axis=0)[:T]
    return F_up

# ------------------------------
# Windowing & sampling
# ------------------------------

def select_windows_classification(x_tc: np.ndarray, events: List[Tuple[float,str,Optional[float]]], fs: int, seq_len: int) -> List[Tuple[int,int,str,float]]:
    """For each onset, build a window of length seq_len around the event.
    Returns (s, e, label_str, onset_sec).
    """
    T = x_tc.shape[0]
    half = seq_len // 2
    out = []
    for onset_sec, label_str, _ in events:
        onset = int(round(onset_sec * fs))
        s = max(0, onset - half)
        e = min(T, s + seq_len)
        s = e - seq_len
        if s < 0:
            continue
        out.append((s, e, label_str, onset_sec))
    return out

# ------------------------------
# Splitting
# ------------------------------

def split_by_time(starts: List[int], total_T: int, ratios: SplitRatios) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    ratios.validate()
    borders = np.cumsum(ratios.as_list())
    train_cut = int(borders[0] * total_T)
    val_cut   = int(borders[1] * total_T)
    starts = np.array(starts)
    train_idx = np.nonzero(starts < train_cut)[0]
    val_idx   = np.nonzero((starts >= train_cut) & (starts < val_cut))[0]
    test_idx  = np.nonzero(starts >= val_cut)[0]
    return train_idx, val_idx, test_idx

# ------------------------------
# Build (classification-only)
# ------------------------------

def crop_to_last_label_end_if_needed(x_tc: np.ndarray, events: List[Tuple[float,str,Optional[float]]], fs: int, enable: bool) -> Tuple[np.ndarray, float]:
    """If enabled, crop the LFP so the recording ends exactly at the last included label end.
    Returns (cropped_x_tc, cutoff_end_sec).
    We define label end as `offset_sec` when present, otherwise `onset_sec`.
    """
    if not enable or not events:
        return x_tc, float(x_tc.shape[0] / fs)
    # last end in seconds
    last_end_sec = max((off if (off is not None) else onset) for (onset, _, off) in events)
    cut_T = min(x_tc.shape[0], int(round(last_end_sec * fs)))
    if cut_T <= 0:
        raise RuntimeError("Computed crop end is before start; check labels/offsets.")
    return x_tc[:cut_T], float(cut_T / fs)


def build_classification(x_tc: np.ndarray, fs: int, events: List[Tuple[float,str,Optional[float]]], cfg: PreproConfig, ratios: SplitRatios, outdir: str, crop_enable: bool):
    # Optional crop to last label end
    x_tc, cutoff_end_sec = crop_to_last_label_end_if_needed(x_tc, events, fs, crop_enable)

    seq_len = cfg.seq_len()
    samples = select_windows_classification(x_tc, events, fs, seq_len)
    if not samples:
        raise RuntimeError("No classification samples after selection. Check labels & timings.")

    starts = [s for (s,_,_,_) in samples]
    X_list, y_str_list, t0_list = [], [], []
    for s,e,label_str,onset_sec in samples:
        window = x_tc[s:e]
        if cfg.feature == "raw":
            Feat = features_raw(window, cfg)
        elif cfg.feature == "bandpower":
            Feat = features_bandpower(window, fs, cfg.band_defs or DEFAULT_BANDS)
        elif cfg.feature == "stft":
            Feat = features_stft(window, fs)
        else:
            raise ValueError(f"Unknown feature: {cfg.feature}")
        X_list.append(Feat.astype(np.float32))
        y_str_list.append(label_str)
        t0_list.append(float(onset_sec))

    X = np.stack(X_list, axis=0)
    y_str = np.array(y_str_list, dtype=object)

    train_idx, val_idx, test_idx = split_by_time(starts, x_tc.shape[0], ratios)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    save_splits(outdir, X, y_str, None, np.array(t0_list), splits)

    # Export labels vocab
    vocab = sorted({s for s in y_str_list})
    with open(os.path.join(outdir, "labels_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": vocab}, f, ensure_ascii=False, indent=2)

    return cutoff_end_sec

# ------------------------------
# Saving
# ------------------------------

def save_splits(outdir: str, X: np.ndarray, y_str: np.ndarray, mask: Optional[np.ndarray], t0: np.ndarray, splits: Dict[str, np.ndarray]):
    os.makedirs(outdir, exist_ok=True)
    idx_all = np.arange(X.shape[0])
    for name, sel in splits.items():
        payload = {"X": X[sel], "y_str": y_str[sel], "idx": idx_all[sel], "t0_sec": t0[sel]}
        if mask is not None:
            payload["mask"] = mask[sel]
        np.savez_compressed(os.path.join(outdir, f"{name}.npz"), **payload)

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LFP → LSTM preprocessing (classification; crop by last label end)")
    p.add_argument("--lfp", type=str, required=True, help="Path to LFP .npy (T,C) or (C,T)")
    p.add_argument("--fs", type=int, required=True, help="Sampling rate (Hz)")
    p.add_argument("--labels", type=str, required=True, help="CSV with labels (onset_sec,label[,offset_sec])")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--seq_ms", type=int, default=600)
    p.add_argument("--hop_ms", type=int, default=50)
    p.add_argument("--feature", choices=["raw","bandpower","stft"], default="raw")
    p.add_argument("--zscore_per", choices=["channel","sequence","global"], default="channel")
    p.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN","VAL","TEST"))
    p.add_argument("--crop_to_last_label_end", action="store_true", help="Crop LFP to end at last label end (default True if provided)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PreproConfig(
        fs=args.fs,
        seq_ms=args.seq_ms,
        hop_ms=args.hop_ms,
        feature=args.feature,
        zscore_per=args.zscore_per,
        band_defs=DEFAULT_BANDS,
    )

    ratios = SplitRatios(*args.split)
    ratios.validate()

    x_tc = load_lfp(args.lfp)  # (T,C)
    events = load_labels_classification(args.labels)

    cutoff_end_sec = None

    os.makedirs(args.outdir, exist_ok=True)

    cutoff_end_sec = build_classification(
        x_tc, args.fs, events, cfg, ratios, args.outdir, crop_enable=bool(args.crop_to_last_label_end)
    )

    meta = {
        "lfp_path": args.lfp,
        "fs": args.fs,
        "task": "classification_stage_preprocessing",
        "labels": args.labels,
        "seq_ms": args.seq_ms,
        "hop_ms": args.hop_ms,
        "feature": args.feature,
        "zscore_per": args.zscore_per,
        "splits": ratios.as_list(),
        "bands": DEFAULT_BANDS,
        "cutoff_end_sec": cutoff_end_sec,
    }
    with open(os.path.join(args.outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done. Files written to:", args.outdir)


if __name__ == "__main__":
    main()
