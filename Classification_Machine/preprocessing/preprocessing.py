# preprocessing/preprocessing.py
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import h5py
import numpy as np

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logger = logging.getLogger("preprocessing")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DATA_BASES_FOR_SEARCH = [
    # Local repo copy
    str(Path(__file__).resolve().parents[1] / "Data"),
    # Google Drive copy
    r"G:\My Drive\FinalProject\Data",
]

# -------------------------------------------------------------------------
# LSTM feature extraction configuration
# -------------------------------------------------------------------------
LSTM_FEATURE_BIN_SEC = 0.1  # length of each time bin in seconds for features
LSTM_FEATURE_BANDS: Tuple[Tuple[float, float], ...] = (
    (1.0, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 12.0),   # alpha
    (12.0, 30.0),  # beta
    (30.0, 70.0),  # low gamma
)


# -------------------------------------------------------------------------
# Label normalization
# -------------------------------------------------------------------------
def normalize_label(label_str: str) -> str:
    """
    Normalize raw label string into one of: 'HAARYE', 'TUT', 'OTHER'.

    IMPORTANT:
    - HAARYE stays HAARYE
    - TUT stays TUT
    - AHAV (אהב) is mapped to OTHER
    - Any other speech word → OTHER
    """
    if label_str is None:
        return "OTHER"

    s = str(label_str).strip().upper()

    # "HAARYE" / ARYE variants
    if "HAARYE" in s or "HARIE" in s or "ARYE" in s or "ARIE" in s:
        return "HAARYE"

    # "TUT" variants
    if "TUT" in s:
        return "TUT"

    # Explicitly treat AHAV as OTHER (but functionally any non-HAARYE/TUT
    # will also fall to OTHER)
    if "AHAV" in s:
        return "OTHER"

    # Everything else is OTHER
    return "OTHER"


# -------------------------------------------------------------------------
# Path resolution helpers
# -------------------------------------------------------------------------
def _log_resolve_attempt(base: str, rel: str, exists: bool) -> None:
    status = "[OK]" if exists else "[missing]"
    logger.debug(f"  Base: {base}\n    -> {os.path.join(base, rel)} {status}")


def resolve_in_bases(relative_path: str) -> str:
    """
    Search for `relative_path` under all DATA_BASES_FOR_SEARCH bases.
    Returns the first existing path, or raises FileNotFoundError.
    """
    logger.info("Resolving data paths (searching DATA_BASES_FOR_SEARCH):")
    for base in DATA_BASES_FOR_SEARCH:
        candidate = os.path.join(base, relative_path)
        exists = os.path.exists(candidate)
        _log_resolve_attempt(base, relative_path, exists)
        if exists:
            return candidate
    raise FileNotFoundError(
        f"Could not resolve path '{relative_path}' under any of DATA_BASES_FOR_SEARCH"
    )


def build_patient_rel_paths(patient_id: str) -> Tuple[str, str, str]:
    """
    (Legacy helper – no longer used by run_classification_preprocessing)

    Build relative paths for:
      - Atias_Labels.txt
      - LFP_signals dir
      - sound_w_times.mat
    Pattern: Patient_03/pt3_LFP_sound/...
    """
    try:
        num_str = patient_id.split("_")[1]
        num_int = int(num_str)
    except Exception:
        raise ValueError(f"Invalid patient_id format: {patient_id}. Expected 'Patient_XX'.")

    session_dir_name = f"pt{num_int}_LFP_sound"

    rel_labels = os.path.join(patient_id, session_dir_name, "Atias_Labels.txt")
    rel_lfp_dir = os.path.join(patient_id, session_dir_name, "LFP_signals")
    rel_offset = os.path.join(patient_id, session_dir_name, "sound_w_times.mat")

    return rel_labels, rel_lfp_dir, rel_offset


# -------------------------------------------------------------------------
# HDF5 helpers
# -------------------------------------------------------------------------
def _iter_datasets(h5obj, path: str = ""):
    """
    Recursively yield (path, dataset) for all h5py.Dataset objects under h5obj.
    """
    if isinstance(h5obj, h5py.Dataset):
        yield path, h5obj
    elif isinstance(h5obj, h5py.Group):
        for key in h5obj.keys():
            sub = h5obj[key]
            sub_path = f"{path}/{key}" if path else key
            yield from _iter_datasets(sub, sub_path)


def _choose_largest_numeric_dataset(f: h5py.File) -> h5py.Dataset:
    """
    Choose the largest numeric dataset in the file.

    This is robust: the continuous LFP signal is always the biggest dataset.
    Small metadata arrays (like 2 samples) will not be chosen.
    """
    candidates = []
    for path, ds in _iter_datasets(f):
        if not np.issubdtype(ds.dtype, np.number):
            continue
        size = ds.size
        candidates.append((size, path, ds))

    if not candidates:
        raise RuntimeError("No numeric datasets found in MAT file.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    size, path, ds = candidates[0]
    logger.debug(f"[DEBUG] Selected dataset '{path}' with shape {ds.shape}, size={size}")
    return ds


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------
def load_lfp_signals_from_mat_dir(lfp_dir: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load LFP signals from a directory of CSC*_LFP.mat files (MAT v7.3, h5py).

    Returns:
        signals: np.ndarray, shape (n_channels, n_samples)
        fs: float, sampling rate (Hz)
        channel_names: List[str]
    """
    lfp_dir_path = Path(lfp_dir)
    mat_files = sorted(lfp_dir_path.glob("CSC*_LFP.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No CSC*_LFP.mat files found in {lfp_dir}")

    all_signals: List[np.ndarray] = []
    channel_names: List[str] = []

    fs = 2000.0  # sampling rate (Hz)

    for mat_file in mat_files:
        with h5py.File(mat_file, "r") as f:
            data_ds = _choose_largest_numeric_dataset(f)
            data = np.array(data_ds[()], dtype=np.float32)

            if data.ndim > 1:
                data = data.reshape(-1)

        n_samples = data.shape[0]
        duration_sec = n_samples / fs
        channel_name = mat_file.stem  # e.g., "CSC10_LFP"

        logger.info(
            "[EXPORT] %s.mat: %d samples @ %.2f Hz → %.3fs",
            channel_name,
            n_samples,
            fs,
            duration_sec,
        )

        all_signals.append(data)
        channel_names.append(channel_name)

    signals = np.stack(all_signals, axis=0)  # [n_channels, n_samples]
    return signals, fs, channel_names


def load_label_events(labels_path: str) -> List[Dict[str, Any]]:
    """
    Load label events from Atias_Labels.txt (or compatible Labels.txt).

    Assumed formats (tries both):
      1) onset_sec  offset_sec  word
      2) word  onset_sec  offset_sec
    """
    events: List[Dict[str, Any]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                onset = float(parts[0])
                offset = float(parts[1])
                word = parts[2]
            except ValueError:
                word = parts[0]
                onset = float(parts[1])
                offset = float(parts[2])

            events.append(
                {
                    "onset": onset,
                    "offset": offset,
                    "label": word,
                }
            )

    logger.info(f"[INFO] Loaded {len(events)} label events")

    # Log distribution *before* windowing, but *after* normalization
    norm_counts: Dict[str, int] = {}
    for ev in events:
        norm = normalize_label(ev["label"])
        norm_counts[norm] = norm_counts.get(norm, 0) + 1

    logger.info("[INFO] Class counts in label events (after normalization, before windowing):")
    for k in sorted(norm_counts.keys()):
        logger.info(f"[INFO]   {k}: {norm_counts[k]}")

    return events


# -------------------------------------------------------------------------
# Auto-detect offset
# -------------------------------------------------------------------------
def auto_detect_offset_seconds(
    label_events: List[Dict[str, Any]],
    fs: float,
    n_samples: int,
    pre_sec: float,
    post_sec: float,
) -> float:
    """
    Automatically detect a global offset_sec such that:
        pre_sec <= onset_i + offset_sec <= duration - post_sec
    for as many events as possible.
    """
    if not label_events:
        logger.warning("[WARN] No label events provided, using offset_sec=0.0")
        return 0.0

    duration = n_samples / fs
    lower_bounds = []
    upper_bounds = []

    for ev in label_events:
        onset = float(ev["onset"])
        lower_bounds.append(pre_sec - onset)
        upper_bounds.append(duration - post_sec - onset)

    low = max(lower_bounds)
    high = min(upper_bounds)

    if low <= high:
        if low <= 0.0 <= high:
            offset_sec = 0.0
        else:
            offset_sec = low if abs(low) < abs(high) else high

        logger.info(
            "[INFO] Auto-detected offset_sec=%.3f s from intersection [%.3f, %.3f]",
            offset_sec,
            low,
            high,
        )
        return offset_sec

    offset_sec = 0.5 * (low + high)
    logger.warning(
        "[WARN] No common offset satisfying all events. "
        "Using offset_sec=%.3f (avg of [%.3f, %.3f]).",
        offset_sec,
        low,
        high,
    )
    return offset_sec


# -------------------------------------------------------------------------
# Window building – normalization
# -------------------------------------------------------------------------
def zscore_per_channel(signals: np.ndarray) -> np.ndarray:
    """
    Z-score per channel: signals is [n_channels, n_samples].
    """
    ch_mean = signals.mean(axis=1, keepdims=True)
    ch_std = signals.std(axis=1, keepdims=True) + 1e-8
    return (signals - ch_mean) / ch_std


# -------------------------------------------------------------------------
# Window building – speech events (IMPROVED)
# -------------------------------------------------------------------------
def build_speech_windows(
    signals: np.ndarray,
    fs: float,
    offset_sec: float,
    label_events: List[Dict[str, Any]],
    pre_sec: float,
    post_sec: float,
) -> List[Dict[str, Any]]:
    """
    Build windows around each labeled speech event (HAARYE, AHAV, TUT, etc).
    AHAV is mapped to OTHER via normalize_label.

    Returns a list of dicts:
      - signals: [T, C] float32
      - label: 'HAARYE' / 'TUT' / 'OTHER'
      - raw_label: original string from file (e.g. 'AHAV')
      - is_speech: True
      - start_time / end_time: in seconds (LFP local time)
      - event_index: index of the original event in label_events
      - onset_orig: onset time from labels file (before offset)
      - offset_orig: offset time from labels file (before offset)
      - center_time_local: onset + offset_sec (seconds in LFP local time)
    """
    n_channels, n_samples = signals.shape
    logger.info(f"[INFO] Signals shape: {n_channels} channels x {n_samples} samples")

    pre_samples = int(round(pre_sec * fs))
    post_samples = int(round(post_sec * fs))

    windows: List[Dict[str, Any]] = []

    total_events = len(label_events)
    skipped_total = 0
    skipped_per_label: Dict[str, int] = {}

    for idx, ev in enumerate(label_events):
        raw_label = ev["label"]
        onset = float(ev["onset"])
        offset_orig = float(ev["offset"])
        normalized = normalize_label(raw_label)

        center_time = onset + offset_sec  # local LFP time (sec)
        center_idx = int(round(center_time * fs))

        start_idx = center_idx - pre_samples
        end_idx = center_idx + post_samples

        # Out of bounds? Skip and log statistics (aggregated).
        if start_idx < 0 or end_idx > n_samples:
            skipped_total += 1
            skipped_per_label[normalized] = skipped_per_label.get(normalized, 0) + 1
            continue

        window_signals = signals[:, start_idx:end_idx].T.astype(np.float32)

        win = {
            "signals": window_signals,
            "label": normalized,
            "raw_label": raw_label,
            "is_speech": True,
            "start_time": float(start_idx / fs),
            "end_time": float(end_idx / fs),
            "event_index": idx,
            "onset_orig": onset,
            "offset_orig": offset_orig,
            "center_time_local": float(center_time),
        }
        windows.append(win)

    kept = len(windows)
    kept_pct = 100.0 * kept / total_events if total_events > 0 else 0.0

    logger.info(
        "[INFO] Speech windows built: %d / %d events (%.2f%% kept)",
        kept,
        total_events,
        kept_pct,
    )

    if skipped_total > 0:
        logger.warning(
            "[WARN] Skipped %d speech events due to window bounds (pre_sec=%.3f, post_sec=%.3f).",
            skipped_total,
            pre_sec,
            post_sec,
        )
        for lbl, cnt in sorted(skipped_per_label.items()):
            logger.warning("[WARN]   Skipped %d events for label '%s'", cnt, lbl)

    return windows


# -------------------------------------------------------------------------
# Window building – background (unlabeled) OTHER (IMPROVED)
# -------------------------------------------------------------------------
def build_background_windows(
    signals: np.ndarray,
    fs: float,
    offset_sec: float,
    label_events: List[Dict[str, Any]],
    pre_sec: float,
    post_sec: float,
    base_other_count: int,
    target_other_total: int,
    gap_margin_sec: float = 0.1,
    stride_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Build OTHER windows from *unlabeled* segments (real background).

    Idea:
      - Take gaps between labeled speech segments (plus before first and after last)
      - In every gap, place windows of length (pre_sec + post_sec) with stride `stride_sec`
        (default: non-overlapping, stride = win_len)
      - Label them as OTHER (raw_label='BACKGROUND', is_speech=False)
      - Stop once OTHER total reaches target_other_total.

    gap_margin_sec: minimal margin from labeled segments (in seconds).
    stride_sec:     step between background windows (in seconds).
                    If None, uses non-overlapping windows (stride = pre_sec + post_sec).
    """
    if target_other_total <= base_other_count:
        logger.info(
            "[INFO] No need to add background OTHER windows "
            "(base_other=%d, target_other_total=%d).",
            base_other_count,
            target_other_total,
        )
        return []

    max_additional = target_other_total - base_other_count

    n_channels, n_samples = signals.shape
    duration = n_samples / fs
    win_len = pre_sec + post_sec
    margin = gap_margin_sec
    step_sec = stride_sec if stride_sec is not None and stride_sec > 0.0 else win_len

    # Convert label events to local LFP times
    events_local: List[Tuple[float, float]] = []
    for ev in label_events:
        onset_local = float(ev["onset"]) + offset_sec
        offset_local = float(ev["offset"]) + offset_sec
        events_local.append((onset_local, offset_local))

    if not events_local:
        # No labels — take windows across entire recording
        gap_intervals = [(0.0, duration)]
    else:
        events_local.sort(key=lambda x: x[0])

        gap_intervals: List[Tuple[float, float]] = []

        # Before first event
        first_onset = events_local[0][0]
        if first_onset > 0.0 + margin:
            gap_intervals.append((0.0, first_onset - margin))

        # Between events
        for (on1, off1), (on2, off2) in zip(events_local[:-1], events_local[1:]):
            gap_start = off1 + margin
            gap_end = on2 - margin
            if gap_end - gap_start >= win_len:
                gap_intervals.append((gap_start, gap_end))

        # After last event
        last_off = events_local[-1][1]
        if duration - last_off > margin:
            gap_intervals.append((last_off + margin, duration))

    logger.info(
        "[INFO] Background gap intervals found: %d (duration=%.3f sec, win_len=%.3f sec, stride=%.3f sec, margin=%.3f sec)",
        len(gap_intervals),
        duration,
        win_len,
        step_sec,
        margin,
    )

    bg_windows: List[Dict[str, Any]] = []

    for (gap_start, gap_end) in gap_intervals:
        # Start placing centers so that full window is inside gap
        center = gap_start + pre_sec
        while center + post_sec <= gap_end and len(bg_windows) < max_additional:
            center_idx = int(round(center * fs))
            start_idx = center_idx - int(round(pre_sec * fs))
            end_idx = center_idx + int(round(post_sec * fs))

            if start_idx < 0 or end_idx > n_samples:
                center += step_sec
                continue

            window_signals = signals[:, start_idx:end_idx].T.astype(np.float32)

            win = {
                "signals": window_signals,
                "label": "OTHER",
                "raw_label": "BACKGROUND",
                "is_speech": False,
                "start_time": float(start_idx / fs),
                "end_time": float(end_idx / fs),
            }
            bg_windows.append(win)

            center += step_sec

        if len(bg_windows) >= max_additional:
            break

    logger.info(
        "[INFO] Background OTHER windows: base=%d, target_total=%d, "
        "requested_additional=%d, built=%d",
        base_other_count,
        target_other_total,
        max_additional,
        len(bg_windows),
    )

    if len(bg_windows) < max_additional:
        logger.warning(
            "[WARN] Could not reach target OTHER count from background gaps. "
            "Requested additional=%d, built=%d.",
            max_additional,
            len(bg_windows),
        )

    return bg_windows


# -------------------------------------------------------------------------
# LSTM feature extraction helpers
# -------------------------------------------------------------------------
def _compute_bandpower_1d(
    signal_1d: np.ndarray,
    fs: float,
    bands: Tuple[Tuple[float, float], ...],
) -> np.ndarray:
    """
    Compute bandpower features for a 1D signal using FFT-based PSD.

    Args:
        signal_1d: 1D array of shape [n_samples].
        fs: sampling rate in Hz.
        bands: tuple of (f_low, f_high) in Hz.

    Returns:
        1D array of shape [len(bands)] with band powers.
    """
    if signal_1d.ndim != 1:
        raise ValueError("signal_1d must be 1D.")

    n = signal_1d.shape[0]
    if n <= 1:
        return np.zeros(len(bands), dtype=np.float32)

    # Compute one-sided FFT and power spectral density
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(signal_1d)
    psd = (np.abs(fft_vals) ** 2) / float(n)

    band_powers: List[float] = []
    for f_low, f_high in bands:
        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if idx.size == 0:
            band_powers.append(0.0)
        else:
            band_powers.append(float(psd[idx].sum()))

    return np.array(band_powers, dtype=np.float32)


def extract_lstm_features_for_window(
    window_signals: np.ndarray,
    fs: float,
    bin_size_sec: float = LSTM_FEATURE_BIN_SEC,
    bands: Tuple[Tuple[float, float], ...] = LSTM_FEATURE_BANDS,
) -> np.ndarray:
    """
    Convert raw window signals [T_raw, C] into LSTM-ready features [T_bins, F].

    Steps:
      - Split the time axis into non-overlapping bins of length bin_size_sec.
      - For each bin and each channel, compute bandpower in the specified bands.
      - Concatenate all channel-bandpowers to a single feature vector per bin.

    Args:
        window_signals: np.ndarray of shape [T_raw, C], float32.
        fs: sampling rate in Hz.
        bin_size_sec: length of each time bin in seconds.
        bands: frequency bands for bandpower computation.

    Returns:
        features: np.ndarray of shape [T_bins, F], where
                  T_bins = floor(T_raw / (bin_size_sec * fs)),
                  F = C * len(bands).
    """
    if window_signals.ndim != 2:
        raise ValueError("window_signals must be 2D [T_raw, C].")

    T_raw, C = window_signals.shape
    bin_samples = int(round(bin_size_sec * fs))
    if bin_samples <= 0:
        raise ValueError("bin_size_sec too small, results in bin_samples <= 0.")

    # Number of full bins we can take
    n_bins = T_raw // bin_samples
    if n_bins == 0:
        # Too short window for at least one bin, fall back to a single bin
        # using the whole window.
        n_bins = 1
        bin_samples = T_raw

    # Trim to an integer number of bins (except in the fallback case above)
    if n_bins * bin_samples <= T_raw:
        trimmed = window_signals[: n_bins * bin_samples, :]
    else:
        trimmed = window_signals

    # Reshape into [n_bins, bin_samples, C]
    trimmed = trimmed.reshape(n_bins, bin_samples, C)

    all_features: List[np.ndarray] = []
    for b in range(n_bins):
        bin_seg = trimmed[b]  # [bin_samples, C]
        bin_feats: List[np.ndarray] = []
        for ch_idx in range(C):
            sig_ch = bin_seg[:, ch_idx]
            bp = _compute_bandpower_1d(sig_ch, fs, bands)  # [len(bands)]
            bin_feats.append(bp)
        # Concatenate features from all channels → [C * len(bands)]
        bin_feat_vec = np.concatenate(bin_feats, axis=0)
        all_features.append(bin_feat_vec.astype(np.float32))

    features = np.stack(all_features, axis=0).astype(np.float32)  # [T_bins, F]
    return features


# -------------------------------------------------------------------------
# Split metadata saving
# -------------------------------------------------------------------------
def save_splits_dummy(
    windows: List[Dict[str, Any]],
    out_splits_path: str,
    n_folds: int = 5,
) -> None:
    """
    Save simple split metadata for compatibility with main.py.
    Actual CV folds are built inside train_classification.py.
    """
    labels = [w["label"] for w in windows]
    counts = {c: labels.count(c) for c in set(labels)}

    data = {
        "meta": {
            "num_samples": len(windows),
            "class_counts": counts,
        },
        "folds": n_folds,
    }

    os.makedirs(os.path.dirname(out_splits_path), exist_ok=True)
    with open(out_splits_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"[INFO] Saved splits → {out_splits_path}")


# -------------------------------------------------------------------------
# High-level API for main.py
# -------------------------------------------------------------------------
def run_classification_preprocessing(
    patient_id: str,
    out_data: Optional[str] = None,
    out_splits: Optional[str] = None,
    pre_sec: float = 0.5,
    post_sec: float = 1.0,
    n_folds: int = 5,
    force: bool = True,
) -> str:
    """
    High-level helper used by main.py to (re)generate classification data.

    If out_data/out_splits are None, they are created under:
        <repo_root>/processed_data/{patient_id}_classification_data.npy
        <repo_root>/processed_data/{patient_id}_classification_splits.json

    Paths for labels / LFP / offset are taken from:
        <repo_root>/config/patients_config.json

    Returns:
        The path to the saved .npy file (out_data).
    """
    base_dir = Path(__file__).resolve().parents[1]

    # Decide output paths
    if out_data is None:
        processed_dir = base_dir / "processed_data"
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_data_path = processed_dir / f"{patient_id}_classification_data.npy"
    else:
        out_data_path = Path(out_data)

    if out_splits is None:
        out_splits_path = out_data_path.parent / f"{out_data_path.stem}_splits.json"
    else:
        out_splits_path = Path(out_splits)

    # Early exit if file exists and not forcing
    if out_data_path.exists() and not force:
        logger.info(
            f"[INFO] {out_data_path} already exists. Use force=True to overwrite."
        )
        return str(out_data_path)

    logger.info(
        "[INFO] Running classification preprocessing for patient_id=%s",
        patient_id,
    )

    # ------------------------------------------------------------------
    # Load patients_config.json and resolve relative paths from there
    # ------------------------------------------------------------------
    config_path = base_dir / "config" / "patients_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"patients_config.json not found at: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    patients_cfg = cfg.get("patients", {})
    if patient_id not in patients_cfg:
        raise KeyError(
            f"Patient '{patient_id}' not found in patients_config.json"
        )

    p_cfg = patients_cfg[patient_id]

    rel_labels = p_cfg["labels_file"]        # e.g. "Patient_01/Labels/Labels.txt"
    rel_lfp_dir = p_cfg["lfp_folder"]       # e.g. "Patient_01/LFP_signals"
    rel_offset = p_cfg.get("offset_file")   # e.g. "Patient_01/sound_w_times.mat"

    sample_rate_cfg = p_cfg.get("sample_rate", None)
    window_size_cfg = p_cfg.get("window_size", None)

    # OTHER-related config (used below)
    other_ratio = float(p_cfg.get("other_ratio", 1.2))
    other_margin_ms = float(p_cfg.get("other_margin_ms", 0.0))
    other_stride_ms = float(p_cfg.get("other_stride_ms", 0.0))
    target_other_pos_ratio = float(p_cfg.get("target_other_pos_ratio", 0.0))

    logger.info("Resolved config for %s:", patient_id)
    logger.info("  labels_file: %s", rel_labels)
    logger.info("  lfp_folder: %s", rel_lfp_dir)
    if rel_offset is not None:
        logger.info("  offset_file: %s", rel_offset)
    if sample_rate_cfg is not None:
        logger.info("  sample_rate (cfg): %s", sample_rate_cfg)
    if window_size_cfg is not None:
        logger.info("  window_size (cfg): %s", window_size_cfg)
    logger.info("  other_ratio (cfg): %.3f", other_ratio)
    if other_margin_ms > 0.0:
        logger.info("  other_margin_ms (cfg): %.1f", other_margin_ms)
    if other_stride_ms > 0.0:
        logger.info("  other_stride_ms (cfg): %.1f", other_stride_ms)
    if target_other_pos_ratio > 0.0:
        logger.info("  target_other_pos_ratio (cfg): %.3f", target_other_pos_ratio)

    # Resolve absolute paths using DATA_BASES_FOR_SEARCH
    labels_path = resolve_in_bases(rel_labels)
    lfp_dir = resolve_in_bases(rel_lfp_dir)
    if rel_offset is not None:
        _ = resolve_in_bases(rel_offset)  # path-checked, not currently used

    # Load signals
    signals, fs, channel_names = load_lfp_signals_from_mat_dir(lfp_dir)
    n_channels, n_samples = signals.shape

    logger.info(
        "[INFO] Loaded LFP signals: %d channels, %d samples (fs=%.2f Hz)",
        n_channels,
        n_samples,
        fs,
    )

    # Check consistency with sample_rate from config (if exists)
    if sample_rate_cfg is not None:
        try:
            sr_cfg = float(sample_rate_cfg)
            if abs(sr_cfg - fs) > 1e-3:
                logger.warning(
                    "[WARN] sample_rate in config (%.3f) does not match detected fs (%.3f).",
                    sr_cfg,
                    fs,
                )
        except Exception:
            logger.warning(
                "[WARN] Could not interpret sample_rate '%s' from config as float.",
                str(sample_rate_cfg),
            )

    # Z-score per channel
    signals = zscore_per_channel(signals)
    logger.info("[INFO] Applied per-channel z-score normalization to raw LFP signals.")

    # Load labels
    label_events = load_label_events(labels_path)

    # Auto-detect global offset
    offset_sec = auto_detect_offset_seconds(
        label_events=label_events,
        fs=fs,
        n_samples=n_samples,
        pre_sec=pre_sec,
        post_sec=post_sec,
    )

    # Richer logging around offset and event times
    logger.info("[INFO] Using offset_sec = %.3f s", offset_sec)

    duration_sec = n_samples / fs
    logger.info("[INFO] Recording duration (LFP): %.3f sec", duration_sec)

    if label_events:
        onsets_local = [float(ev["onset"]) + offset_sec for ev in label_events]
        logger.info(
            "[INFO] Local onset range after offset: [%.3f, %.3f] sec",
            min(onsets_local),
            max(onsets_local),
        )
    else:
        logger.warning("[WARN] No label events to report onset range for.")

    # Build speech windows (HAARYE / AHAV→OTHER / TUT)
    speech_windows = build_speech_windows(
        signals=signals,
        fs=fs,
        offset_sec=offset_sec,
        label_events=label_events,
        pre_sec=pre_sec,
        post_sec=post_sec,
    )

    # Count current labels from speech events only
    speech_counts: Dict[str, int] = {}
    for w in speech_windows:
        speech_counts[w["label"]] = speech_counts.get(w["label"], 0) + 1

    logger.info("[INFO] Class counts from speech events (after windowing):")
    for k in sorted(speech_counts.keys()):
        logger.info(f"[INFO]   {k}: {speech_counts[k]}")

    base_other = speech_counts.get("OTHER", 0)
    max_main_class = max(
        speech_counts.get("HAARYE", 0),
        speech_counts.get("TUT", 0),
    )

    # Compute target OTHER using 'other_ratio' from config if possible.
    # other_ratio ~ desired OTHER / max(HAARYE, TUT).
    if max_main_class > 0:
        target_other_total = int(other_ratio * max_main_class)
        if target_other_total < base_other:
            target_other_total = base_other
    else:
        target_other_total = base_other

    # If target_other_pos_ratio > 0, we can optionally bump the target up further
    # so that OTHER ≈ target_other_pos_ratio * (#positive speech without OTHER).
    if target_other_pos_ratio > 0.0:
        pos_without_other = (
            speech_counts.get("HAARYE", 0) + speech_counts.get("TUT", 0)
        )
        if pos_without_other > 0:
            target_from_pos = int(target_other_pos_ratio * pos_without_other)
            if target_from_pos > target_other_total:
                target_other_total = target_from_pos

    logger.info(
        "[INFO] Target OTHER count: base_other=%d, max_main_class=%d, target_other_total=%d",
        base_other,
        max_main_class,
        target_other_total,
    )

    # Background OTHER parameters from config (if available)
    gap_margin_sec = other_margin_ms / 1000.0 if other_margin_ms > 0.0 else 0.1
    stride_sec = other_stride_ms / 1000.0 if other_stride_ms > 0.0 else None

    # Build background OTHER windows from unlabeled gaps
    bg_windows = build_background_windows(
        signals=signals,
        fs=fs,
        offset_sec=offset_sec,
        label_events=label_events,
        pre_sec=pre_sec,
        post_sec=post_sec,
        base_other_count=base_other,
        target_other_total=target_other_total,
        gap_margin_sec=gap_margin_sec,
        stride_sec=stride_sec,
    )

    windows = speech_windows + bg_windows

    # ---------------------------------------------------------------------
    # Build LSTM-friendly features for each window: [T_bins, F]
    # ---------------------------------------------------------------------
    logger.info("[INFO] Building LSTM features for each window ...")
    example_feat_shape = None

    for w in windows:
        sig = w["signals"]  # [T_raw, C]
        feats = extract_lstm_features_for_window(
            window_signals=sig,
            fs=fs,
            bin_size_sec=LSTM_FEATURE_BIN_SEC,
            bands=LSTM_FEATURE_BANDS,
        )
        w["features"] = feats  # [T_bins, F]

        if example_feat_shape is None:
            example_feat_shape = feats.shape

    if example_feat_shape is not None:
        logger.info(
            "[INFO] Example LSTM feature shape per window: [T=%d, F=%d]",
            example_feat_shape[0],
            example_feat_shape[1],
        )
    else:
        logger.warning("[WARN] No windows available for LSTM feature extraction.")

    # Final counts
    final_counts: Dict[str, int] = {}
    for w in windows:
        final_counts[w["label"]] = final_counts.get(w["label"], 0) + 1

    logger.info("[INFO] Class counts in preprocessing (before splits):")
    for k in sorted(final_counts.keys()):
        logger.info(f"[INFO]   {k}: {final_counts[k]}")
    logger.info(f"[INFO] Total windows: {len(windows)}")

    # Save .npy
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_data_path, np.array(windows, dtype=object))
    logger.info(f"[INFO] Saved {len(windows)} windows → {out_data_path}")

    # Save splits meta
    save_splits_dummy(windows, str(out_splits_path), n_folds=n_folds)

    return str(out_data_path)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess LFP data for classification.")
    parser.add_argument("--patient_id", required=True, help="Patient ID, e.g., Patient_03")
    parser.add_argument("--out_data", required=True, help="Path to .npy file to save windows")
    parser.add_argument("--out_splits", required=True, help="Path to .json file to save splits")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if exists")
    parser.add_argument(
        "--pre_sec",
        type=float,
        default=0.5,
        help="Seconds before event onset for window.",
    )
    parser.add_argument(
        "--post_sec",
        type=float,
        default=1.0,
        help="Seconds after event onset for window.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds metadata for CV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_classification_preprocessing(
        patient_id=args.patient_id,
        out_data=args.out_data,
        out_splits=args.out_splits,
        pre_sec=args.pre_sec,
        post_sec=args.post_sec,
        n_folds=args.n_folds,
        force=args.force,
    )


if __name__ == "__main__":
    main()