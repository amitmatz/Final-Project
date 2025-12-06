import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import h5py
import numpy as np

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("preprocessing")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_BASES_FOR_SEARCH = [
    # Local repo copy
    str(Path(__file__).resolve().parents[1] / "Data"),
    # Google Drive copy
    r"G:\My Drive\FinalProject\Data",
]


# -----------------------------------------------------------------------------
# Label normalization
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Path resolution helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# HDF5 helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
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
    Load label events from Atias_Labels.txt.

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


# -----------------------------------------------------------------------------
# Auto-detect offset
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Window building – normalization
# -----------------------------------------------------------------------------
def zscore_per_channel(signals: np.ndarray) -> np.ndarray:
    """
    Z-score per channel: signals is [n_channels, n_samples].
    """
    ch_mean = signals.mean(axis=1, keepdims=True)
    ch_std = signals.std(axis=1, keepdims=True) + 1e-8
    return (signals - ch_mean) / ch_std


# -----------------------------------------------------------------------------
# Window building – speech events (IMPROVED)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Window building – background (unlabeled) OTHER (IMPROVED)
# -----------------------------------------------------------------------------
def build_background_windows(
    signals: np.ndarray,
    fs: float,
    offset_sec: float,
    label_events: List[Dict[str, Any]],
    pre_sec: float,
    post_sec: float,
    base_other_count: int,
    target_other_total: int,
) -> List[Dict[str, Any]]:
    """
    Build OTHER windows from *unlabeled* segments (real background).

    Idea:
      - Take gaps between labeled speech segments (plus before first and after last)
      - In every gap, place non-overlapping windows of length (pre_sec + post_sec)
      - Label them as OTHER (raw_label='BACKGROUND', is_speech=False)
      - Stop once OTHER total reaches target_other_total.
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
    margin = 0.1  # small safety margin in seconds
    step_sec = win_len  # non-overlapping background windows

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
        "[INFO] Background gap intervals found: %d (duration=%.3f sec, win_len=%.3f sec)",
        len(gap_intervals),
        duration,
        win_len,
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


# -----------------------------------------------------------------------------
# Split metadata saving
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
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

    out_data_path = Path(args.out_data)
    out_splits_path = Path(args.out_splits)

    if out_data_path.exists() and not args.force:
        logger.info(f"[INFO] {out_data_path} already exists. Use --force to overwrite.")
        return

    logger.info("[INFO] Running preprocessing ...")

    # Resolve paths
    rel_labels, rel_lfp_dir, rel_offset = build_patient_rel_paths(args.patient_id)

    labels_path = resolve_in_bases(rel_labels)
    lfp_dir = resolve_in_bases(rel_lfp_dir)
    _ = resolve_in_bases(rel_offset)  # not currently used, but path-checked

    # Load signals
    signals, fs, channel_names = load_lfp_signals_from_mat_dir(lfp_dir)
    n_channels, n_samples = signals.shape

    logger.info(
        "[INFO] Loaded LFP signals: %d channels, %d samples (fs=%.2f Hz)",
        n_channels,
        n_samples,
        fs,
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
        pre_sec=args.pre_sec,
        post_sec=args.post_sec,
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
        pre_sec=args.pre_sec,
        post_sec=args.post_sec,
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

    # We want OTHER to be (slightly) the largest class, but not crazy:
    # e.g., ~20% more than the largest among HAARYE/TUT.
    target_other_total = int(1.2 * max_main_class) if max_main_class > 0 else base_other

    logger.info(
        "[INFO] Target OTHER count: base_other=%d, max_main_class=%d, target_other_total=%d",
        base_other,
        max_main_class,
        target_other_total,
    )

    # Build background OTHER windows from unlabeled gaps
    bg_windows = build_background_windows(
        signals=signals,
        fs=fs,
        offset_sec=offset_sec,
        label_events=label_events,
        pre_sec=args.pre_sec,
        post_sec=args.post_sec,
        base_other_count=base_other,
        target_other_total=target_other_total,
    )

    windows = speech_windows + bg_windows

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
    save_splits_dummy(windows, str(out_splits_path), n_folds=args.n_folds)


if __name__ == "__main__":
    main()
