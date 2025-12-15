# Classification/ train_classification.py
import os
import sys
import copy
import logging
import math
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ===========================================================================
# LSTMClassifier – internal definition (can be overridden by external model)
# ===========================================================================


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        """
        Simple LSTM-based classifier.

        input_size:   feature dimension per time step
        hidden_size:  size of LSTM hidden state
        num_layers:   number of LSTM layers
        num_classes:  number of output classes
        bidirectional: use bidirectional LSTM if True
        dropout:     dropout inside LSTM (if num_layers > 1) and before final FC
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        """
        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits


# ===========================================================================
# OPTIONAL OVERRIDE: if you have Classification/model.py or model.py
# ===========================================================================

try:
    from Classification.model import LSTMClassifier as ExternalLSTMClassifier

    LSTMClassifier = ExternalLSTMClassifier
    logger.info("INFO: Using LSTMClassifier from Classification.model")
except Exception:
    try:
        from model import LSTMClassifier as ExternalLSTMClassifier

        LSTMClassifier = ExternalLSTMClassifier
        logger.info("INFO: Using LSTMClassifier from model.py")
    except Exception:
        logger.info(
            "INFO: Using internal LSTMClassifier definition in train_classification.py"
        )


# ===========================================================================
# Utility functions
# ===========================================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_confusion_matrix(
    confusion: np.ndarray,
    idx_to_label: Dict[int, str],
    prefix: str = "",
) -> None:
    """
    Log confusion matrix in a nice table format:
    rows=true labels, cols=predicted labels
    """
    n_classes = confusion.shape[0]
    header = "             " + "   ".join(
        [f"{idx_to_label[j]:>7}" for j in range(n_classes)]
    )
    logger.info("%sConfusion Matrix [rows=true, cols=pred]:", prefix)
    logger.info("%s%s", prefix, header)
    for i in range(n_classes):
        row_label = f"{idx_to_label[i]:>7}"
        row_values = "  ".join([f"{confusion[i, j]:>7}" for j in range(n_classes)])
        logger.info("%s%8s  %s", prefix, row_label, row_values)


def compute_per_class_metrics(
    confusion: np.ndarray,
    idx_to_label: Dict[int, str],
) -> Tuple[Dict[int, Dict[str, float]], float]:
    """
    Compute precision/recall/F1 per class and macro-F1.
    """
    n_classes = confusion.shape[0]
    metrics: Dict[int, Dict[str, float]] = {}
    f1_list: List[float] = []

    for i in range(n_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics[i] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(confusion[i, :].sum()),
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    return metrics, macro_f1


# ===========================================================================
# Dataset
# ===========================================================================


class ClassificationDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        features: (N, seq_len, feat_dim)
        labels:   (N,)
        """
        assert features.ndim == 3, "features must be (N, seq_len, feat_dim)"
        assert labels.ndim == 1, "labels must be (N,)"
        assert (
            features.shape[0] == labels.shape[0]
        ), "features and labels must have same length"

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y)


# ===========================================================================
# Data loading and preprocessing
# ===========================================================================


def _infer_label_field(sample: Dict[str, Any]) -> str:
    """
    Try to infer the text label from a sample dict.
    """
    if "label" in sample:
        return str(sample["label"])
    if "word" in sample:
        return str(sample["word"])
    if "class" in sample:
        return str(sample["class"])
    return "OTHER"


def load_classification_data(
    data_path: str,
    pool_size: int = 50,
    class_map: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[str, int]]:
    """
    Load classification data from the .npy produced in preprocessing.

    Temporal pooling is applied with pool_size (less aggressive by default).
    No normalization is performed here – normalization is done per-fold using
    only the TRAIN set to avoid information leakage.

    Returns:
        X:  (N, seq_len, feat_dim)
        y:  (N,)
        idx_to_label: {idx: label_name}
        label_to_idx: {label_name: idx}
    """
    logger.info("INFO: Loading data from %s ...", data_path)
    raw = np.load(data_path, allow_pickle=True)
    logger.info("INFO: [LOAD] Total windows in file: %d", len(raw))

    signals_list: List[np.ndarray] = []
    labels_str: List[str] = []

    background_count = 0

    for sample in raw:
        if not isinstance(sample, dict):
            raise ValueError("Each entry in the npy file must be a dict.")

        is_speech = bool(sample.get("is_speech", True))
        if not is_speech:
            label_str = "OTHER"
            background_count += 1
        else:
            label_str = _infer_label_field(sample).upper()

        sig = sample.get("signals", None)
        if sig is None:
            raise ValueError("Sample is missing 'signals' key.")

        sig = np.asarray(sig, dtype=np.float32)

        if sig.ndim != 2:
            raise ValueError("signals must be 2D (time, channels) or (channels, time).")

        # Ensure shape: (time, channels)
        if sig.shape[0] == 48 and sig.shape[1] != 48:
            sig = sig.T
        elif sig.shape[1] == 48:
            pass
        else:
            if sig.shape[1] < sig.shape[0]:
                pass
            else:
                sig = sig.T

        signals_list.append(sig)
        labels_str.append(label_str)

    logger.info(
        "INFO: [LOAD] Background windows (is_speech=False, all mapped to OTHER): %d",
        background_count,
    )

    # Build label mapping
    unique_labels = sorted(list(set(labels_str)))
    if class_map is None:
        preferred_order = ["HAARYE", "OTHER", "TUT"]
        label_to_idx: Dict[str, int] = {}
        idx = 0
        for name in preferred_order:
            if name in unique_labels:
                label_to_idx[name] = idx
                idx += 1
        for name in unique_labels:
            if name not in label_to_idx:
                label_to_idx[name] = idx
                idx += 1
    else:
        label_to_idx = class_map

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Temporal pooling
    pooled_list: List[np.ndarray] = []
    logger_once = getattr(load_classification_data, "_logger_once", False)

    for sig in signals_list:
        t, c = sig.shape
        if not logger_once:
            flat_dim = t * c
            logger.info(
                "INFO: Feature shape per sample BEFORE temporal pooling: "
                "seq_len=%d, feat_dim=%d, flat_dim=%d",
                t,
                c,
                flat_dim,
            )
            load_classification_data._logger_once = True  # type: ignore[attr-defined]
            logger_once = True

        t_trim = (t // pool_size) * pool_size
        if t_trim == 0:
            raise ValueError(
                f"Sequence too short for pool_size {pool_size}: length={t}"
            )

        sig_trim = sig[:t_trim]
        new_len = t_trim // pool_size
        sig_reshaped = sig_trim.reshape(new_len, pool_size, c)
        pooled = sig_reshaped.mean(axis=1)
        pooled_list.append(pooled)

    X = np.stack(pooled_list, axis=0)
    seq_len, feat_dim = X.shape[1], X.shape[2]
    flat_dim_after = seq_len * feat_dim
    logger.info(
        "INFO: After temporal pooling: seq_len=%d, feat_dim_per_step=%d, flat_dim=%d (pool_size=%d)",
        seq_len,
        feat_dim,
        flat_dim_after,
        pool_size,
    )
    logger.info(
        "INFO: Effective input size to LSTM per sample: seq_len=%d, feat_dim=%d, flat_dim=%d",
        seq_len,
        feat_dim,
        flat_dim_after,
    )

    # Map labels to integers
    y = np.array([label_to_idx[s] for s in labels_str], dtype=np.int64)

    logger.info("INFO: [LOAD] Final class counts (raw, before normalization):")
    total_samples = X.shape[0]
    for idx, name in idx_to_label.items():
        count = int((y == idx).sum())
        logger.info("INFO:   %s: %d", name, count)
    logger.info("INFO: Total samples: %d", total_samples)

    logger.info("INFO: Label distribution after loading (per-class counts):")
    for idx, name in idx_to_label.items():
        count = int((y == idx).sum())
        logger.info("INFO:   %s: %d", name, count)

    return X.astype(np.float32), y, idx_to_label, label_to_idx


# ===========================================================================
# Data augmentation
# ===========================================================================


def augment_signals_gaussian(x: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x + noise


def _apply_time_envelope(x: np.ndarray) -> np.ndarray:
    """
    Smooth random time-varying gain envelope.
    """
    seq_len = x.shape[0]
    if seq_len <= 1:
        return x

    num_knots = 4
    knot_positions = np.linspace(0, seq_len - 1, num_knots)
    knot_gains = np.random.uniform(0.8, 1.2, size=num_knots)
    t = np.arange(seq_len)
    envelope = np.interp(t, knot_positions, knot_gains).reshape(-1, 1)
    return x * envelope.astype(np.float32)


def _simple_time_warp(x: np.ndarray, max_warp: float = 0.1) -> np.ndarray:
    """
    Simple time warping by resampling the sequence length slightly.
    """
    seq_len, feat_dim = x.shape
    if seq_len <= 2 or max_warp <= 0.0:
        return x

    factor = 1.0 + np.random.uniform(-max_warp, max_warp)
    new_len = max(2, int(round(seq_len * factor)))

    orig_t = np.linspace(0.0, 1.0, seq_len)
    new_t = np.linspace(0.0, 1.0, new_len)

    x_resampled = np.zeros((new_len, feat_dim), dtype=np.float32)
    for ch in range(feat_dim):
        x_resampled[:, ch] = np.interp(new_t, orig_t, x[:, ch])

    final = np.zeros_like(x, dtype=np.float32)
    back_t = np.linspace(0.0, 1.0, seq_len)
    for ch in range(feat_dim):
        final[:, ch] = np.interp(back_t, np.linspace(0.0, 1.0, new_len), x_resampled[:, ch])

    return final


def random_augment_sample(
    sample: np.ndarray,
    noise_std: float = 0.05,
    max_time_shift_steps: int = 2,
    is_haarye: bool = False,
) -> np.ndarray:
    """
    Structural + noise augmentations.
    """
    x = np.array(sample, dtype=np.float32, copy=True)
    seq_len = x.shape[0]

    eff_noise_std = noise_std
    eff_max_shift = max_time_shift_steps

    if is_haarye:
        eff_noise_std = noise_std * 0.5
        eff_max_shift = max(1, max_time_shift_steps // 2) if max_time_shift_steps > 0 else 0

        if np.random.rand() < 0.4:
            x = _simple_time_warp(x, max_warp=0.08)
        if np.random.rand() < 0.5:
            x = _apply_time_envelope(x)
        if eff_max_shift > 0 and seq_len > 1 and np.random.rand() < 0.5:
            shift = np.random.randint(-eff_max_shift, eff_max_shift + 1)
            if shift != 0:
                x = np.roll(x, shift=shift, axis=0)
        if eff_noise_std > 0.0 and np.random.rand() < 0.7:
            noise = np.random.normal(
                loc=0.0,
                scale=eff_noise_std,
                size=x.shape,
            ).astype(np.float32)
            x = x + noise

        return x.astype(np.float32)

    if np.random.rand() < 0.3:
        x = _simple_time_warp(x, max_warp=0.1)
    if np.random.rand() < 0.5:
        x = _apply_time_envelope(x)

    if eff_max_shift > 0 and seq_len > 1:
        shift = np.random.randint(-eff_max_shift, eff_max_shift + 1)
        if shift != 0:
            x = np.roll(x, shift=shift, axis=0)

    if eff_noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=eff_noise_std, size=x.shape).astype(
            np.float32
        )
        x = x + noise

    return x.astype(np.float32)


def build_balanced_train_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    augment_minor: bool,
    logger,
    noise_std: float = 0.05,
    max_time_shift_steps: int = 2,
    haarye_class_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance TRAIN set:
      - if augment_minor=False: downsample to smallest class
      - if augment_minor=True: upsample minorities with augmentation to largest class
    """
    classes = np.unique(y_train)
    class_to_indices: Dict[int, List[int]] = {
        int(c): np.where(y_train == c)[0].tolist() for c in classes
    }
    raw_counts = {c: len(idxs) for c, idxs in class_to_indices.items()}

    if not augment_minor:
        min_count = min(raw_counts.values())
        logger.info(
            "INFO: Balancing TRAIN by DOWNsampling to smallest class: "
            "min_count=%d, raw_counts=%s",
            min_count,
            raw_counts,
        )

        new_X: List[np.ndarray] = []
        new_y: List[int] = []

        for c in classes:
            c_int = int(c)
            idxs = class_to_indices[c_int]
            if len(idxs) > min_count:
                chosen = np.random.choice(idxs, size=min_count, replace=False)
            else:
                chosen = np.array(idxs, dtype=int)

            for idx in chosen:
                new_X.append(X_train[idx])
                new_y.append(c_int)

        new_X_arr = np.stack(new_X).astype(np.float32)
        new_y_arr = np.array(new_y, dtype=np.int64)
        return new_X_arr, new_y_arr

    max_count = max(raw_counts.values())
    logger.info(
        "INFO: Balancing TRAIN by UPsampling minority classes with data augmentation "
        "to max_count=%d, raw_counts=%s",
        max_count,
        raw_counts,
    )

    if haarye_class_id is not None:
        logger.info(
            "INFO: HAARYE class id detected for special augmentation: %d",
            haarye_class_id,
        )

    new_X: List[np.ndarray] = []
    new_y: List[int] = []

    for c in classes:
        c_int = int(c)
        idxs = class_to_indices[c_int]
        current_count = len(idxs)

        for idx in idxs:
            new_X.append(X_train[idx])
            new_y.append(c_int)

        if current_count == max_count:
            continue

        needed = max_count - current_count
        logger.info(
            "INFO: Class %d: current_count=%d, augmenting with %d synthetic samples.",
            c_int,
            current_count,
            needed,
        )

        idxs_array = np.array(idxs, dtype=int)
        is_haarye_class = haarye_class_id is not None and c_int == haarye_class_id

        for _ in range(needed):
            base_idx = int(np.random.choice(idxs_array))
            base_sample = X_train[base_idx]
            aug_sample = random_augment_sample(
                base_sample,
                noise_std=noise_std,
                max_time_shift_steps=max_time_shift_steps,
                is_haarye=is_haarye_class,
            )
            new_X.append(aug_sample)
            new_y.append(c_int)

    new_X_arr = np.stack(new_X).astype(np.float32)
    new_y_arr = np.array(new_y, dtype=np.int64)

    if logger is not None:
        final_counts = {
            int(c): int((new_y_arr == c).sum()) for c in np.unique(new_y_arr)
        }
        logger.info(
            "INFO: [Balanced TRAIN] Final per-class counts (after augmentation):"
        )
        for c_int in sorted(final_counts.keys()):
            logger.info("INFO:   class %d: %d", c_int, final_counts[c_int])

    return new_X_arr, new_y_arr


# ===========================================================================
# Cyclic stratified CV splitter (VAL → TEST → TRAIN rotation)
# ===========================================================================


def build_cyclic_stratified_folds(
    y: np.ndarray,
    label_to_idx: Dict[str, int],
    val_per_class: int,
    test_per_class: int,
    cv_folds: int,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Cyclic stratified folds with per-class control on VAL/TEST samples.

    Constraints:
      - In each fold, VAL and TEST are disjoint.
      - Across all folds, each sample appears in VAL at most once
        and in TEST at most once.
    """
    if val_per_class <= 0:
        raise ValueError("val_per_class must be > 0.")
    if test_per_class < 0:
        raise ValueError("test_per_class must be >= 0.")
    if test_per_class > val_per_class:
        raise ValueError(
            f"test_per_class={test_per_class} cannot be greater than val_per_class={val_per_class} "
            "because VAL/TEST are drawn from the same per-class chunks."
        )

    rng = np.random.RandomState(seed)
    all_indices = np.arange(len(y))

    class_indices: Dict[int, np.ndarray] = {}
    max_folds_per_class: Dict[int, int] = {}

    logger.info(
        "INFO: Cyclic CV: VAL per class per fold = %d, TEST per class per fold = %d",
        val_per_class,
        test_per_class,
    )

    for label_str, label_idx in label_to_idx.items():
        idxs = np.where(y == label_idx)[0]
        rng.shuffle(idxs)
        class_indices[label_idx] = idxs

        max_folds = len(idxs) // val_per_class
        max_folds_per_class[label_idx] = max_folds

    logger.info(
        "INFO: Max possible folds per class (cyclic VAL/TEST, no reuse of VAL/TEST samples):"
    )
    for label_idx, idxs in class_indices.items():
        max_folds = max_folds_per_class[label_idx]
        logger.info(
            "INFO:   %d: max_folds = %d (count = %d)",
            label_idx,
            max_folds,
            len(idxs),
        )

    positive_folds = [v for v in max_folds_per_class.values() if v > 0]
    if not positive_folds:
        raise ValueError("No class has enough samples for even 1 fold.")

    max_possible_folds = min(positive_folds)

    logger.info(
        "INFO: Maximum possible CV folds (global, all classes): %d", max_possible_folds
    )

    if cv_folds > max_possible_folds:
        logger.error(
            "ERROR: Requested cv_folds=%d but maximum possible without reusing "
            "VAL/TEST samples is %d. Reduce cv_folds or val_per_class.",
            cv_folds,
            max_possible_folds,
        )
        raise ValueError(
            f"Requested cv_folds={cv_folds} but maximum possible without reuse is "
            f"{max_possible_folds}. Reduce cv_folds or val_per_class."
        )

    F = cv_folds
    logger.info("INFO: Creating cyclic folds ...")

    folds: List[Dict[str, np.ndarray]] = []

    for f in range(F):
        val_inds: List[int] = []
        test_inds: List[int] = []

        for _, label_idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
            idxs = class_indices[label_idx]
            max_folds_c = max_folds_per_class[label_idx]
            if max_folds_c == 0:
                continue

            chunk_size = val_per_class
            needed = max_folds_c * chunk_size
            idxs_use = idxs[:needed]

            chunks = np.split(idxs_use, max_folds_c)

            if len(chunks[f]) < val_per_class:
                continue
            val_c = chunks[f][:val_per_class]

            test_chunk = chunks[(f + 1) % max_folds_c]
            test_c = test_chunk[:test_per_class] if test_per_class > 0 else []

            val_inds.extend(val_c.tolist())
            if test_per_class > 0:
                test_inds.extend(list(test_c))

        val_inds = np.array(sorted(val_inds), dtype=np.int64)
        test_inds = np.array(sorted(test_inds), dtype=np.int64)

        overlap = np.intersect1d(val_inds, test_inds)
        if overlap.size > 0:
            raise ValueError(
                f"Internal error: VAL and TEST overlap in fold {f + 1} "
                f"(overlap size={overlap.size})."
            )

        mask = np.ones(len(y), dtype=bool)
        mask[val_inds] = False
        mask[test_inds] = False
        train_inds = all_indices[mask]

        folds.append(
            {
                "train": train_inds,
                "val": val_inds,
                "test": test_inds,
            }
        )

        logger.info(
            "INFO: [Cyclic CV] Fold %d built: TRAIN=%d, VAL=%d, TEST=%d",
            f + 1,
            len(train_inds),
            len(val_inds),
            len(test_inds),
        )

    logger.info(
        "INFO: ===== Cyclic stratified CV (folds; VAL->TEST rotation; no reuse of VAL/TEST samples) ====="
    )
    return folds


# ===========================================================================
# Training loop for a single fold
# ===========================================================================


def train_one_fold(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    idx_to_label: Dict[int, str],
    fold_splits: Dict[str, np.ndarray],
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
    dropout: float,
    use_early_stopping: bool,
    early_stopping_patience: int,
    early_stopping_metric: str,
    other_class_weight: float,
    augment_minor: bool,
    use_class_weights: bool,
) -> Dict[str, Any]:
    """
    Train one CV fold and return metrics & best model state.
    Normalization is computed using TRAIN only in this fold and
    applied to TRAIN/VAL/TEST to avoid data leakage.
    """
    train_idx = fold_splits["train"]
    val_idx = fold_splits["val"]
    test_idx = fold_splits["test"]

    n_classes = len(idx_to_label)
    logger.info(
        "INFO: [Fold %d] Sizes: TRAIN=%d, VAL=%d, TEST=%d",
        fold_idx + 1,
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    def log_counts(indices: np.ndarray, name: str) -> None:
        logger.info(
            "INFO: [Fold %d] %s (raw, before normalization/balance) per-class counts:",
            fold_idx + 1,
            name,
        )
        for c in range(n_classes):
            count_c = int((y[indices] == c).sum())
            logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    log_counts(val_idx, "VAL")
    log_counts(test_idx, "TEST")
    logger.info(
        "INFO: [Fold %d] TRAIN (raw, before normalization/balance) per-class counts:",
        fold_idx + 1,
    )
    for c in range(n_classes):
        count_c = int((y[train_idx] == c).sum())
        logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    X_train_raw = X[train_idx].copy()
    y_train_raw = y[train_idx].copy()
    X_val_raw = X[val_idx].copy()
    y_val = y[val_idx].copy()
    X_test_raw = X[test_idx].copy()
    y_test = y[test_idx].copy()

    seq_len = X.shape[1]
    feat_dim = X.shape[2]

    X_train_2d = X_train_raw.reshape(-1, feat_dim)
    mean = X_train_2d.mean(axis=0, keepdims=True)
    std = X_train_2d.std(axis=0, keepdims=True) + 1e-6

    def _normalize_subset(x_subset: np.ndarray) -> np.ndarray:
        x2d = x_subset.reshape(-1, feat_dim)
        x_norm2d = (x2d - mean) / std
        return x_norm2d.reshape(x_subset.shape).astype(np.float32)

    X_train_norm = _normalize_subset(X_train_raw)
    X_val = _normalize_subset(X_val_raw)
    X_test = _normalize_subset(X_test_raw)

    logger.info(
        "INFO: [Fold %d] Applied per-fold normalization using TRAIN statistics "
        "(mean/std per channel).",
        fold_idx + 1,
    )

    haarye_class_id: Optional[int] = None
    for c_idx, name in idx_to_label.items():
        if name.upper() == "HAARYE":
            haarye_class_id = c_idx
            break

    X_train, y_train = build_balanced_train_set(
        X_train=X_train_norm,
        y_train=y_train_raw,
        augment_minor=augment_minor,
        logger=logger,
        noise_std=0.05,
        max_time_shift_steps=2,
        haarye_class_id=haarye_class_id,
    )

    if augment_minor:
        logger.info(
            "INFO: [Fold %d] Data augmentation ENABLED (upsampling minority classes "
            "to largest class with structural + noise/time-shift augmentations).",
            fold_idx + 1,
        )
    else:
        logger.info(
            "INFO: [Fold %d] Data augmentation DISABLED (using downsampling to smallest class).",
            fold_idx + 1,
        )

    logger.info("INFO: [Fold %d] TRAIN per-class counts AFTER balancing:", fold_idx + 1)
    for c in range(n_classes):
        count_c = int((y_train == c).sum())
        logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    logger.info("INFO: [Fold %d] VAL per-class counts (unchanged):", fold_idx + 1)
    for c in range(n_classes):
        count_c = int((y_val == c).sum())
        logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    logger.info("INFO: [Fold %d] TEST per-class counts (unchanged):", fold_idx + 1)
    for c in range(n_classes):
        count_c = int((y_test == c).sum())
        logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    logger.info(
        "INFO: [Fold %d] Using feature dimension per step: %d, sequence length: %d, batch_size=%d",
        fold_idx + 1,
        feat_dim,
        seq_len,
        batch_size,
    )

    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    test_dataset = ClassificationDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    num_classes = len(idx_to_label)
    model = LSTMClassifier(
        feat_dim,
        hidden_size,
        num_layers,
        num_classes,
        bidirectional,
        dropout,
    ).to(device)

    if use_class_weights:
        logger.info(
            "INFO: [Fold %d] Using weighted CrossEntropyLoss (special treatment for OTHER).",
            fold_idx + 1,
        )
        weights = np.ones(num_classes, dtype=np.float32)
        has_other = False
        for idx, name in idx_to_label.items():
            if name.upper() == "OTHER":
                weights[idx] = other_class_weight
                has_other = True
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        if has_other:
            logger.info(
                "INFO: OTHER class found, using weight=%.3f in loss.", other_class_weight
            )
        else:
            logger.info(
                "INFO: OTHER class not found, using uniform weights in weighted loss."
            )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        logger.info(
            "INFO: [Fold %d] Using unweighted CrossEntropyLoss (no class weights).",
            fold_idx + 1,
        )
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def run_epoch(loader: DataLoader, train_mode: bool) -> Tuple[float, float, float]:
        if train_mode:
            model.train()
        else:
            model.eval()

        all_preds: List[int] = []
        all_targets: List[int] = []
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.set_grad_enabled(train_mode):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                if train_mode:
                    optimizer.zero_grad()

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                if train_mode:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)

                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_x.size(0)

                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(batch_y.detach().cpu().tolist())

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)

        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_targets, all_preds):
            confusion[t, p] += 1
        _, macro_f1 = compute_per_class_metrics(confusion, idx_to_label)

        return avg_loss, acc, macro_f1

    best_state = copy.deepcopy(model.state_dict())
    if early_stopping_metric.lower() == "macro_f1":
        best_metric = -math.inf
    else:
        best_metric = math.inf

    best_val_loss = math.inf
    best_val_macro_f1 = 0.0
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, _ = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc, val_macro_f1 = run_epoch(val_loader, train_mode=False)

        logger.info(
            "INFO: Fold %d | Epoch %d/%d: Train Loss = %.4f, Train Acc = %.2f%% | "
            "Val Loss = %.4f, Val Acc = %.2f%%, Val Macro-F1 = %.2f%%",
            fold_idx + 1,
            epoch,
            num_epochs,
            train_loss,
            train_acc * 100.0,
            val_loss,
            val_acc * 100.0,
            val_macro_f1 * 100.0,
        )

        if early_stopping_metric.lower() == "macro_f1":
            current_metric = val_macro_f1
            improved = current_metric > best_metric + 1e-6
        else:
            current_metric = val_loss
            improved = current_metric < best_metric - 1e-6

        if improved:
            best_metric = current_metric
            best_val_loss = val_loss
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if use_early_stopping and epochs_without_improve >= early_stopping_patience:
            logger.info(
                "INFO: Fold %d | Early stopping at epoch %d (no improvement in %s for %d epochs).",
                fold_idx + 1,
                epoch,
                early_stopping_metric,
                early_stopping_patience,
            )
            break

    model.load_state_dict(best_state)

    # TEST evaluation
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch_y.cpu().tolist())

    test_loss = total_loss / max(total_samples, 1)
    test_acc = total_correct / max(total_samples, 1)

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        confusion[t, p] += 1

    per_class_metrics, test_macro_f1 = compute_per_class_metrics(
        confusion, idx_to_label
    )

    # Macro precision / recall for this fold
    prec_list = [m["precision"] for m in per_class_metrics.values()]
    rec_list = [m["recall"] for m in per_class_metrics.values()]
    test_macro_precision = float(np.mean(prec_list)) if prec_list else 0.0
    test_macro_recall = float(np.mean(rec_list)) if rec_list else 0.0

    logger.info("INFO: Fold %d — Test Accuracy: %.2f%%", fold_idx + 1, test_acc * 100.0)
    log_confusion_matrix(confusion, idx_to_label)

    logger.info("INFO: Fold %d per-class metrics:", fold_idx + 1)
    for c in range(n_classes):
        m = per_class_metrics[c]
        logger.info(
            "INFO:   %s: Prec=%.2f%%, Rec=%.2f%%, F1=%.2f%%, Support=%d",
            idx_to_label[c],
            m["precision"] * 100.0,
            m["recall"] * 100.0,
            m["f1"] * 100.0,
            int(m["support"]),
        )

    logger.info(
        "INFO: [Fold %d] Test Loss = %.4f, Test Acc = %.2f%%, "
        "Test Macro-F1 = %.2f%%, Test Macro-Precision = %.2f%%, Test Macro-Recall = %.2f%%",
        fold_idx + 1,
        test_loss,
        test_acc * 100.0,
        test_macro_f1 * 100.0,
        test_macro_precision * 100.0,
        test_macro_recall * 100.0,
    )

    result = {
        "fold_idx": fold_idx,
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_macro_precision": test_macro_precision,
        "test_macro_recall": test_macro_recall,
        "test_confusion": confusion,
        "per_class_metrics": per_class_metrics,
    }
    return result


# ===========================================================================
# Public API: train_cross_validation
# ===========================================================================


def train_cross_validation(
    data_path: str,
    patient_id: str = "Unknown",
    cv_folds: int = 5,
    val_per_class: int = 3,
    test_per_class: int = 3,
    num_epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 128,
    num_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.3,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 15,
    early_stopping_metric: str = "macro_f1",
    other_class_weight: float = 0.8,
    augment_minor: bool = False,
    use_class_weights: bool = False,
    pool_size: int = 50,
    force_preproc: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
    log_dir: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Main entry point used by main.py.
    """
    if "num_folds" in kwargs:
        old = kwargs.pop("num_folds")
        logger.info(
            "INFO: Received num_folds=%s in kwargs, overriding cv_folds=%d", old, cv_folds
        )
        cv_folds = int(old)

    for k in kwargs.keys():
        logger.info(
            "INFO: Ignoring extra argument passed to train_cross_validation: %s", k
        )

    del log_dir

    if force_preproc:
        logger.info(
            "INFO: force_preproc=True was requested, but preprocessing must be handled "
            "outside train_classification.py (this module expects a ready .npy at data_path)."
        )

    set_seed(seed)

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    logger.info("INFO: Using device: %s", device_t)
    logger.info("INFO: Requested cv_folds=%d", cv_folds)
    logger.info("INFO: Using pool_size=%d for temporal pooling.", pool_size)

    X, y, idx_to_label, label_to_idx = load_classification_data(
        data_path, pool_size=pool_size
    )

    # --- Chance level (uniform over classes) -----------------------------  # <<< NEW
    n_classes = len(idx_to_label)                                          # <<< NEW
    chance_level = 1.0 / max(n_classes, 1)                                 # <<< NEW
    logger.info(                                                           # <<< NEW
        "INFO: [Chance] Chance-level accuracy (1/%d classes): %.2f%%",     # <<< NEW
        n_classes,                                                         # <<< NEW
        chance_level * 100.0,                                              # <<< NEW
    )                                                                      # <<< NEW

    unique, counts = np.unique(y, return_counts=True)
    majority_idx = int(unique[np.argmax(counts)])
    majority_name = idx_to_label[majority_idx]
    majority_acc = counts.max() / len(y)
    logger.info(
        "INFO: ===== LABEL / INFORMATION QUALITY CHECK (simple baseline) ====="
    )
    logger.info(
        "INFO: [Baseline] Majority-class baseline accuracy: %.2f%% (class '%s', count=%d / %d)",
        majority_acc * 100.0,
        majority_name,
        counts.max(),
        len(y),
    )
    logger.info("INFO: ===== END OF LABEL / INFORMATION QUALITY CHECK =====")

    folds = build_cyclic_stratified_folds(
        y=y,
        label_to_idx=label_to_idx,
        val_per_class=val_per_class,
        test_per_class=test_per_class,
        cv_folds=cv_folds,
        seed=seed,
    )

    logger.info("INFO: Running cross-validation for patient_id=%s", patient_id)

    all_results: List[Dict[str, Any]] = []
    n_classes = len(idx_to_label)
    overall_confusion = np.zeros((n_classes, n_classes), dtype=int)

    for f_idx, split in enumerate(folds):
        result = train_one_fold(
            fold_idx=f_idx,
            X=X,
            y=y,
            idx_to_label=idx_to_label,
            fold_splits=split,
            device=device_t,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            other_class_weight=other_class_weight,
            augment_minor=augment_minor,
            use_class_weights=use_class_weights,
        )
        all_results.append(result)
        overall_confusion += result["test_confusion"].astype(int)

        logger.info(
            "INFO: Fold %d: best val loss=%.4f, best val macro-F1=%.2f%%, "
            "test acc=%.2f%%, test macro-F1=%.2f%%, test loss=%.4f",
            f_idx + 1,
            result["best_val_loss"],
            result["best_val_macro_f1"] * 100.0,
            result["test_acc"] * 100.0,
            result["test_macro_f1"] * 100.0,
            result["test_loss"],
        )

    logger.info("INFO: ========== CLASSIFICATION OVERALL RESULTS ========== ")
    for r in all_results:
        logger.info(
            "INFO: Fold %d: best val loss=%.4f, best val macro-F1=%.2f%%, "
            "test acc=%.2f%%, test macro-F1=%.2f%%, test loss=%.4f",
            r["fold_idx"] + 1,
            r["best_val_loss"],
            r["best_val_macro_f1"] * 100.0,
            r["test_acc"] * 100.0,
            r["test_macro_f1"] * 100.0,
            r["test_loss"],
        )

    test_accs = [r["test_acc"] for r in all_results]
    macro_f1s = [r["test_macro_f1"] for r in all_results]
    macro_precs = [r["test_macro_precision"] for r in all_results]
    macro_recs = [r["test_macro_recall"] for r in all_results]

    mean_test_acc = float(np.mean(test_accs)) if len(test_accs) > 0 else 0.0
    max_test_acc = float(np.max(test_accs)) if len(test_accs) > 0 else 0.0
    mean_macro_f1 = float(np.mean(macro_f1s)) if len(macro_f1s) > 0 else 0.0
    max_macro_f1 = float(np.max(macro_f1s)) if len(macro_f1s) > 0 else 0.0

    mean_macro_prec = float(np.mean(macro_precs)) if len(macro_precs) > 0 else 0.0
    max_macro_prec = float(np.max(macro_precs)) if len(macro_precs) > 0 else 0.0
    mean_macro_rec = float(np.mean(macro_recs)) if len(macro_recs) > 0 else 0.0
    max_macro_rec = float(np.max(macro_recs)) if len(macro_recs) > 0 else 0.0

    logger.info(
        "INFO: Mean test accuracy over %d folds: %.2f%%",
        len(all_results),
        mean_test_acc * 100.0,
    )

    logger.info(
        "INFO: ========= OVERALL TEST CONFUSION MATRIX (all folds combined) ========="
    )
    log_confusion_matrix(overall_confusion, idx_to_label)

    overall_metrics, overall_macro_f1 = compute_per_class_metrics(
        overall_confusion, idx_to_label
    )
    logger.info("INFO: Overall per-class metrics:")
    for c in range(n_classes):
        m = overall_metrics[c]
        logger.info(
            "INFO:   %s: Prec=%.2f%%, Rec=%.2f%%, F1=%.2f%%, Support=%d",
            idx_to_label[c],
            m["precision"] * 100.0,
            m["recall"] * 100.0,
            m["f1"] * 100.0,
            int(m["support"]),
        )
    logger.info(
        "INFO: Overall macro-F1 (all folds combined): %.2f%%",
        overall_macro_f1 * 100.0,
    )

    # ========= T-TEST vs baseline accuracy =========
    t_statistic: Optional[float] = None
    t_p_value: Optional[float] = None
    if len(test_accs) > 1:
        try:
            from scipy.stats import ttest_1samp

            t_stat, p_val = ttest_1samp(test_accs, popmean=majority_acc)
            t_statistic = float(t_stat)
            t_p_value = float(p_val)
            logger.info(
                "INFO: T-TEST vs baseline accuracy (%.2f%%): t=%.4f, p=%.4g",
                majority_acc * 100.0,
                t_stat,
                p_val,
            )
            logger.info(  # <<< NEW
                "INFO: p-test (p-value vs baseline) = %.4g", p_val  # <<< NEW
            )  # <<< NEW
        except Exception as e:
            logger.info(
                "INFO: Could not compute T-TEST (scipy missing or error: %s).", str(e)
            )
    else:
        logger.info(
            "INFO: Not enough folds (%d) to compute a meaningful T-TEST.",
            len(test_accs),
        )

    # ===== FINAL SUMMARY FOR TABLE (per patient) =====
    logger.info("INFO: ===== SUMMARY FOR TABLE =====")
    logger.info("INFO: Patient: %s", patient_id)
    logger.info(
        "INFO: Chance level = %.2f%% (1/%d classes)",           # <<< NEW
        chance_level * 100.0,                                   # <<< NEW
        len(idx_to_label),                                      # <<< NEW
    )                                                           # <<< NEW
    logger.info(
        "INFO: Avg Accuracy = %.2f%% | Max Accuracy = %.2f%%",
        mean_test_acc * 100.0,
        max_test_acc * 100.0,
    )
    logger.info(
        "INFO: Avg F1 = %.2f%% | Max F1 = %.2f%%",
        mean_macro_f1 * 100.0,
        max_macro_f1 * 100.0,
    )
    logger.info(
        "INFO: Avg Precision = %.2f%% | Max Precision = %.2f%%",
        mean_macro_prec * 100.0,
        max_macro_prec * 100.0,
    )
    logger.info(
        "INFO: Avg Recall = %.2f%% | Max Recall = %.2f%%",
        mean_macro_rec * 100.0,
        max_macro_rec * 100.0,
    )
    if t_statistic is not None:
        logger.info(
            "INFO: Avg T Test = %.4f | Max T test = %.4f | p-test = %.4g",  # <<< NEW (label p-test)
            t_statistic,
            t_statistic,
            t_p_value if t_p_value is not None else float("nan"),
        )
    else:
        logger.info(
            "INFO: Avg T Test = N/A | Max T test = N/A (T-test not computed)."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--patient_id", type=str, default="Unknown")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--val_per_class", type=int, default=3)
    parser.add_argument("--test_per_class", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_metric", type=str, default="macro_f1")
    parser.add_argument("--other_class_weight", type=float, default=0.5)
    parser.add_argument("--augment_minor", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--force_preproc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_cross_validation(
        data_path=args.data_path,
        patient_id=args.patient_id,
        cv_folds=args.cv_folds,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        use_early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        other_class_weight=args.other_class_weight,
        augment_minor=args.augment_minor,
        use_class_weights=args.use_class_weights,
        pool_size=args.pool_size,
        force_preproc=args.force_preproc,
        seed=args.seed,
    )
