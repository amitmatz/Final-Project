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
    pool_size: int = 125,
    class_map: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[str, int]]:
    """
    Load classification data from the .npy produced in preprocessing.

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

    # Global Z-score per channel
    X_2d = X.reshape(-1, feat_dim)
    mean = X_2d.mean(axis=0, keepdims=True)
    std = X_2d.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X_2d - mean) / std
    X_norm = X_norm.reshape(X.shape)

    logger.info(
        "INFO: Applied global per-channel Z-score normalization after temporal pooling."
    )

    logger.info("INFO: [LOAD] Final class counts after normalization:")
    total_samples = X_norm.shape[0]
    for idx, name in idx_to_label.items():
        count = int((y == idx).sum())
        logger.info("INFO:   %s: %d", name, count)
    logger.info("INFO: Total samples: %d", total_samples)

    logger.info("INFO: Label distribution after loading (per-class counts):")
    for idx, name in idx_to_label.items():
        count = int((y == idx).sum())
        logger.info("INFO:   %s: %d", name, count)

    return X_norm, y, idx_to_label, label_to_idx


# ===========================================================================
# Data augmentation
# ===========================================================================


def augment_signals_gaussian(x: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """
    Add small Gaussian noise to signals.
    x: (N, seq_len, feat_dim)
    """
    noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x + noise


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
    Build cyclic stratified folds with per-class control on VAL/TEST samples.

    For each class c:
      - We compute the maximum number of folds we can create
        without reusing VAL samples of that class.
      - If the requested cv_folds exceeds the global maximum possible
        (across all classes), we raise an error with an explanation.
      - Within the allowed number of folds, chunks for VAL/TEST are disjoint,
        so no sample is used twice in VAL or twice in TEST.
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
    n_classes = len(label_to_idx)
    all_indices = np.arange(len(y))

    class_indices: Dict[int, np.ndarray] = {}
    max_folds_per_class: Dict[int, int] = {}

    logger.info(
        "INFO: Cyclic CV: VAL per class per fold = %d, TEST per class per fold = %d",
        val_per_class,
        test_per_class,
    )

    # 1) Per-class max folds (no reuse of VAL samples)
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

    # אם ביקשנו יותר מהאפשרי – נעצור עם שגיאה ברורה ולא נחתוך בשקט
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

    # 2) Build folds
    for f in range(F):
        val_inds: List[int] = []
        test_inds: List[int] = []

        # iterate classes in index order
        for _, label_idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
            idxs = class_indices[label_idx]
            max_folds_c = max_folds_per_class[label_idx]
            if max_folds_c == 0:
                continue

            chunk_size = val_per_class
            needed = max_folds_c * chunk_size
            idxs_use = idxs[:needed]

            # split idxs_use into max_folds_c equal chunks
            chunks = np.split(idxs_use, max_folds_c)

            # VAL for this fold from chunk f
            if len(chunks[f]) < val_per_class:
                continue
            val_c = chunks[f][:val_per_class]

            # TEST for this fold from next chunk cyclically
            test_chunk = chunks[(f + 1) % max_folds_c]
            test_c = test_chunk[:test_per_class] if test_per_class > 0 else []

            val_inds.extend(val_c.tolist())
            if test_per_class > 0:
                test_inds.extend(test_c.tolist())

        val_inds = np.array(sorted(val_inds), dtype=np.int64)
        test_inds = np.array(sorted(test_inds), dtype=np.int64)

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
) -> Dict[str, Any]:
    """
    Train one CV fold and return metrics & best model state.
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
            "INFO: [Fold %d] %s (raw, before training) per-class counts:",
            fold_idx + 1,
            name,
        )
        for c in range(n_classes):
            count_c = int((y[indices] == c).sum())
            logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    log_counts(val_idx, "VAL")
    log_counts(test_idx, "TEST")
    logger.info(
        "INFO: [Fold %d] TRAIN (raw, before balance/augmentation) per-class counts:",
        fold_idx + 1,
    )
    for c in range(n_classes):
        count_c = int((y[train_idx] == c).sum())
        logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    # Balance TRAIN by DOWNsampling to smallest class (after removing VAL/TEST)
    train_labels = y[train_idx]
    per_class_indices: Dict[int, np.ndarray] = {}
    for c in range(n_classes):
        per_class_indices[c] = train_idx[train_labels == c]

    min_count = min(len(v) for v in per_class_indices.values())
    logger.info(
        "INFO: Balancing TRAIN by DOWNsampling to smallest class: min_count=%d, raw_counts=%s",
        min_count,
        {c: len(v) for c, v in per_class_indices.items()},
    )

    rng = np.random.RandomState(42 + fold_idx)
    balanced_train_indices: List[int] = []
    for c in range(n_classes):
        idxs = per_class_indices[c]
        if len(idxs) > min_count:
            chosen = rng.choice(idxs, size=min_count, replace=False)
        else:
            chosen = idxs
        balanced_train_indices.extend(chosen.tolist())

    balanced_train_indices = np.array(sorted(balanced_train_indices), dtype=np.int64)

    logger.info("INFO: [Fold %d] TRAIN per-class counts AFTER balancing:", fold_idx + 1)
    for c in range(n_classes):
        count_c = int((y[balanced_train_indices] == c).sum())
        logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    logger.info("INFO: [Fold %d] VAL per-class counts (unchanged):", fold_idx + 1)
    for c in range(n_classes):
        count_c = int((y[val_idx] == c).sum())
        logger.info("INFO:   %s: %d", idx_to_label[c], count_c)

    # Extract data
    X_train = X[balanced_train_indices].copy()
    y_train = y[balanced_train_indices].copy()
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Data augmentation (optional) – keep class balance
    if augment_minor:
        logger.info(
            "INFO: [Fold %d] Data augmentation ENABLED (Gaussian noise on non-OTHER classes).",
            fold_idx + 1,
        )
        for c_idx, name in idx_to_label.items():
            if name.upper() == "OTHER":
                continue
            mask_c = y_train == c_idx
            if mask_c.sum() > 0:
                X_train[mask_c] = augment_signals_gaussian(X_train[mask_c])
    else:
        logger.info("INFO: [Fold %d] Augmentation DISABLED for this fold.", fold_idx + 1)

    seq_len = X.shape[1]
    feat_dim = X.shape[2]

    logger.info(
        "INFO: [Fold %d] Using feature dimension per step: %d, sequence length: %d, batch_size=%d",
        fold_idx + 1,
        feat_dim,
        seq_len,
        batch_size,
    )

    # Dataset and loaders
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

    # Model
    num_classes = len(idx_to_label)
    # חשוב: קריאה במיקום בלבד כדי להתאים גם ל-model.py שלך
    model = LSTMClassifier(
        feat_dim,
        hidden_size,
        num_layers,
        num_classes,
        bidirectional,
        dropout,
    ).to(device)

    logger.info(
        "INFO: [Fold %d] Using weighted CrossEntropyLoss (special treatment for OTHER).",
        fold_idx + 1,
    )

    # Special treatment for OTHER class via loss weight
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
        logger.info("INFO: OTHER class not found, using uniform weights in loss.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        best_metric = -math.inf  # maximize
    else:
        best_metric = math.inf  # minimize (loss)

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

    # Load best state
    model.load_state_dict(best_state)

    # Evaluate on TEST
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
        "INFO: [Fold %d] Test Loss = %.4f, Test Acc = %.2f%%, Test Macro-F1 = %.2f%%",
        fold_idx + 1,
        test_loss,
        test_acc * 100.0,
        test_macro_f1 * 100.0,
    )

    result = {
        "fold_idx": fold_idx,
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
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
    val_per_class: int = 1,
    test_per_class: int = 1,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 128,
    num_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.3,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 20,
    early_stopping_metric: str = "macro_f1",
    other_class_weight: float = 0.5,
    augment_minor: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
    log_dir: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Main entry point used by main.py.

    - LSTM-based classifier.
    - Cyclic CV: samples move VAL -> TEST -> TRAIN without reuse of VAL/TEST.
    - TRAIN is balanced by downsampling to the smallest class (after removing VAL/TEST).
    - Optional data augmentation (Gaussian noise) with augment_minor.
    - Macro-F1 is used as early stopping metric by default.
    - OTHER class handled via loss weight.
    """
    # תמיכה לאחור: אם main עדיין מעביר num_folds – נשתמש בו כ-cv_folds
    if "num_folds" in kwargs:
        old = kwargs.pop("num_folds")
        logger.info("INFO: Received num_folds=%s in kwargs, overriding cv_folds=%d", old, cv_folds)
        cv_folds = int(old)

    # ignore any remaining extra args
    for k in kwargs.keys():
        logger.info(
            "INFO: Ignoring extra argument passed to train_cross_validation: %s", k
        )

    del log_dir  # currently not used, kept for API compatibility

    set_seed(seed)

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    logger.info("INFO: Using device: %s", device_t)
    logger.info("INFO: Requested cv_folds=%d", cv_folds)

    # Load data
    X, y, idx_to_label, label_to_idx = load_classification_data(data_path)

    # Baseline: majority class
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

    # Build folds (with error if cv_folds too large)
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

    # Summary
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
    mean_test_acc = float(np.mean(test_accs)) if len(test_accs) > 0 else 0.0
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

    # ========= T-TEST (מול baseline accuracy) =========
    if len(test_accs) > 1:
        try:
            from scipy.stats import ttest_1samp

            t_stat, p_val = ttest_1samp(test_accs, popmean=majority_acc)
            logger.info(
                "INFO: T-TEST vs baseline accuracy (%.2f%%): t=%.4f, p=%.4g",
                majority_acc * 100.0,
                t_stat,
                p_val,
            )
        except Exception as e:
            logger.info(
                "INFO: Could not compute T-TEST (scipy missing or error: %s).", str(e)
            )
    else:
        logger.info(
            "INFO: Not enough folds (%d) to compute a meaningful T-TEST.",
            len(test_accs),
        )


if __name__ == "__main__":
    # Optional debug entry point (not used by main.py)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--patient_id", type=str, default="Unknown")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--val_per_class", type=int, default=1)
    parser.add_argument("--test_per_class", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
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
        seed=args.seed,
    )
