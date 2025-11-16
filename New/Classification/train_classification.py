# Classification/train_classification.py
# Unified training script with TrainConfig and train() entry point.
# Code: English-only (per user preference).

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# Utility: seeding for reproducibility
# -------------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------
# Config
# -------------------------------
@dataclass
class TrainConfig:
    # Required paths
    data_path: str
    splits_path: str
    # CV
    cv_folds: int = 5
    # Optimization
    batch_size: int = 64
    max_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20  # early stopping patience
    # Model
    hidden_channels: int = 64
    conv_kernel: int = 7
    pool_kernel: int = 4
    dropout: float = 0.2
    # Channels (optional filtering)
    keep_channels: Optional[List[int]] = None
    # Misc
    num_workers: int = 0
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Logging
    verbose: bool = True


# -------------------------------
# Data loading (robust to formats)
# -------------------------------
def _as_numpy(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the classification dataset from a flexible .npy format.

    Supports:
    1) np.save dict-like: {'X': (N,T,C), 'y': (N,)}
       (saved as a 0-d object array or 1-d object array of size 1)
    2) A list/array of dicts with keys: 'signals' (T,C), 'label' (int/str)
    3) tuple-like (X, y)
    4) ndarray X only (N,T,C) with sidecar *_labels.npy
    """
    arr = np.load(data_path, allow_pickle=True)

    # ---- Case 2: array/list of dicts ----
    if isinstance(arr, (list, np.ndarray)):
        # numpy object array with multiple dicts
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim >= 1 and arr.size > 1:
            first = arr[0]
            if isinstance(first, dict) and ("signals" in first and "label" in first):
                X_list, y_list = [], []
                for it in arr:
                    sig = it.get("signals")
                    lab = it.get("label")
                    if sig is None or lab is None:
                        raise ValueError("Each item must contain 'signals' and 'label'.")
                    X_list.append(np.asarray(sig))
                    # DO NOT cast to int here; allow str labels and map later
                    y_list.append(lab)
                X = np.stack(X_list, axis=0)
                y = np.asarray(y_list)
                return X, y

        # Python list of dicts
        if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
            X_list, y_list = [], []
            for it in arr:
                sig = it.get("signals")
                lab = it.get("label")
                if sig is None or lab is None:
                    raise ValueError("Each item must contain 'signals' and 'label'.")
                X_list.append(np.asarray(sig))
                y_list.append(lab)
            X = np.stack(X_list, axis=0)
            y = np.asarray(y_list)
            return X, y

    # ---- Case 1: dict-like saved as scalar object array or 1-element object array ----
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        obj = None
        if arr.shape == ():  # 0-d object array
            obj = arr.item()
        elif arr.ndim == 1 and arr.size == 1:  # 1-d array with single object
            obj = arr[0]
        if isinstance(obj, dict):
            X = obj.get("X")
            y = obj.get("y")
            if X is None or y is None:
                raise ValueError("Dict dataset must contain keys 'X' and 'y'.")
            return np.asarray(X), np.asarray(y)

    # ---- Case 3: tuple-like (X, y) ----
    if isinstance(arr, tuple) and len(arr) == 2:
        X, y = arr
        return np.asarray(X), np.asarray(y)

    # ---- Case 4: raw ndarray X only, expect sidecar labels ----
    if isinstance(arr, np.ndarray) and arr.ndim == 3:
        side = Path(data_path).with_name(Path(data_path).stem + "_labels.npy")
        if not side.exists():
            raise ValueError(
                f"Found ndarray X with shape {arr.shape} but no labels. "
                f"Expected sidecar file: {side.name}"
            )
        y = np.load(side, allow_pickle=True)
        return np.asarray(arr), np.asarray(y)

    # Last-resort: maybe scalar object with list/tuple
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            X, y = obj
            return np.asarray(X), np.asarray(y)

    raise ValueError(
        f"Unsupported dataset format in {data_path!r}. "
        f"Type={type(arr)}, shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}"
    )


def _apply_keep_channels(X: np.ndarray, keep_channels: Optional[Sequence[int]]) -> np.ndarray:
    if not keep_channels:
        return X
    keep = np.asarray(keep_channels, dtype=int)
    return X[:, :, keep]


# -------------------------------
# Splits: per your exact spec
#
# For each fold:
#   - exactly 1 sample per class -> VAL
#   - exactly 1 sample per class -> TEST
#   - TRAIN: same number of samples per class:
#         train_per_class = min_class_count - 2
#     (i.e. limited by the smallest class, after taking one for VAL and one for TEST)
#
# Classes with more samples are randomly sub-sampled (with seed) so that
# train count is identical across classes.
# VAL/TEST indices rotate across folds (CV).
# -------------------------------
def build_strict_per_class_splits(
    y: np.ndarray, num_folds: int, seed: int
) -> List[Dict[str, List[int]]]:
    """
    Build cross-validation folds according to the project spec:

    - Let classes = unique(y).
    - Let count[c] = number of samples in class c.
    - Let N_min = min_c count[c].

    For every fold:
      * VAL: 1 sample per class.
      * TEST: 1 sample per class.
      * TRAIN: same number of samples per class:
          train_per_class = N_min - 2
        (if N_min < 3 → error).

    For classes with more samples than N_min:
      TRAIN samples are sub-sampled randomly (with a fixed seed)
      from the remaining examples (not used as VAL/TEST in that fold).

    VAL/TEST samples rotate across folds (so we still have CV behaviour).
    """
    set_seed(seed)
    classes = np.unique(y).tolist()

    # bucket indices per class
    per_class: Dict[int, List[int]] = {c: [] for c in classes}
    for idx, lab in enumerate(y):
        per_class[lab].append(idx)

    # per-class counts and minimum
    counts = {int(c): len(per_class[c]) for c in classes}
    n_min = min(counts.values())

    if n_min < 3:
        raise ValueError(
            "Not enough samples per class for the requested split. "
            "Need at least 3 samples per class (1 train + 1 val + 1 test) but got "
            f"per-class counts = {counts}"
        )

    train_per_class = n_min - 2  # minus 1 for VAL and 1 for TEST

    # shuffle deterministically per class
    for c in classes:
        random.shuffle(per_class[c])

    # prepare folds
    folds: List[Dict[str, List[int]]] = [{"train": [], "val": [], "test": []} for _ in range(num_folds)]

    # fill each fold
    for c in classes:
        idxs = per_class[c]  # indices of class c
        nc = len(idxs)

        for f in range(num_folds):
            # --- pick VAL / TEST by rotating positions ---
            v_pos = f % nc
            t_pos = (f + 1) % nc
            if t_pos == v_pos:
                t_pos = (t_pos + 1) % nc  # safety; with nc >= 3 this is fine

            v_idx = idxs[v_pos]
            t_idx = idxs[t_pos]

            folds[f]["val"].append(v_idx)
            folds[f]["test"].append(t_idx)

            # --- remaining candidates for TRAIN in this fold (for this class) ---
            remaining = [i for i in idxs if i not in (v_idx, t_idx)]

            if len(remaining) < train_per_class:
                # Should not happen given nc >= n_min, but guard anyway
                chosen = remaining
            else:
                chosen = random.sample(remaining, train_per_class)

            folds[f]["train"].extend(chosen)

    # final cleanup: ensure no overlaps and sort
    for split in folds:
        val_set = set(split["val"])
        test_set = set(split["test"])
        train_set = set(split["train"]) - val_set - test_set

        split["train"] = sorted(train_set)
        split["val"] = sorted(val_set)
        split["test"] = sorted(test_set)

    return folds


# -------------------------------
# Dataset
# -------------------------------
class WindowsDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: Sequence[int],
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        augment: bool = False,
    ):
        """
        X: (N, T, C), y: (N,)
        If norm_stats=(mean,std) in shape (C,), normalization is applied channel-wise.
        """
        self.X = X[indices]  # (n, T, C)
        self.y = y[indices].astype(np.int64)
        self.augment = augment

        if norm_stats is not None:
            mean, std = norm_stats
            self.X = (self.X - mean[None, None, :]) / (std[None, None, :] + 1e-8)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        xi = self.X[i]  # (T, C)
        yi = self.y[i]
        if self.augment:
            xi = xi + np.random.normal(0.0, 0.01, size=xi.shape).astype(xi.dtype)
        xi = torch.tensor(xi, dtype=torch.float32).transpose(0, 1)  # (C, T)
        yi = torch.tensor(yi, dtype=torch.long)
        return xi, yi


# -------------------------------
# Model: small but strong 1D-CNN
# -------------------------------
class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, hidden: int = 64, k: int = 7, p: int = 4, drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=p),

            nn.Conv1d(hidden, hidden * 2, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=p),

            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):  # x: (B,C,T)
        h = self.net(x)          # (B, 2H, T')
        h = h.mean(dim=-1)       # GAP -> (B, 2H)
        h = self.dropout(h)
        return self.head(h)


# -------------------------------
# Metrics
# -------------------------------
@torch.no_grad()
def confusion_matrix(pred: np.ndarray, true: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    num_classes = cm.shape[0]
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s) * 100.0)


# -------------------------------
# Train helpers
# -------------------------------
def make_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (counts * n_classes)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(
    cfg: TrainConfig,
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    split: Dict[str, List[int]],
    class_names: Optional[List[str]] = None,
) -> Tuple[float, float, np.ndarray]:
    device = cfg.device

    tr_idx = split["train"]
    va_idx = split["val"]
    te_idx = split["test"]

    # Normalization stats on TRAIN only
    X_tr = X[tr_idx]
    mean = X_tr.mean(axis=(0, 1))  # (C,)
    std = X_tr.std(axis=(0, 1)) + 1e-8

    ds_tr = WindowsDataset(X, y, tr_idx, norm_stats=(mean, std), augment=True)
    ds_va = WindowsDataset(X, y, va_idx, norm_stats=(mean, std), augment=False)
    ds_te = WindowsDataset(X, y, te_idx, norm_stats=(mean, std), augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_va = DataLoader(ds_va, batch_size=max(1, len(va_idx)), shuffle=False, num_workers=cfg.num_workers)
    dl_te = DataLoader(ds_te, batch_size=max(1, len(te_idx)), shuffle=False, num_workers=cfg.num_workers)

    in_ch = X.shape[2]
    n_classes = int(np.max(y)) + 1

    model = CNN1D(
        in_ch=in_ch,
        n_classes=n_classes,
        hidden=cfg.hidden_channels,
        k=cfg.conv_kernel,
        p=cfg.pool_kernel,
        drop=cfg.dropout,
    ).to(device)

    class_w = make_class_weights(y[tr_idx], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6, verbose=False)

    best_va_f1 = -1.0
    best_state = None
    no_improve = 0

    if cfg.verbose:
        print(f"DEBUG: [Fold {fold_idx+1}] TRAIN size={len(tr_idx)}  VAL size={len(va_idx)}  TEST size={len(te_idx)}")

    for epoch in range(1, cfg.max_epochs + 1):
        # --- Train ---
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validate ---
        model.eval()
        va_preds, va_trues = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                va_preds.append(pred)
                va_trues.append(yb.numpy())
        if va_preds:
            va_pred = np.concatenate(va_preds)
            va_true = np.concatenate(va_trues)
            cm_va = confusion_matrix(va_pred, va_true, n_classes)
            va_f1 = macro_f1_from_confusion(cm_va)
        else:
            va_f1 = 0.0

        scheduler.step(va_f1)

        if cfg.verbose:
            print(f"DEBUG: Epoch {epoch}: Val Macro-F1 = {va_f1:.2f}%")

        improved = va_f1 > best_va_f1 + 1e-6
        if improved:
            best_va_f1 = va_f1
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                if cfg.verbose:
                    print(f"DEBUG: Early stopping at epoch {epoch} (no improvement in {cfg.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test ---
    model.eval()
    te_preds, te_trues = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            te_preds.append(pred)
            te_trues.append(yb.numpy())
    te_pred = np.concatenate(te_preds)
    te_true = np.concatenate(te_trues)

    cm_te = confusion_matrix(te_pred, te_true, n_classes)
    acc = float((te_pred == te_true).mean() * 100.0)
    f1 = macro_f1_from_confusion(cm_te)

    if cfg.verbose:
        print(f"INFO: Fold {fold_idx+1} — Test Accuracy: {acc:.2f}%  |  Macro-F1: {f1:.2f}%")
        print("INFO: Confusion Matrix (fold) [rows=true, cols=pred]:")

        names = class_names if class_names else [str(i) for i in range(n_classes)]
        header = "      " + "".join([f"{n:>8s}" for n in names])
        print("INFO:", header)
        for r in range(n_classes):
            row_lab = names[r]
            row = f"{row_lab:>6s}" + "".join([f"{cm_te[r, c]:>8d}" for c in range(n_classes)])
            print("INFO:", row)

    return acc, f1, cm_te


# -------------------------------
# Public API
# -------------------------------
def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    # Load data
    if not os.path.isfile(cfg.data_path):
        raise FileNotFoundError(f"Data file not found: {cfg.data_path}")
    X, y_raw = _as_numpy(cfg.data_path)  # X:(N,T,C) y_raw:(N,)
    if X.ndim != 3:
        raise ValueError(f"Expected X with 3 dims (N,T,C); got shape {X.shape}")
    if y_raw.ndim != 1 or y_raw.shape[0] != X.shape[0]:
        raise ValueError(f"Labels mismatch. y shape {y_raw.shape}, X shape {X.shape}")

    # Optional channel selection BEFORE normalization
    X = _apply_keep_channels(X, cfg.keep_channels)

    # Map string labels to ints (deterministic order)
    class_names: Optional[List[str]] = None
    if y_raw.dtype.kind in {"U", "S", "O"}:  # unicode/bytes/object -> treat as strings
        unique = np.unique(y_raw).tolist()
        # Stable order: sort by name for determinism
        unique = sorted(unique)
        label_to_int = {lab: i for i, lab in enumerate(unique)}
        y = np.array([label_to_int[lab] for lab in y_raw], dtype=np.int64)
        class_names = unique
    else:
        y = y_raw.astype(np.int64)

    N, T, C = X.shape
    num_classes = int(np.max(y)) + 1

    # Debug: show class counts before balancing
    if cfg.verbose:
        uniq, cnts = np.unique(y, return_counts=True)
        print("INFO: Class counts (original dataset):")
        for lab, cnt in zip(uniq, cnts):
            name = class_names[lab] if class_names else str(lab)
            print(f"INFO:   {name}: {cnt}")

    # Build folds according to custom per-class spec
    folds = build_strict_per_class_splits(y, cfg.cv_folds, cfg.seed)

    # Aggregate metrics
    accs, f1s = [], []
    summed_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    print("INFO:")
    print("INFO: ===== Cross-Validation (balanced per class: "
          "1 VAL + 1 TEST per class per fold, TRAIN balanced by smallest class) =====")

    for fi, split in enumerate(folds):
        acc, f1, cm = train_one_fold(cfg, fi, X, y, split, class_names)
        accs.append(acc)
        f1s.append(f1)
        summed_cm += cm

    acc_mean, acc_std = np.mean(accs), np.std(accs)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s)

    print("INFO:")
    print(f"INFO: Folds: {cfg.cv_folds}  |  Mode: per-class balanced")
    print(f"INFO: Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"INFO: Macro-F1: {f1_mean:.2f}% ± {f1_std:.2f}%")
    print("INFO: Summed Confusion Matrix over folds (rows=true, cols=pred):")
    print("INFO:")

    names = class_names if class_names else [str(i) for i in range(num_classes)]
    header = "      " + "".join([f"{n:>8s}" for n in names])
    print("INFO:", header)
    for r in range(num_classes):
        row_lab = names[r]
        row = f"{row_lab:>6s}" + "".join([f"{summed_cm[r, c]:>8d}" for c in range(num_classes)])
        print("INFO:", row)
