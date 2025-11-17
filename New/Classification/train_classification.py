import os
import json
import math
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
CLASSES = ["HAARYE", "OTHER", "TUT"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_path: str
    splits_path: Optional[str] = None  # unused now, kept for compatibility
    cv_folds: int = 5

    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 25

    lr: float = 1e-3
    weight_decay: float = 1e-4

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    keep_channels: Optional[List[int]] = None  # indices to keep, e.g. [0,1,2]


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label_str(s: str) -> str:
    """
    Normalize any raw label string into one of:
        - "HAARYE"
        - "TUT"
        - "OTHER"

    This mirrors the logic in preprocessing._norm_label so that even if the
    .npy file contains slightly different label strings, we still collapse
    everything to the same 3 classes.
    """
    up = (str(s) or "").strip().upper()
    if up.startswith("HAARY"):       # "HAARYE", "HAARY", etc.
        return "HAARYE"
    if up in {"TUT", "TUT.", "TUT,"}:
        return "TUT"
    if up in {"האריה"}:
        return "HAARYE"
    if up in {"תות"}:
        return "TUT"
    # Everything else is OTHER by definition
    return "OTHER"


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------
def load_dataset(
    data_path: str, keep_channels: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the preprocessed .npy created by preprocessing/preprocessing.py
    and returns:
        X: [N, T, C]
        y: [N] array of *normalized* string labels in {HAARYE, OTHER, TUT}
    """
    arr = np.load(data_path, allow_pickle=True)
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for item in arr:
        sig = item["signals"]  # [T, C]
        if keep_channels is not None:
            sig = sig[:, keep_channels]
        # Normalize label robustly (fix for "OTHER" count mismatch)
        raw_label = item["label"]
        norm_label = normalize_label_str(raw_label)

        X_list.append(sig.astype(np.float32))
        y_list.append(norm_label)

    X = np.stack(X_list, axis=0)  # [N, T, C]
    y = np.array(y_list)

    # Debug: show full label distribution after normalization
    unique, counts = np.unique(y, return_counts=True)
    logger.info("Label distribution after normalization:")
    for lab, cnt in zip(unique, counts):
        logger.info("  %s: %d", lab, int(cnt))

    return X, y


class LfpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: List[int]):
        self.X = X[indices]  # [N, T, C]
        self.y = y[indices]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]  # [T, C]
        # Model will get [C, T] for Conv1d
        x_tensor = torch.from_numpy(x.T)  # [C, T]
        y_label = self.y[idx]
        y_idx = LABEL_TO_IDX[y_label]
        y_tensor = torch.tensor(y_idx, dtype=torch.long)
        return x_tensor, y_tensor


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        h = self.net(x)  # [B, 128, 1]
        h = h.squeeze(-1)  # [B, 128]
        logits = self.fc(h)  # [B, num_classes]
        return logits


# ---------------------------------------------------------------------
# Class weights (to fight OTHER dominance + slightly help HAARYE & TUT)
# ---------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency-like weights and then gently boost HAARYE and TUT.
    """
    counts: Dict[str, int] = {c: int(np.sum(y == c)) for c in CLASSES}
    total = sum(counts.values())

    raw: Dict[str, float] = {}
    for c in CLASSES:
        # basic inverse frequency
        raw[c] = total / (len(CLASSES) * max(counts[c], 1))

    # Gently boost minority / "speech" classes
    raw["HAARYE"] *= 1.5
    raw["TUT"] *= 1.3

    w_arr = np.array([raw[c] for c in CLASSES], dtype=np.float32)
    # normalize so mean(weight) = 1
    w_arr = w_arr / w_arr.mean()
    return torch.tensor(w_arr, dtype=torch.float32)


# ---------------------------------------------------------------------
# Balanced per-class CV splits with non-repeated VAL/TEST
# ---------------------------------------------------------------------
def build_balanced_per_class_splits_non_repeating(
    y: np.ndarray, cv_folds: int, seed: int
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Build cross-validation splits such that:

    - We have three classes: HAARYE, OTHER, TUT.
    - In EACH fold:
        * 1 VAL + 1 TEST sample from each class.
        * TRAIN is balanced: the same number of samples from each class.
        * train_per_class = min_class_size - 2
          (exactly: smallest class minus 1 val minus 1 test).
    - A sample that was used as VAL/TEST in some fold:
        * Will NOT be used as VAL/TEST again in later folds (per class).
        * BUT can be used as TRAIN in later folds (as requested).

    Returns:
        List of (train_idx, val_idx, test_idx) for each fold.
    """
    rng = np.random.default_rng(seed)

    indices_by_class: Dict[str, List[int]] = {}
    for c in CLASSES:
        idxs = np.where(y == c)[0].tolist()
        rng.shuffle(idxs)
        indices_by_class[c] = idxs

    class_counts = {c: len(idxs) for c, idxs in indices_by_class.items()}
    logger.info("Class counts (original dataset):")
    for c in CLASSES:
        logger.info("  %s: %d", c, class_counts[c])

    # Identify smallest class explicitly
    min_class = min(class_counts, key=class_counts.get)
    min_size = class_counts[min_class]
    train_per_class = max(1, min_size - 2)  # minus 1 VAL, minus 1 TEST

    logger.info(
        "Smallest class: %s (%d samples) -> train samples per class per fold: %d",
        min_class,
        min_size,
        train_per_class,
    )

    # Track which indices have already been used as VAL/TEST per class
    used_vt: Dict[str, set] = {c: set() for c in CLASSES}

    folds: List[Tuple[List[int], List[int], List[int]]] = []

    for fold in range(cv_folds):
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        logger.debug("=== Building fold %d/%d ===", fold + 1, cv_folds)

        for c in CLASSES:
            all_idxs = indices_by_class[c]

            # ------- choose VAL + TEST for this fold (not reused as VAL/TEST) -------
            candidates_vt = [i for i in all_idxs if i not in used_vt[c]]
            if len(candidates_vt) < 2:
                logger.warning(
                    "Not enough unseen VAL/TEST candidates for class %s in fold %d, reusing some.",
                    c,
                    fold + 1,
                )
                chosen_vt = rng.choice(all_idxs, size=2, replace=False)
            else:
                chosen_vt = rng.choice(candidates_vt, size=2, replace=False)

            v_idx = int(chosen_vt[0])
            t_idx = int(chosen_vt[1])
            used_vt[c].add(v_idx)
            used_vt[c].add(t_idx)

            val_idx.append(v_idx)
            test_idx.append(t_idx)

            # ------- choose TRAIN indices for this fold -------
            # Allowed train candidates: all indices except the VAL/TEST of THIS fold.
            train_candidates = [i for i in all_idxs if i not in (v_idx, t_idx)]

            if len(train_candidates) <= train_per_class:
                chosen_train = train_candidates
            else:
                chosen_train = rng.choice(
                    train_candidates, size=train_per_class, replace=False
                ).tolist()

            train_idx.extend(chosen_train)

        logger.debug(
            "[Fold %d] TRAIN size=%d  VAL size=%d  TEST size=%d",
            fold + 1,
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        folds.append((train_idx, val_idx, test_idx))

    return folds


# ---------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------
def evaluate_model(
    model: nn.Module, loader: DataLoader, device: str
) -> Tuple[float, float, np.ndarray]:
    """
    Returns:
        accuracy, macro_f1, confusion_matrix(3x3)
    """
    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(yb.cpu().numpy().tolist())

    if len(all_true) == 0:
        return 0.0, 0.0, np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro")
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(CLASSES))))
    return acc * 100.0, macro_f1 * 100.0, cm


def train_single_fold(
    fold_idx: int,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    class_weights: torch.Tensor,
) -> Tuple[float, float, np.ndarray]:
    """
    Train on a single fold and return:
        test_acc_percent, test_macro_f1_percent, test_confusion_matrix
    """
    device = cfg.device
    set_seed(cfg.seed + fold_idx)

    train_ds = LfpDataset(X, y, train_idx)
    val_ds = LfpDataset(X, y, val_idx)
    test_ds = LfpDataset(X, y, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    _, T, C = X.shape
    model = SimpleConvNet(in_channels=C, num_classes=len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best_val_f1 = -math.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / max(len(train_ds), 1)

        # Validation
        val_acc, val_macro_f1, _ = evaluate_model(model, val_loader, device)
        scheduler.step(val_macro_f1)

        logger.debug(
            "Fold %d | Epoch %d: Train Loss = %.4f, Val Macro-F1 = %.2f%%",
            fold_idx + 1,
            epoch,
            train_loss,
            val_macro_f1,
        )

        # Early stopping based on Val Macro-F1
        if val_macro_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_macro_f1
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                logger.debug(
                    "Fold %d | Early stopping at epoch %d (no improvement in %d epochs).",
                    fold_idx + 1,
                    epoch,
                    cfg.patience,
                )
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_acc, test_macro_f1, test_cm = evaluate_model(model, test_loader, device)
    logger.info(
        "Fold %d — Test Accuracy: %.2f%%  |  Macro-F1: %.2f%%",
        fold_idx + 1,
        test_acc,
        test_macro_f1,
    )

    # Pretty-print confusion matrix for this fold
    logger.info("Confusion Matrix (fold) [rows=true, cols=pred]:")
    header = "        " + "   ".join(f"{c:>6}" for c in CLASSES)
    logger.info(header)
    for i, c in enumerate(CLASSES):
        row_vals = "   ".join(f"{int(v):>6}" for v in test_cm[i])
        logger.info(f"{c:>6}   {row_vals}")

    return test_acc, test_macro_f1, test_cm


# ---------------------------------------------------------------------
# Main train entry
# ---------------------------------------------------------------------
def train(cfg: TrainConfig) -> None:
    logger.info("Starting training. data_path=%s", cfg.data_path)

    if not os.path.isfile(cfg.data_path):
        raise FileNotFoundError(f"data_path not found: {cfg.data_path}")

    # Load whole dataset (preprocessing already did the windowing)
    X, y = load_dataset(cfg.data_path, cfg.keep_channels)

    # Build balanced per-class CV folds according to your rules
    logger.info(
        "\n===== Cross-Validation (balanced per class: 1 VAL + 1 TEST per class per fold, TRAIN balanced by smallest class) ====="
    )
    folds = build_balanced_per_class_splits_non_repeating(
        y, cfg.cv_folds, cfg.seed
    )

    # Compute class weights once on the full dataset
    class_weights = compute_class_weights(y)
    logger.info("Using class weights (CrossEntropy):")
    for c, w in zip(CLASSES, class_weights.numpy()):
        logger.info("  %s: %.3f", c, w)

    all_test_acc: List[float] = []
    all_test_f1: List[float] = []
    sum_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
        test_acc, test_macro_f1, test_cm = train_single_fold(
            fold_idx, train_idx, val_idx, test_idx, X, y, cfg, class_weights
        )
        all_test_acc.append(test_acc)
        all_test_f1.append(test_macro_f1)
        sum_cm += test_cm

    # Summary over folds
    acc_mean = float(np.mean(all_test_acc))
    acc_std = float(np.std(all_test_acc))
    f1_mean = float(np.mean(all_test_f1))
    f1_std = float(np.std(all_test_f1))

    logger.info(
        "\nFolds: %d  |  Mode: per-class balanced", cfg.cv_folds
    )
    logger.info(
        "Accuracy: %.2f%% ± %.2f%%", acc_mean, acc_std
    )
    logger.info(
        "Macro-F1: %.2f%% ± %.2f%%", f1_mean, f1_std
    )

    logger.info("Summed Confusion Matrix over folds (rows=true, cols=pred):\n")
    header = "        " + "   ".join(f"{c:>6}" for c in CLASSES)
    logger.info(header)
    for i, c in enumerate(CLASSES):
        row_vals = "   ".join(f"{int(v):>6}" for v in sum_cm[i])
        logger.info(f"{c:>6}   {row_vals}")


if __name__ == "__main__":
    # Minimal CLI for standalone testing (usually you call this from main.py)
    import argparse

    p = argparse.ArgumentParser(description="LFP Classification Trainer")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keep_channels",
        type=str,
        default=None,
        help="Comma-separated channel indices to keep, e.g. '0,1,2'.",
    )

    args = p.parse_args()

    if args.keep_channels is not None:
        keep_ch = [int(tok) for tok in args.keep_channels.split(",") if tok.strip() != ""]
    else:
        keep_ch = None

    cfg = TrainConfig(
        data_path=args.data_path,
        cv_folds=args.cv_folds,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        keep_channels=keep_ch,
    )

    train(cfg)
