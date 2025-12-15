from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_recall_fscore_support,
)
from scipy.stats import ttest_rel  # for paired t-test

from defines import PROCESSED_DATA_DIR

DEBUG = True

# =============================
# Device
# =============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Model / training hyperparams
# =============================

# LSTM + Conv sizes (moderate model with slight tuning)
CONV_CHANNELS = 80          # num feature maps in conv
LSTM_HIDDEN_SIZE = 128      # per direction (BiLSTM -> 2*128)
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2          # between LSTM layers (reduced)
FC_HIDDEN_SIZE = 128
FC_DROPOUT = 0.3            # reduced FC dropout

CONV_KERNEL_SIZE1 = 15
CONV_KERNEL_SIZE2 = 5
POOL_KERNEL_SIZE = 2

WEIGHT_DECAY = 1e-4

# Label cleanup thresholds (fraction of speech samples in window)
# Stricter cleanup: keep only "clean" speech / non-speech windows.
SPEECH_FRACTION_HIGH = 0.80   # >= -> label as speech
SPEECH_FRACTION_LOW = 0.20    # <= -> label as non-speech
# windows with fraction in (LOW, HIGH) are dropped as ambiguous


# =============================
# Model definition
# =============================

class ChannelDropout(nn.Module):
    """
    Dropout over channels (spatial dropout for Conv1d).

    During training, randomly drops whole channels with probability p.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if not self.training or self.p <= 0.0:
            return x
        # mask shape: (B, C, 1)
        mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.p).float()
        return x * mask


class LSTMDetector(nn.Module):
    def __init__(
        self,
        input_size: int,
        conv_channels: int = CONV_CHANNELS,
        lstm_hidden: int = LSTM_HIDDEN_SIZE,
        lstm_layers: int = LSTM_NUM_LAYERS,
        fc_hidden: int = FC_HIDDEN_SIZE,
        lstm_dropout: float = LSTM_DROPOUT,
        fc_dropout: float = FC_DROPOUT,
        channel_dropout_p: float = 0.1,
    ):
        super().__init__()

        # Conv front-end
        self.conv1 = nn.Conv1d(input_size, conv_channels, CONV_KERNEL_SIZE1)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, CONV_KERNEL_SIZE2)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU()
        self.channel_dropout = ChannelDropout(p=channel_dropout_p)
        self.pool = nn.MaxPool1d(POOL_KERNEL_SIZE)

        # BiLSTM
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Simple additive attention over time dimension
        self.attn = nn.Linear(2 * lstm_hidden, 1)

        # Fully-connected head
        self.fc_hidden = nn.Linear(2 * lstm_hidden, fc_hidden)
        self.dropout_fc = nn.Dropout(fc_dropout)
        self.fc_out = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.channel_dropout(x)
        x = self.pool(x)

        # back to (B, T', C')
        x = x.permute(0, 2, 1)

        # LSTM
        out, _ = self.lstm(x)  # (B, T', 2 * hidden)

        # Attention over time dimension
        # scores: (B, T')
        attn_scores = self.attn(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # context: (B, 2 * hidden)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)

        # Head
        x = self.relu(self.fc_hidden(context))
        x = self.dropout_fc(x)
        logits = self.fc_out(x)
        return logits


# =============================
# Focal Loss implementation
# =============================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for logits.

    Args:
        alpha: weight for positive class (scalar between 0 and 1)
        gamma: focusing parameter
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1), targets: (B, 1) in {0,1}
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        # pt = p for y=1, 1-p for y=0
        pt = probs * targets + (1 - probs) * (1 - targets)

        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================
# Utility functions
# =============================

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_channels_and_time(csv_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSVs exported by preprocessing:
        time, signal

    Returns:
        signals: (N_samples, N_channels)
        time: (N_samples,)
    """
    csv_files = sorted(csv_dir.glob("*.csv"))
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found in {csv_dir}")

    channel_dfs = []
    first_time = None

    for chf in csv_files:
        df_ch = pd.read_csv(chf)
        if "time" not in df_ch.columns or "signal" not in df_ch.columns:
            continue
        if first_time is None:
            first_time = df_ch["time"].values.astype(float)
        channel_dfs.append(df_ch["signal"].values.astype(float))
        if DEBUG:
            print(f"[DEBUG] Loaded channel {chf.name} with {len(df_ch)} samples")

    if len(channel_dfs) == 0:
        raise RuntimeError(f"No valid channel CSVs in {csv_dir} (missing time/signal)")

    # Stack -> (N_samples, N_channels)
    signals = np.column_stack(channel_dfs)
    time = first_time

    if DEBUG:
        tmin, tmax = float(time.min()), float(time.max())
        print(f"[DEBUG] Using raw CSV time range: [{tmin:.3f}, {tmax:.3f}]")

    return signals, time


def _bandpass_and_smooth(
    signals: np.ndarray,
    sr: float,
    lowcut: float,
    highcut: float,
) -> np.ndarray:
    """
    Optional feature engineering:
      * band-pass filter (Butterworth, 4th order)
      * short moving-average smoothing (~10ms)
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.999

    if DEBUG:
        print(f"[DEBUG] Applying band-pass filter {lowcut:.1f}–{highcut:.1f}Hz...")

    b, a = butter(4, [low, high], btype="bandpass")

    # Apply per channel
    filtered = np.zeros_like(signals)
    for ch in range(signals.shape[1]):
        filtered[:, ch] = filtfilt(b, a, signals[:, ch])

    # Moving-average smoothing window ~10ms
    win_len = max(1, int(sr * 0.01))  # 10ms
    if win_len > 1:
        if DEBUG:
            print("[DEBUG] Applying moving average smoothing (~10ms)...")
        kernel = np.ones(win_len) / win_len
        smoothed = np.zeros_like(filtered)
        for ch in range(filtered.shape[1]):
            smoothed[:, ch] = np.convolve(filtered[:, ch], kernel, mode="same")
        return smoothed
    else:
        return filtered


def _extract_windows(
    signals: np.ndarray,
    time: np.ndarray,
    df_labels: pd.DataFrame,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows over the labeled time range and label them using
    labels_aligned.csv with a coverage-based cleanup.

    For each window:
        * compute fraction of samples inside speech intervals
        * if fraction >= SPEECH_FRACTION_HIGH -> label = 1 (speech)
        * if fraction <= SPEECH_FRACTION_LOW  -> label = 0 (non-speech)
        * otherwise, drop the window (ambiguous)
    """
    num_channels = signals.shape[1]
    if DEBUG:
        print(f"[DEBUG] Combined signals shape: {signals.shape} (samples x channels)")
        print(f"[DEBUG] Window_size: {window_size}, stride: {stride}")
        print(
            f"[DEBUG] Label cleanup: keep speech_fraction >= {SPEECH_FRACTION_HIGH:.2f} "
            f"or <= {SPEECH_FRACTION_LOW:.2f}"
        )

    # Normalize per-channel
    if DEBUG:
        print("[DEBUG] Normalizing each channel (zero mean, unit std)...")
    means = signals.mean(axis=0)
    stds = signals.std(axis=0)
    stds[stds == 0] = 1.0
    signals = (signals - means) / stds

    # Range sanity check
    if DEBUG:
        tmin, tmax = float(time.min()), float(time.max())
        lmin = float(df_labels["start_adj"].min())
        lmax = float(df_labels["end_adj"].max())
        print(f"[DEBUG] time range: [{tmin:.3f}, {tmax:.3f}] ; labels range: [{lmin:.3f}, {lmax:.3f}]")

    n_samples = signals.shape[0]
    speech_mask = np.zeros(n_samples, dtype=np.uint8)  # 1 for speech, 0 for non-speech
    labeled_mask = np.zeros(n_samples, dtype=bool)

    first_idx = n_samples
    last_idx = 0

    # Build sample-level masks from segment labels
    for _, row in df_labels.iterrows():
        start_time = float(row["start_adj"])
        end_time = float(row["end_adj"])
        label_str = str(row["label"])
        is_speech = label_str.lower() not in ["none", "nospeech"]

        start_idx = np.searchsorted(time, start_time, side="left")
        end_idx = np.searchsorted(time, end_time, side="right") - 1

        if start_idx >= n_samples or end_idx < 0 or end_idx <= start_idx:
            continue

        start_idx = max(0, start_idx)
        end_idx = min(n_samples - 1, end_idx)

        labeled_mask[start_idx:end_idx + 1] = True
        if is_speech:
            speech_mask[start_idx:end_idx + 1] = 1

        first_idx = min(first_idx, start_idx)
        last_idx = max(last_idx, end_idx)

    if first_idx >= last_idx:
        raise RuntimeError("No valid overlap between labels and signal time axis.")

    if DEBUG:
        total_labeled = int(labeled_mask[first_idx:last_idx + 1].sum())
        total_span = last_idx - first_idx + 1
        print(
            f"[DEBUG] Labeled region indices: [{first_idx}, {last_idx}] "
            f"({total_span} samples, {total_labeled} labeled)"
        )

    X_windows: List[np.ndarray] = []
    y_labels: List[int] = []

    # Slide only over the labeled region
    for start in range(first_idx, last_idx - window_size + 1, stride):
        end = start + window_size
        if end > n_samples:
            break

        # Enforce that the entire window is within labeled region
        if not labeled_mask[start:end].all():
            continue

        frac_speech = float(speech_mask[start:end].mean())

        if frac_speech >= SPEECH_FRACTION_HIGH:
            label = 1
        elif frac_speech <= SPEECH_FRACTION_LOW:
            label = 0
        else:
            # ambiguous window: mixture of speech and non-speech -> drop
            continue

        X_windows.append(signals[start:end, :])
        y_labels.append(label)

    X_windows = np.asarray(X_windows, dtype=float)
    y_labels = np.asarray(y_labels, dtype=int)

    if DEBUG:
        print(f"[DEBUG] Total windows after label cleanup: {len(X_windows)}")
        if len(X_windows) == 0:
            print(
                "[DEBUG] No windows found after label cleanup. "
                "Check window_size/stride or label coverage."
            )
        else:
            n_pos = int((y_labels == 1).sum())
            n_neg = int((y_labels == 0).sum())
            print(
                f"[DEBUG] Class distribution before balancing: "
                f"class0={n_neg}, class1={n_pos}"
            )

    if len(X_windows) == 0:
        raise RuntimeError(
            "No windows were extracted for training after label cleanup – "
            "check window_size/stride or labels coverage"
        )

    return X_windows, y_labels


def _balance_classes(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Global balancing: keep the same number of positive and negative windows.
    Returns the balanced subset.
    """
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    n_pos = len(idx_pos)
    n_neg = len(idx_neg)
    n_min = min(n_pos, n_neg)

    if n_min == 0:
        raise RuntimeError("One of the classes has zero samples – cannot balance.")

    # randomly choose n_min from each class
    rng = np.random.default_rng(42)
    idx_pos_sel = rng.choice(idx_pos, size=n_min, replace=False)
    idx_neg_sel = rng.choice(idx_neg, size=n_min, replace=False)

    idx_all = np.concatenate([idx_pos_sel, idx_neg_sel])
    # Sort to keep a stable order (not strictly necessary)
    idx_all = np.sort(idx_all)

    X_bal = X[idx_all]
    y_bal = y[idx_all]

    if DEBUG:
        print(
            f"[DEBUG] After global balancing: {len(y_bal)} windows "
            f"(class0={np.sum(y_bal == 0)}, class1={np.sum(y_bal == 1)})"
        )

    return X_bal, y_bal


def _make_cv_splits(
    y: np.ndarray,
    n_folds: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create cyclic, stratified cross-validation splits.

    For each class separately:
        * shuffle indices
        * split into n_folds contiguous blocks
        * for fold k:
              TEST = block k
              VAL  = block (k+1 mod n_folds)
              TRAIN = all remaining blocks
    This ensures that for each class every sample appears exactly once in TEST
    and exactly once in VAL across the n_folds.
    """
    y = np.asarray(y)
    classes = np.unique(y)

    rng = np.random.default_rng(42)

    per_class_folds = {}  # class -> list of (train_idx_c, val_idx_c, test_idx_c)

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_c = len(idx_c)

        # Split idx_c into n_folds blocks with sizes differing by at most 1
        base_size = n_c // n_folds
        rem = n_c % n_folds
        blocks = []
        start = 0
        for f in range(n_folds):
            size = base_size + (1 if f < rem else 0)
            end = start + size
            blocks.append(idx_c[start:end])
            start = end

        # Some blocks may be empty if n_c < n_folds, but in our setting n_c is large.
        class_folds = []
        for k in range(n_folds):
            test_block = blocks[k]
            val_block = blocks[(k + 1) % n_folds]
            train_blocks = [
                blocks[j] for j in range(n_folds)
                if j not in (k, (k + 1) % n_folds)
            ]
            if len(train_blocks) > 0:
                train_idx_c = np.concatenate(train_blocks)
            else:
                train_idx_c = np.array([], dtype=int)

            class_folds.append((train_idx_c, val_block, test_block))

        per_class_folds[c] = class_folds

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for k in range(n_folds):
        train_list = []
        val_list = []
        test_list = []
        for c in classes:
            train_c, val_c, test_c = per_class_folds[c][k]
            train_list.append(train_c)
            val_list.append(val_c)
            test_list.append(test_c)

        train_idx = np.concatenate(train_list)
        val_idx = np.concatenate(val_list)
        test_idx = np.concatenate(test_list)

        # Shuffle within each split for randomness
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        splits.append((train_idx, val_idx, test_idx))

    return splits


def _find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "accuracy",
    beta: float = 1.0,
) -> Tuple[float, float]:
    """
    Scan thresholds in [0.05, 0.95] and choose the one that maximizes:
        * Accuracy  (if metric == 'accuracy')
        * F_beta    (if metric == 'f1', using given beta)
    """
    y_true = y_true.astype(int)
    thresholds = np.linspace(0.05, 0.95, 37)

    best_thr = 0.5
    best_score = -1.0

    for thr in thresholds:
        y_pred = (probs >= thr).astype(int)
        if metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr, best_score


def _smooth_probs(probs: np.ndarray, k: int) -> np.ndarray:
    """
    Simple smoothing over probabilities: replace each prob with the mean of
    a window of size k centered on it. For edges we shrink the window.
    """
    if k <= 1:
        return probs

    n = len(probs)
    out = np.zeros_like(probs)
    half = k // 2
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        out[i] = probs[s:e].mean()
    return out


def _init_weights(m: nn.Module):
    """
    Weight initialization for Conv1d, Linear, and LSTM layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)


# =============================
# Train entry
# =============================

class TrainerConfigError(Exception):
    pass


def train(
    config,
    patient_id: str,
    use_fe: bool = False,
    fe_lowcut: float = 1.0,
    fe_highcut: float = 150.0,
    use_focal: bool = False,
    threshold_metric: str = "f1",   # default F1
    smooth_k: int = 1,
):
    """
    Main training function for detection model, with 10-fold cross-validation.

    Args:
        config          : Dict loaded from patients_config.json
        patient_id      : Which patient entry from config["patients"] to use
        use_fe          : Enable feature engineering (band-pass + smoothing)
        fe_lowcut       : Low cut frequency (Hz) for band-pass (FE)
        fe_highcut      : High cut frequency (Hz) for band-pass (FE)
        use_focal       : If True, use FocalLoss; otherwise BCEWithLogitsLoss
        threshold_metric: "accuracy" or "f1" – used to choose decision threshold on val
        smooth_k        : Smoothing window size on test probabilities; 1 => no smoothing
    """
    if patient_id not in config["patients"]:
        raise TrainerConfigError(f"Patient {patient_id} not found in config")

    patient_info = config["patients"][patient_id]

    # per-patient directory under PROCESSED_DATA_DIR
    patient_base_dir = Path(PROCESSED_DATA_DIR) / patient_id
    csv_dir = patient_base_dir / "csvs"
    labels_csv = patient_base_dir / "labels_aligned.csv"

    if DEBUG:
        print(f"[DEBUG] Using per-patient processed dir: {patient_base_dir}")
        print(f"[DEBUG] Using device: {DEVICE}")

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"labels_aligned.csv not found at {labels_csv}. "
            f"Run preprocessing first (or use --force_preproc for {patient_id})."
        )
    if not csv_dir.exists():
        raise FileNotFoundError(
            f"CSV directory not found at {csv_dir}. "
            f"Run preprocessing first (or use --force_preproc for {patient_id})."
        )

    sample_rate = int(patient_info.get("sample_rate", 2000))

    # window_size / stride from config (updated by preprocessing config)
    window_size = int(patient_info.get("window_size", 1200))
    stride = int(patient_info.get("stride", int(window_size * 0.5)))

    batch_size = int(patient_info.get("batch_size", 64))
    learning_rate = float(patient_info.get("learning_rate", 1e-3))
    num_epochs = int(patient_info.get("epochs", 80))

    if DEBUG:
        print(f"[DEBUG] Training on patient: {patient_id}")
        print(
            f"[DEBUG] Parameters - window_size: {window_size}, stride: {stride}, "
            f"batch: {batch_size}, lr: {learning_rate}, epochs: {num_epochs}"
        )
        print(f"[DEBUG] Feature engineering: {'ON' if use_fe else 'OFF'}")
        if use_fe:
            print(
                f"[DEBUG] FE band-pass frequencies: {fe_lowcut:.1f}–{fe_highcut:.1f} Hz"
            )
        print(f"[DEBUG] Loss: {'FocalLoss' if use_focal else 'BCEWithLogitsLoss'}")
        print(f"[DEBUG] Threshold metric: {threshold_metric}")
        print(f"[DEBUG] Test smoothing k: {smooth_k}")

    # 1) Load signals and time
    signals, time = _load_channels_and_time(csv_dir)

    # 2) Optional feature engineering
    if use_fe:
        signals = _bandpass_and_smooth(
            signals,
            sr=sample_rate,
            lowcut=fe_lowcut,
            highcut=fe_highcut,
        )
        if DEBUG:
            print(
                f"[DEBUG] Combined signals shape after FE: {signals.shape} "
                f"(samples x channels)"
            )
    else:
        if DEBUG:
            print("[DEBUG] Feature engineering disabled, using raw signals")

    # 3) Load labels
    df_labels = pd.read_csv(labels_csv)
    if not {"start_adj", "end_adj", "label"}.issubset(df_labels.columns):
        raise RuntimeError(
            "labels_aligned.csv does not contain required columns: "
            "start_adj, end_adj, label"
        )

    # 4) Build windows with coverage-based label cleanup
    X_windows, y_labels = _extract_windows(
        signals=signals,
        time=time,
        df_labels=df_labels,
        window_size=window_size,
        stride=stride,
    )

    # 5) Balance classes globally
    X_bal, y_bal = _balance_classes(X_windows, y_labels)

    # 6) 10-fold cross-validation setup (stratified cyclic per class)
    n_folds = 10
    splits = _make_cv_splits(y_bal, n_folds=n_folds)

    fold_acc_raw = []
    fold_prec_raw = []
    fold_rec_raw = []
    fold_f1_raw = []

    fold_acc_smooth = []
    fold_prec_smooth = []
    fold_rec_smooth = []
    fold_f1_smooth = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n[CV] Fold {fold_idx}/{n_folds}")

        X_train = X_bal[train_idx]
        y_train = y_bal[train_idx]
        X_val = X_bal[val_idx]
        y_val = y_bal[val_idx]
        X_test = X_bal[test_idx]
        y_test = y_bal[test_idx]

        if DEBUG:
            print(f"[CV] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            print(
                f"[CV] Train class0={np.sum(y_train == 0)}, "
                f"class1={np.sum(y_train == 1)}"
            )
            print(
                f"[CV] Val   class0={np.sum(y_val == 0)}, "
                f"class1={np.sum(y_val == 1)}"
            )
            print(
                f"[CV] Test  class0={np.sum(y_test == 0)}, "
                f"class1={np.sum(y_test == 1)}"
            )

        # Tensor conversion (initially on CPU)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Class imbalance info
        pos_count = int(y_train.sum())
        neg_count = len(y_train) - pos_count

        # For BCE: pos_weight = neg/pos
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # For focal: alpha for positive examples ~ fraction of negative class
        alpha_pos = (
            neg_count / (pos_count + neg_count)
            if (pos_count + neg_count) > 0
            else 0.5
        )

        if DEBUG:
            if use_focal:
                print(
                    f"[CV] pos_count: {pos_count}, neg_count: {neg_count}, "
                    f"alpha_pos (Focal): {alpha_pos:.2f}"
                )
            else:
                print(
                    f"[CV] pos_count: {pos_count}, neg_count: {neg_count}, "
                    f"pos_weight (BCE): {pos_weight:.2f}"
                )

        # Model
        num_channels = X_train.shape[2]
        model = LSTMDetector(input_size=num_channels).to(DEVICE)
        model.apply(_init_weights)

        if DEBUG:
            print(f"[CV] Model Architecture: {model}")

        # Loss
        if use_focal:
            criterion = FocalLoss(alpha=alpha_pos, gamma=2.0, reduction="mean")
        else:
            pos_weight_tensor = torch.tensor(
                [pos_weight], dtype=torch.float32, device=DEVICE
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Optimizer + scheduler (Reduce LR on plateau of val loss)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        )

        # Move validation tensors to DEVICE once
        X_val_tensor_device = X_val_tensor.to(DEVICE)
        y_val_tensor_device = y_val_tensor.to(DEVICE)

        # Train with early stopping on chosen validation metric
        best_val_score = 0.0
        best_state = None
        epochs_no_improve = 0
        patience_es = 10

        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= max(1, len(train_loader))

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor_device)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().reshape(-1)
                y_val_np = y_val_tensor_device.cpu().numpy().reshape(-1).astype(int)

            # Early stopping metric: same family as threshold_metric
            if threshold_metric == "accuracy":
                val_preds_es = (val_probs >= 0.5).astype(int)
                val_metric_value = accuracy_score(y_val_np, val_preds_es)
            else:
                # F1 with beta=1.0 at threshold 0.5
                val_preds_es = (val_probs >= 0.5).astype(int)
                val_metric_value = fbeta_score(
                    y_val_np, val_preds_es, beta=1.0, zero_division=0
                )

            scheduler.step(epoch_loss)

            print(
                f"[CV] Fold {fold_idx}, Epoch [{epoch}/{num_epochs}], "
                f"Loss: {epoch_loss:.4f}, "
                f"Val {threshold_metric.upper()}: {val_metric_value * 100:.2f}%"
            )

            # Early stopping on chosen metric
            if val_metric_value > best_val_score:
                best_val_score = val_metric_value
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience_es:
                    print(
                        f"[CV] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience_es} epochs on val {threshold_metric})."
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        print(
            f"[CV] Fold {fold_idx} best VAL {threshold_metric.upper()}: "
            f"{best_val_score * 100:.2f}%"
        )

        # ---- Threshold selection on VAL ----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor_device)
            val_probs_raw = torch.sigmoid(val_logits).cpu().numpy().reshape(-1)
            y_val_np = y_val_tensor_device.cpu().numpy().reshape(-1).astype(int)

        # If smoothing is used in TEST, use it also for threshold selection on VAL
        if smooth_k > 1:
            val_probs_for_thr = _smooth_probs(val_probs_raw, k=smooth_k)
        else:
            val_probs_for_thr = val_probs_raw

        # Use standard F1 (beta=1.0) when threshold_metric == "f1"
        beta_for_f = 1.0
        best_thr, best_score = _find_best_threshold(
            y_true=y_val_np,
            probs=val_probs_for_thr,
            metric=threshold_metric,
            beta=beta_for_f,
        )

        if threshold_metric == "accuracy":
            print(
                f"[DEBUG] Best threshold on val (metric=Accuracy): {best_thr:.2f} "
                f"(Acc={best_score:.3f})"
            )
        else:
            print(
                f"[DEBUG] Best threshold on val (metric=F{beta_for_f:.2f}): {best_thr:.2f} "
                f"(F_beta={best_score:.3f})"
            )

        # Also print VAL metrics at best_thr (helps compare VAL vs TEST)
        val_preds_best = (val_probs_for_thr >= best_thr).astype(int)
        val_acc_best = accuracy_score(y_val_np, val_preds_best)
        val_prec_best, val_rec_best, val_f1_best, _ = precision_recall_fscore_support(
            y_val_np, val_preds_best, average="binary", zero_division=0
        )
        print(
            f"[CV] Fold {fold_idx} VAL @best_thr={best_thr:.2f}: "
            f"Acc={val_acc_best * 100:.2f}%, "
            f"Prec={val_prec_best * 100:.2f}%, "
            f"Rec={val_rec_best * 100:.2f}%, "
            f"F1={val_f1_best * 100:.2f}%"
        )

        # ---- Final evaluation on TEST ----
        X_test_tensor_device = X_test_tensor.to(DEVICE)
        y_test_tensor_device = y_test_tensor.to(DEVICE)

        with torch.no_grad():
            test_logits = model(X_test_tensor_device)
            test_probs_raw = torch.sigmoid(test_logits).cpu().numpy().reshape(-1)
            y_test_np = y_test_tensor_device.cpu().numpy().reshape(-1).astype(int)

        # RAW metrics (no smoothing)
        test_preds_raw = (test_probs_raw >= best_thr).astype(int)

        acc_raw = accuracy_score(y_test_np, test_preds_raw)
        prec_raw, rec_raw, f1_raw, _ = precision_recall_fscore_support(
            y_test_np, test_preds_raw, average="binary", zero_division=0
        )

        tp_raw = int(((test_preds_raw == 1) & (y_test_np == 1)).sum())
        tn_raw = int(((test_preds_raw == 0) & (y_test_np == 0)).sum())
        fp_raw = int(((test_preds_raw == 1) & (y_test_np == 0)).sum())
        fn_raw = int(((test_preds_raw == 0) & (y_test_np == 1)).sum())

        fold_acc_raw.append(acc_raw)
        fold_prec_raw.append(prec_raw)
        fold_rec_raw.append(rec_raw)
        fold_f1_raw.append(f1_raw)

        print(
            f"[FOLD {fold_idx}/{n_folds}] Test Accuracy (raw): {acc_raw * 100:.2f}% "
            f"({tp_raw + tn_raw}/{len(y_test_np)} windows correct)"
        )
        print(
            f"[FOLD {fold_idx}/{n_folds}] TP: {tp_raw}, FP: {fp_raw}, "
            f"TN: {tn_raw}, FN: {fn_raw}"
        )
        print(
            f"[FOLD {fold_idx}/{n_folds}] Precision (raw): {prec_raw * 100:.2f}%, "
            f"Recall (raw): {rec_raw * 100:.2f}%, F1 (raw): {f1_raw * 100:.2f}%"
        )
        print(
            f"[FOLD {fold_idx}/{n_folds}] Best threshold used on test: {best_thr:.2f}"
        )

        # Smoothing metrics (optional, simple moving average only)
        if smooth_k > 1:
            probs_smooth = _smooth_probs(test_probs_raw, k=smooth_k)
            preds_smooth = (probs_smooth >= best_thr).astype(int)

            acc_sm = accuracy_score(y_test_np, preds_smooth)
            prec_sm, rec_sm, f1_sm, _ = precision_recall_fscore_support(
                y_test_np, preds_smooth, average="binary", zero_division=0
            )

            tp_sm = int(((preds_smooth == 1) & (y_test_np == 1)).sum())
            tn_sm = int(((preds_smooth == 0) & (y_test_np == 0)).sum())
            fp_sm = int(((preds_smooth == 1) & (y_test_np == 0)).sum())
            fn_sm = int(((preds_smooth == 0) & (y_test_np == 1)).sum())

            fold_acc_smooth.append(acc_sm)
            fold_prec_smooth.append(prec_sm)
            fold_rec_smooth.append(rec_sm)
            fold_f1_smooth.append(f1_sm)

            print(
                f"[FOLD {fold_idx}/{n_folds}] Test Accuracy (smoothed, k={smooth_k}): "
                f"{acc_sm * 100:.2f}% ({tp_sm + tn_sm}/{len(y_test_np)} windows correct)"
            )
            print(
                f"[FOLD {fold_idx}/{n_folds}] TP_s: {tp_sm}, FP_s: {fp_sm}, "
                f"TN_s: {tn_sm}, FN_s: {fn_sm}"
            )
            print(
                f"[FOLD {fold_idx}/{n_folds}] Precision (smoothed): {prec_sm * 100:.2f}%, "
                f"Recall (smoothed): {rec_sm * 100:.2f}%, F1 (smoothed): {f1_sm * 100:.2f}%"
            )
            # Per-fold deltas for easier interpretation
            print(
                f"[FOLD {fold_idx}/{n_folds}] ΔAcc (smoothed - raw): "
                f"{(acc_sm - acc_raw) * 100:.2f} pp, "
                f"ΔF1: {(f1_sm - f1_raw) * 100:.2f} pp"
            )

    # ===== Summary over folds =====
    acc_raw_mean = np.mean(fold_acc_raw)
    acc_raw_std = np.std(fold_acc_raw)
    prec_raw_mean = np.mean(fold_prec_raw)
    prec_raw_std = np.std(fold_prec_raw)
    rec_raw_mean = np.mean(fold_rec_raw)
    rec_raw_std = np.std(fold_rec_raw)
    f1_raw_mean = np.mean(fold_f1_raw)
    f1_raw_std = np.std(fold_f1_raw)

    # Also compute max metrics for the table
    acc_raw_max = np.max(fold_acc_raw)
    prec_raw_max = np.max(fold_prec_raw)
    rec_raw_max = np.max(fold_rec_raw)
    f1_raw_max = np.max(fold_f1_raw)

    print("\n===== 10-fold cross-validation summary (RAW) =====")
    print(f"Mean Test Accuracy (raw):  {acc_raw_mean * 100:.2f}%")
    print(f"Std  Test Accuracy (raw):  {acc_raw_std * 100:.2f}%")
    print(f"Mean Precision (raw):      {prec_raw_mean * 100:.2f}%")
    print(f"Std  Precision (raw):      {prec_raw_std * 100:.2f}%")
    print(f"Mean Recall (raw):         {rec_raw_mean * 100:.2f}%")
    print(f"Std  Recall (raw):         {rec_raw_std * 100:.2f}%")
    print(f"Mean F1 (raw):             {f1_raw_mean * 100:.2f}%")
    print(f"Std  F1 (raw):             {f1_raw_std * 100:.2f}%")

    acc_sm_mean = None
    f1_sm_mean = None
    acc_p = None   # p-value for accuracy t-test (if computed)
    f1_p = None    # p-value for F1 t-test (if computed)

    if smooth_k > 1 and len(fold_acc_smooth) > 0:
        acc_sm_mean = np.mean(fold_acc_smooth)
        acc_sm_std = np.std(fold_acc_smooth)
        prec_sm_mean = np.mean(fold_prec_smooth)
        prec_sm_std = np.std(fold_prec_smooth)
        rec_sm_mean = np.mean(fold_rec_smooth)
        rec_sm_std = np.std(fold_rec_smooth)
        f1_sm_mean = np.mean(fold_f1_smooth)
        f1_sm_std = np.std(fold_f1_smooth)

        print("\n===== 10-fold cross-validation summary (SMOOTHED) =====")
        print(
            f"Mean Test Accuracy (smoothed, k={smooth_k}): "
            f"{acc_sm_mean * 100:.2f}%"
        )
        print(
            f"Std  Test Accuracy (smoothed, k={smooth_k}): "
            f"{acc_sm_std * 100:.2f}%"
        )
        print(
            f"Mean Precision (smoothed): "
            f"{prec_sm_mean * 100:.2f}%"
        )
        print(
            f"Std  Precision (smoothed): "
            f"{prec_sm_std * 100:.2f}%"
        )
        print(
            f"Mean Recall (smoothed): "
            f"{rec_sm_mean * 100:.2f}%"
        )
        print(
            f"Std  Recall (smoothed): "
            f"{rec_sm_std * 100:.2f}%"
        )
        print(
            f"Mean F1 (smoothed): "
            f"{f1_sm_mean * 100:.2f}%"
        )
        print(
            f"Std  F1 (smoothed): "
            f"{f1_sm_std * 100:.2f}%"
        )

        # Explicit deltas (smoothed - raw), in percentage points
        delta_acc = (acc_sm_mean - acc_raw_mean) * 100.0
        delta_f1 = (f1_sm_mean - f1_raw_mean) * 100.0
        print("\n===== SMOOTHED vs RAW (per-fold means) =====")
        print(f"ΔAccuracy (smoothed - raw): {delta_acc:+.2f} pp")
        print(f"ΔF1       (smoothed - raw): {delta_f1:+.2f} pp")

        # ===== Paired t-test: SMOOTHED vs RAW over folds =====
        if len(fold_acc_smooth) == len(fold_acc_raw):
            acc_t, acc_p = ttest_rel(fold_acc_smooth, fold_acc_raw)
            f1_t, f1_p = ttest_rel(fold_f1_smooth, fold_f1_raw)
            print("\n===== Paired t-test (SMOOTHED vs RAW, per-fold) =====")
            print(f"Accuracy: t = {acc_t:.4f}, p = {acc_p:.4e}")
            print(f"F1:       t = {f1_t:.4f}, p = {f1_p:.4e}")
        else:
            print(
                "\n[WARN] Cannot run paired t-test: "
                "different number of folds for RAW and SMOOTHED metrics."
            )
    elif smooth_k <= 1:
        print(
            "\n[T-TEST] Smoothing disabled (smooth_k=1); "
            "paired t-test requires both RAW and SMOOTHED metrics."
        )

    # ===== Run configuration summary (for easier comparison between runs) =====
    print("\n===== RUN CONFIGURATION =====")
    print(f"Patient id: {patient_id}")
    print(f"Feature engineering: {'ON' if use_fe else 'OFF'}")
    if use_fe:
        print(
            f"  FE band-pass frequencies: {fe_lowcut:.1f}–{fe_highcut:.1f} Hz"
        )
    print(f"Loss function: {'FocalLoss' if use_focal else 'BCEWithLogitsLoss'}")
    print(f"Threshold metric: {threshold_metric}")
    print(f"Test smoothing window k: {smooth_k}")
    print(f"Window size: {window_size}, stride: {stride}")
    print(f"Batch size: {batch_size}, learning rate: {learning_rate}, epochs: {num_epochs}")

    # Compact summary line – easy to copy into a spreadsheet for comparisons
    # acc_* and f1_* are printed as decimals (0.xx)
    if acc_sm_mean is not None:
        acc_sm_str = f"{acc_sm_mean:.4f}"
    else:
        acc_sm_str = "NA"
    if f1_sm_mean is not None:
        f1_sm_str = f"{f1_sm_mean:.4f}"
    else:
        f1_sm_str = "NA"

    print(
        "\n[RUN SUMMARY] "
        f"patient={patient_id}, "
        f"FE={use_fe}, "
        f"focal={use_focal}, "
        f"smooth_k={smooth_k}, "
        f"thr_metric={threshold_metric}, "
        f"acc_raw={acc_raw_mean:.4f}, "
        f"f1_raw={f1_raw_mean:.4f}, "
        f"acc_sm={acc_sm_str}, "
        f"f1_sm={f1_sm_str}"
    )

    # ===== Extra compact summary for the table (matching your header) =====
    # Avg / Max metrics are decimals (0.xx); T-test values are p-values if available.
    table_ttest_acc = acc_p if acc_p is not None else "NA"
    table_ttest_f1 = f1_p if f1_p is not None else "NA"

    print(
        "\n[TABLE SUMMARY] "
        f"Patient={patient_id}, "
        f"AvgAccuracy={acc_raw_mean:.4f}, MaxAccuracy={acc_raw_max:.4f}, "
        f"AvgF1={f1_raw_mean:.4f}, MaxF1={f1_raw_max:.4f}, "
        f"AvgPrecision={prec_raw_mean:.4f}, MaxPrecision={prec_raw_max:.4f}, "
        f"AvgRecall={rec_raw_mean:.4f}, MaxRecall={rec_raw_max:.4f}, "
        f"AvgTTest={table_ttest_acc}, MaxTTest={table_ttest_f1}"
    )
