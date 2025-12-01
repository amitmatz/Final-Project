import logging
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")


# -----------------------------
# Helpers
# -----------------------------

def log_class_counts(counts: Dict[int, int], inv_label_map: Dict[int, str]):
    logger.info("Per-class counts:")
    for c, cnt in counts.items():
        logger.info(f"  {inv_label_map[c]}: {cnt}")


def build_stratified_cv_folds(
    labels: np.ndarray,
    cv_folds: int,
    val_per_class: int = 1,
    test_per_class: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """
    Build stratified folds such that for each class:
      - each fold uses exactly val_per_class to VAL and test_per_class to TEST
      - these VAL/TEST samples are disjoint across folds.
    Remaining samples go to TRAIN in each fold.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed=42)

    folds = []
    for _ in range(cv_folds):
        folds.append({"train_idx": [], "val_idx": [], "test_idx": []})

    for c in classes:
        c_indices = np.where(labels == c)[0]
        rng.shuffle(c_indices)

        needed_per_class = cv_folds * (val_per_class + test_per_class)
        if needed_per_class > len(c_indices):
            raise ValueError(
                f"Not enough samples in class {c} for {cv_folds} folds with "
                f"{val_per_class} val and {test_per_class} test per fold."
            )

        vt_indices = c_indices[:needed_per_class]
        vt_indices = vt_indices.reshape(cv_folds, val_per_class + test_per_class)

        # assign val/test
        for f in range(cv_folds):
            fold_slice = vt_indices[f]
            val_idx_f = fold_slice[:val_per_class]
            test_idx_f = fold_slice[val_per_class : val_per_class + test_per_class]
            folds[f]["val_idx"].extend(val_idx_f.tolist())
            folds[f]["test_idx"].extend(test_idx_f.tolist())

        # assign train for this class in each fold
        for f in range(cv_folds):
            val_test_f = set(folds[f]["val_idx"]) | set(folds[f]["test_idx"])
            train_for_class_f = [i for i in c_indices if i not in val_test_f]
            folds[f]["train_idx"].extend(train_for_class_f)

    for f in range(cv_folds):
        for key in ["train_idx", "val_idx", "test_idx"]:
            folds[f][key] = np.array(folds[f][key], dtype=int)

    return folds


# -----------------------------
# Temporal pooling helper
# -----------------------------

def temporal_pooling_1d_to_seq(
    flat_features: np.ndarray,
    original_seq_len: int = 3000,
    target_seq_len: int = 24,
) -> np.ndarray:
    """
    Convert 1D flattened features (time x channels) into a shorter sequence
    via temporal average pooling.

    flat_features: shape [flat_dim], where flat_dim = original_seq_len * feat_dim_per_step
    Returns: [target_seq_len, feat_dim_per_step]
    """
    flat_features = np.asarray(flat_features, dtype=np.float32).reshape(-1)
    flat_dim = flat_features.shape[0]

    if flat_dim % original_seq_len != 0:
        raise ValueError(
            f"Flat dim {flat_dim} is not divisible by original_seq_len={original_seq_len}"
        )

    feat_dim_per_step = flat_dim // original_seq_len
    features_2d = flat_features.reshape(original_seq_len, feat_dim_per_step)

    factor = original_seq_len // target_seq_len
    if factor * target_seq_len != original_seq_len:
        # If not divisible, truncate the tail
        usable_len = factor * target_seq_len
        features_2d = features_2d[:usable_len]
    else:
        usable_len = original_seq_len

    # reshape to [target_seq_len, factor, feat_dim_per_step]
    features_3d = features_2d.reshape(target_seq_len, factor, feat_dim_per_step)
    # average over time within each chunk
    pooled = features_3d.mean(axis=1)  # [target_seq_len, feat_dim_per_step]
    return pooled.astype(np.float32)


# -----------------------------
# LSTM model
# -----------------------------

class LSTMClassifier(nn.Module):
    """
    Simple LSTM classifier for sequence input:
    Input: [batch, seq_len, input_dim]
    Output: logits [batch, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        logits = self.fc(last_hidden)
        return logits


# -----------------------------
# Main training entry point
# -----------------------------

def train_cross_validation(
    data_path: str,
    cv_folds: int = 5,
    batch_size: int = 32,
    augment_minor: bool = False,
    feature_dim: int = 0,
    **kwargs,
):
    """
    Train LSTM classifier with stratified cross-validation on temporally pooled
    sequence features.

    data_path: path to .npy produced by preprocessing.py
    cv_folds: number of CV folds
    batch_size: batch size for DataLoader (passed from main.py)
    augment_minor: if True, oversample ONLY the globally smallest class in TRAIN.
    feature_dim: kept only for compatibility with main.py (ignored here)
    kwargs: absorbs any extra unused arguments from main.py
    """
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Total samples file: {data_path}")

    # -----------------------------
    # Load data
    # -----------------------------
    data_list = np.load(data_path, allow_pickle=True).tolist()
    if not isinstance(data_list, list):
        raise ValueError("Expected data_list to be a list of dicts from npy file.")

    if len(data_list) == 0:
        raise ValueError("Empty data_list loaded from npy file.")

    logger.info(f"Total samples: {len(data_list)}")

    example = data_list[0]
    if "label" not in example:
        raise ValueError("Expected key 'label' in samples.")

    # Handle feature key(s)
    feature_key = None
    if "features" in example:
        feature_key = "features"
    elif "flattened_features" in example:
        feature_key = "flattened_features"
    else:
        # fallback: allow 'signals' and flatten
        if "signals" in example:
            feature_key = "signals"
        else:
            raise ValueError(
                "Expected one of keys 'features', 'flattened_features', or 'signals' in samples."
            )

    # labels mapping
    labels_str = [sample["label"] for sample in data_list]
    unique_labels = sorted(list(set(labels_str)))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    inv_label_map = {v: k for k, v in label_map.items()}
    labels = np.array([label_map[lab] for lab in labels_str], dtype=int)

    logger.info("Label distribution after normalization per-class counts:")
    global_counts = {label_map[lab]: labels_str.count(lab) for lab in unique_labels}
    log_class_counts(global_counts, inv_label_map)

    # -----------------------------
    # Flatten + temporal pooling for ALL samples once
    # -----------------------------
    # Example to infer shapes
    ex_features = np.asarray(example[feature_key], dtype=np.float32)

    if ex_features.ndim == 1:
        flat_example = ex_features.reshape(-1)
    elif ex_features.ndim == 2:
        flat_example = ex_features.reshape(-1)
    else:
        flat_example = ex_features.reshape(-1)

    flat_dim_orig = flat_example.shape[0]
    logger.info(f"Original flattened feature dim per sample: {flat_dim_orig}")

    ORIGINAL_SEQ_LEN = 3000
    TARGET_SEQ_LEN = 24

    if flat_dim_orig % ORIGINAL_SEQ_LEN != 0:
        raise ValueError(
            f"Flat dim {flat_dim_orig} is not divisible by ORIGINAL_SEQ_LEN={ORIGINAL_SEQ_LEN}"
        )

    feat_dim_per_step = flat_dim_orig // ORIGINAL_SEQ_LEN
    factor = ORIGINAL_SEQ_LEN // TARGET_SEQ_LEN

    logger.info(
        f"Applying temporal pooling: original_seq_len={ORIGINAL_SEQ_LEN} "
        f"-> target_seq_len={TARGET_SEQ_LEN} (factor={factor})"
    )

    all_sequences = []
    for sample in data_list:
        feats = np.asarray(sample[feature_key], dtype=np.float32).reshape(-1)
        seq = temporal_pooling_1d_to_seq(
            feats, original_seq_len=ORIGINAL_SEQ_LEN, target_seq_len=TARGET_SEQ_LEN
        )
        all_sequences.append(seq)

    X_all = np.stack(all_sequences, axis=0)  # [N, seq_len, feat_dim_per_step]
    seq_len = X_all.shape[1]
    feat_dim_actual = X_all.shape[2]
    flat_dim_new = seq_len * feat_dim_actual

    logger.info(
        f"After temporal pooling: seq_len={seq_len}, feat_dim_per_step={feat_dim_actual}, "
        f"flat_dim={flat_dim_new}"
    )
    logger.info(
        f"Effective input size to LSTM per sample: seq_len={seq_len}, feat_dim={feat_dim_actual}"
    )

    # -----------------------------
    # Build CV folds
    # -----------------------------
    logger.info(
        "===== Stratified CV (folds; 1 per class TEST, 1 per class VAL; "
        "VAL/TEST samples are disjoint across folds) ====="
    )
    folds = build_stratified_cv_folds(labels, cv_folds=cv_folds, val_per_class=1, test_per_class=1)

    all_test_true_all_folds: List[int] = []
    all_test_pred_all_folds: List[int] = []

    # -----------------------------
    # Loop over folds
    # -----------------------------
    max_epochs = 200
    early_stopping_patience = 20

    for fold_idx, fold in enumerate(folds, start=1):
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        test_idx = fold["test_idx"]

        logger.info(f"=== Building fold {fold_idx}/{cv_folds} ===")
        logger.info(
            f"[Fold {fold_idx}] Sizes: TRAIN={len(train_idx)}, VAL={len(val_idx)}, TEST={len(test_idx)}"
        )

        def get_counts(idxs: np.ndarray) -> Dict[int, int]:
            counts = {c: 0 for c in range(len(unique_labels))}
            for i in idxs:
                counts[labels[int(i)]] += 1
            return counts

        # Log per-class counts for TRAIN / VAL / TEST
        logger.info(f"[Fold {fold_idx} TRAIN] per-class counts:")
        log_class_counts(get_counts(train_idx), inv_label_map)
        logger.info(f"[Fold {fold_idx} VAL] per-class counts:")
        log_class_counts(get_counts(val_idx), inv_label_map)
        logger.info(f"[Fold {fold_idx} TEST] per-class counts:")
        log_class_counts(get_counts(test_idx), inv_label_map)

        # Slice sequences + labels for this fold
        X_train = X_all[train_idx]
        y_train = labels[train_idx]
        X_val = X_all[val_idx]
        y_val = labels[val_idx]
        X_test = X_all[test_idx]
        y_test = labels[test_idx]

        # -----------------------------
        # Normalize features using TRAIN only
        # -----------------------------
        train_flat = X_train.reshape(-1, feat_dim_actual)
        feat_mean = train_flat.mean(axis=0, keepdims=True)
        feat_std = train_flat.std(axis=0, keepdims=True) + 1e-6

        def normalize(X: np.ndarray) -> np.ndarray:
            n, t, d = X.shape
            X_flat = X.reshape(-1, d)
            X_norm = (X_flat - feat_mean) / feat_std
            return X_norm.reshape(n, t, d)

        X_train = normalize(X_train)
        X_val = normalize(X_val)
        X_test = normalize(X_test)

        # -----------------------------
        # Augmentation ONLY on smallest class (per-fold)
        # -----------------------------
        if augment_minor:
            class_counts = {c: int((y_train == c).sum()) for c in range(len(unique_labels))}
            logger.info(f"Augmentation enabled (fold {fold_idx}). Train counts BEFORE:")
            log_class_counts(class_counts, inv_label_map)

            # Find the smallest class
            minority_class = min(class_counts, key=class_counts.get)
            n_min = class_counts[minority_class]

            # Target size: second smallest class (not full balance)
            sorted_counts_vals = sorted(class_counts.values())
            if len(sorted_counts_vals) >= 2:
                target_size = sorted_counts_vals[1]
            else:
                target_size = n_min

            n_to_add = max(0, target_size - n_min)

            if n_to_add > 0:
                logger.info(
                    f"  -> Oversampling minority class {minority_class} "
                    f"({inv_label_map[minority_class]}): {n_min} -> {target_size} "
                    f"(add {n_to_add} samples)"
                )
                rng = np.random.default_rng(seed=fold_idx * 100 + 1)
                src_indices = np.where(y_train == minority_class)[0]
                aug_indices = rng.choice(src_indices, size=n_to_add, replace=True)

                X_aug = X_train[aug_indices] + rng.normal(
                    loc=0.0,
                    scale=0.05,
                    size=X_train[aug_indices].shape,
                )
                y_aug = np.full(n_to_add, minority_class, dtype=y_train.dtype)

                X_train = np.concatenate([X_train, X_aug], axis=0)
                y_train = np.concatenate([y_train, y_aug], axis=0)
            else:
                logger.info(
                    "  -> No augmentation needed (minority already >= second smallest)."
                )

            new_counts = {c: int((y_train == c).sum()) for c in range(len(unique_labels))}
            logger.info("TRAIN per-class counts AFTER augmentation:")
            log_class_counts(new_counts, inv_label_map)
        else:
            logger.info("Augmentation disabled for this run (augment_minor=False).")

        # -----------------------------
        # DataLoaders
        # -----------------------------
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info("Using device: %s", device)
        logger.info(
            f"Using feature dimension per step: {feat_dim_actual}, "
            f"sequence length: {seq_len}, batch_size={batch_size}"
        )

        # -----------------------------
        # Loss, model, optimizer, scheduler
        # -----------------------------
        # Class weights based on post-augmentation distribution
        class_counts_train = np.bincount(y_train, minlength=len(unique_labels))
        freq = class_counts_train / class_counts_train.sum()
        inv_freq = 1.0 / (freq + 1e-6)
        inv_freq /= inv_freq.mean()
        class_weights = torch.tensor(inv_freq, dtype=torch.float32, device=device)

        logger.info("Using class weights (CrossEntropy) based on post-augmentation distribution:")
        for c in range(len(unique_labels)):
            logger.info(f"  {inv_label_map[c]}: {class_weights[c].item():.3f}")

        model = LSTMClassifier(
            input_dim=feat_dim_actual,
            hidden_dim=64,
            num_layers=1,
            num_classes=len(unique_labels),
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )

        # -----------------------------
        # Training loop with early stopping
        # -----------------------------
        best_val_f1 = -np.inf
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_losses = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # Validation
            model.eval()
            val_losses = []
            all_val_preds = []
            all_val_true = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_losses.append(loss.item())

                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_val_preds.extend(preds.tolist())
                    all_val_true.extend(yb.cpu().numpy().tolist())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            if all_val_true:
                val_f1 = f1_score(all_val_true, all_val_preds, average="macro")
                val_acc = (np.array(all_val_true) == np.array(all_val_preds)).mean()
            else:
                val_f1 = 0.0
                val_acc = 0.0

            logger.info(
                f"Fold {fold_idx} | Epoch {epoch}: "
                f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                f"Val Macro-F1 = {val_f1*100:.2f}%, Val Acc = {val_acc*100:.2f}%"
            )

            # Scheduler (maximize val_f1)
            scheduler.step(val_f1)

            # Early stopping
            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = val_f1
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    logger.info(
                        f"Fold {fold_idx} | Early stopping at epoch {epoch} "
                        f"(no improvement in {early_stopping_patience} epochs)."
                    )
                    break

        if best_state is not None:
            logger.info(
                f"Fold {fold_idx} | Best Val Macro-F1 = {best_val_f1*100:.2f}% "
            )
            model.load_state_dict(best_state)

        # -----------------------------
        # Test evaluation
        # -----------------------------
        model.eval()
        all_test_true = []
        all_test_preds = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_test_preds.extend(preds.tolist())
                all_test_true.extend(yb.cpu().numpy().tolist())

        if all_test_true:
            test_acc = (np.array(all_test_true) == np.array(all_test_preds)).mean()
            test_f1 = f1_score(all_test_true, all_test_preds, average="macro")
        else:
            test_acc = 0.0
            test_f1 = 0.0

        cm = confusion_matrix(
            all_test_true, all_test_preds, labels=list(range(len(unique_labels)))
        )

        logger.info(
            f"Fold {fold_idx} — Test Accuracy: {test_acc*100:.2f}%  "
            f"|  Macro-F1: {test_f1*100:.2f}%"
        )
        logger.info("Fold %d Confusion Matrix [rows=true, cols=pred]:", fold_idx)
        header = "     " + "".join(f"{lab:>8}" for lab in unique_labels)
        logger.info(header)
        for i, lab in enumerate(unique_labels):
            row = f"{lab:>8}" + "".join(f"{cm[i, j]:>8}" for j in range(len(unique_labels)))
            logger.info(row)

        # Per-class metrics for this fold
        logger.info(f"Fold {fold_idx} per-class metrics:")
        tp = np.diag(cm)
        support = cm.sum(axis=1)
        predicted = cm.sum(axis=0)
        for c in range(len(unique_labels)):
            prec = tp[c] / predicted[c] if predicted[c] > 0 else 0.0
            rec = tp[c] / support[c] if support[c] > 0 else 0.0
            if prec + rec > 0:
                f1_c = 2 * prec * rec / (prec + rec)
            else:
                f1_c = 0.0
            logger.info(
                f"  {inv_label_map[c]}: Prec={prec*100:.2f}%, "
                f"Rec={rec*100:.2f}%, F1={f1_c*100:.2f}%, Support={support[c]}"
            )

        all_test_true_all_folds.extend(all_test_true)
        all_test_pred_all_folds.extend(all_test_preds)

    # -----------------------------
    # Overall summary over folds (all test sets merged)
    # -----------------------------
    logger.info("")
    logger.info("===== Overall CV Results =====")
    if all_test_true_all_folds:
        overall_acc = (
            np.array(all_test_true_all_folds) == np.array(all_test_pred_all_folds)
        ).mean()
        overall_f1 = f1_score(
            all_test_true_all_folds, all_test_pred_all_folds, average="macro"
        )
    else:
        overall_acc = 0.0
        overall_f1 = 0.0

    logger.info(f"Overall Test Accuracy (all folds merged): {overall_acc*100:.2f}%")
    logger.info(f"Overall Macro-F1 (all folds merged): {overall_f1*100:.2f}%")

    overall_cm = confusion_matrix(
        all_test_true_all_folds,
        all_test_pred_all_folds,
        labels=list(range(len(unique_labels))),
    )
    logger.info(
        "Summed Confusion Matrix (over folds) [rows=true, cols=pred]:"
    )
    header = "     " + "".join(f"{lab:>8}" for lab in unique_labels)
    logger.info(header)
    for i, lab in enumerate(unique_labels):
        row = f"{lab:>8}" + "".join(
            f"{overall_cm[i, j]:>8}" for j in range(len(unique_labels))
        )
        logger.info(row)

    # Overall per-class metrics
    logger.info("Overall per-class metrics:")
    tp = np.diag(overall_cm)
    support = overall_cm.sum(axis=1)
    predicted = overall_cm.sum(axis=0)
    for c in range(len(unique_labels)):
        prec = tp[c] / predicted[c] if predicted[c] > 0 else 0.0
        rec = tp[c] / support[c] if support[c] > 0 else 0.0
        if prec + rec > 0:
            f1_c = 2 * prec * rec / (prec + rec)
        else:
            f1_c = 0.0
        logger.info(
            f"  {inv_label_map[c]}: Prec={prec*100:.2f}%, "
            f"Rec={rec*100:.2f}%, F1={f1_c*100:.2f}%, Support={support[c]}"
        )
