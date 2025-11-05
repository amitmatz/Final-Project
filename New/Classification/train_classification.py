# Classification/train_classification.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# ---------------------- Config ----------------------

@dataclass
class TrainConfig:
    # identifiers
    patient_id: str = "Patient_01"
    device: str = "cpu"

    # data & batching
    batch_size: int = 64
    epochs: int = 60
    balanced_one_batch: bool = False  # if True -> use balanced batches per epoch (many batches)

    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # or "plateau"
    clip_grad_norm: float = 1.0

    # model
    lstm_layers: int = 2
    conv_norm: str = "group"  # "batch" | "group"
    k1: int = 17
    k2: int = 5
    use_dilation: bool = False
    use_mha: bool = False  # placeholder to keep API stable
    dropout: float = 0.3

    # loss
    use_focal: bool = False
    gamma_focal: float = 1.5
    label_smoothing: float = 0.0
    cb_beta: float = 0.9999  # class-balanced weighting beta

    # augment / regularization (applied on train only)
    mixup_alpha: float = 0.0
    time_mask_prob: float = 0.0
    time_mask_max_ms: int = 120
    time_mask_max_frac: Optional[float] = None
    channel_drop_prob: float = 0.0

    # CV
    tau_grid: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    cv_folds: int = 5
    val_frac_of_rest: float = 0.15  # ~15% of remaining data used for validation


# ---------------------- Dataset ----------------------

class ClassificationDataset(Dataset):
    """
    Holds X [N,T,C] and y [N] AFTER optional subsetting (idxs).
    __getitem__ expects LOCAL indices 0..len(self)-1.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, idxs: Optional[np.ndarray] = None):
        if idxs is None:
            self.X = X
            self.y = y
        else:
            self.X = X[idxs]
            self.y = y[idxs]
        assert self.X.shape[0] == self.y.shape[0], "X/y size mismatch"
        self.N = self.X.shape[0]
        self.T = self.X.shape[1]
        self.C = self.X.shape[2]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        xi = torch.from_numpy(self.X[i]).float()    # [T,C]
        yi = int(self.y[i])
        return xi, yi


# ---------------------- Sampler ----------------------

class BalancedBatchSampler(Sampler[List[int]]):
    """
    Yields MANY balanced batches per epoch (not one).
    Each batch has the same number of samples per class; if not divisible, it fills as evenly as possible.
    Works on LOCAL training set (length len(train_idx)).
    """
    def __init__(self, y_local: np.ndarray, batch_size: int, num_classes: int, rng: random.Random):
        assert batch_size >= num_classes, "batch_size must be >= num_classes for balanced batches."
        self.y_local = np.asarray(y_local)
        self.bs = batch_size
        self.k = num_classes
        self.rng = rng

        self.class_to_local_idxs: Dict[int, List[int]] = {}
        for cls in range(num_classes):
            self.class_to_local_idxs[cls] = np.where(self.y_local == cls)[0].tolist()

        # epoch length = approximately ceil(N / bs)
        self.N = len(self.y_local)
        self.batches_per_epoch = max(1, math.ceil(self.N / self.bs))

    def __iter__(self):
        # shuffle class pools every epoch
        for cls in range(self.k):
            self.rng.shuffle(self.class_to_local_idxs[cls])

        ptrs = {cls: 0 for cls in range(self.k)}

        # number per class in each batch (as equal as possible)
        base = self.bs // self.k
        rem = self.bs - base * self.k
        per_class = [base + (1 if i < rem else 0) for i in range(self.k)]

        for _ in range(self.batches_per_epoch):
            batch: List[int] = []
            for cls, need in enumerate(per_class):
                pool = self.class_to_local_idxs[cls]
                # collect 'need' samples with replacement if required
                chosen: List[int] = []
                for _ in range(need):
                    if ptrs[cls] >= len(pool):
                        # refill by sampling with replacement if class exhausted
                        chosen.append(self.rng.choice(pool))
                    else:
                        chosen.append(pool[ptrs[cls]])
                        ptrs[cls] += 1
                batch.extend(chosen)
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


# ---------------------- Model ----------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, norm="group", dilation=False, dropout=0.0):
        super().__init__()
        d = 2 if dilation else 1
        pad = (k - 1) // 2 * d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=d)
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_ch)
        else:
            groups = max(1, min(8, out_ch))
            self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.ReLU(inplace=True)
        self.do = nn.Dropout(p=dropout)

    def forward(self, x):  # x: [B,C,T]
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.do(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, in_ch=48, k1=17, k2=5, norm="group", dilation=False,
                 lstm_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.c1 = ConvBlock(in_ch, 64, k1, norm=norm, dilation=dilation, dropout=dropout)
        self.c2 = ConvBlock(64, 96, k2, norm=norm, dilation=dilation, dropout=dropout)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=96, hidden_size=128, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # x: [B,T,C]
        x = x.transpose(1, 2)        # [B,C,T]
        x = self.c1(x)
        x = self.c2(x)
        x = self.pool(x)             # [B,96,T/2]
        x = x.transpose(1, 2)        # [B,T',96]
        out, _ = self.lstm(x)        # [B,T',256]
        e = self.attn(out)           # [B,T',1]
        w = torch.softmax(e, dim=1)  # [B,T',1]
        z = (out * w).sum(dim=1)     # [B,256]
        logits = self.head(z)        # [B,3]
        return logits


# ---------------------- Losses ----------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt) ** self.gamma * logp.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def class_balanced_weights(y: np.ndarray, beta: float, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta + 1e-8)
    w = 1.0 / (eff + 1e-8)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ---------------------- Augmentations ----------------------

def apply_time_mask(x: torch.Tensor, prob: float, max_w: int, rng: random.Random) -> torch.Tensor:
    if prob <= 0 or max_w <= 0:
        return x
    B, T, C = x.shape
    for b in range(B):
        if rng.random() < prob:
            w = rng.randint(1, max_w)
            s = rng.randint(0, max(0, T - w))
            x[b, s:s + w, :] = 0.0
    return x


def apply_channel_drop(x: torch.Tensor, drop_prob: float, rng: random.Random) -> torch.Tensor:
    if drop_prob <= 0:
        return x
    B, T, C = x.shape
    for b in range(B):
        for c in range(C):
            if rng.random() < drop_prob:
                x[b, :, c] = 0.0
    return x


# ---------------------- Metrics ----------------------

CLS2IDX = {"HAARYE": 0, "TUT": 1, "OTHER": 2}
IDX2CLS = {v: k for k, v in CLS2IDX.items()}

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes=3) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes=3) -> Tuple[float, float, float]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    precs, recs, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes=3) -> float:
    _, _, f1 = precision_recall_f1(y_true, y_pred, num_classes)
    return f1


# ---------------------- Training (one split) ----------------------

def train_one_split(cfg: TrainConfig,
                    X: np.ndarray, y: np.ndarray,
                    train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, any]:

    device = torch.device(cfg.device)

    ds_train = ClassificationDataset(X, y, idxs=train_idx)
    ds_val   = ClassificationDataset(X, y, idxs=val_idx)
    ds_test  = ClassificationDataset(X, y, idxs=test_idx)

    y_train_local = ds_train.y.astype(int)
    num_classes = 3

    # Build model
    model = LSTMClassifier(
        in_ch=ds_train.C,
        k1=cfg.k1, k2=cfg.k2,
        norm=cfg.conv_norm,
        dilation=cfg.use_dilation,
        lstm_layers=cfg.lstm_layers,
        num_classes=num_classes,
        dropout=cfg.dropout
    ).to(device)

    # Loss
    if cfg.use_focal:
        criterion = FocalLoss(gamma=cfg.gamma_focal)
        class_weights = None
        print(f"[INFO] Using FocalLoss (gamma={cfg.gamma_focal})")
    else:
        class_weights = class_balanced_weights(y_train_local, cfg.cb_beta, num_classes).to(device)
        criterion = lambda logits, ytrue: F.cross_entropy(
            logits, ytrue, weight=class_weights, label_smoothing=cfg.label_smoothing
        )
        print(f"[INFO] Using CrossEntropy (label_smoothing={cfg.label_smoothing}, beta_cb={cfg.cb_beta})")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    else:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # DataLoaders
    rng = random.Random(1337)

    if cfg.balanced_one_batch:
        sampler = BalancedBatchSampler(y_train_local, cfg.batch_size, num_classes, rng)
        loader_train = DataLoader(ds_train, batch_sampler=sampler)
        print(f"[DEBUG] [BalancedBatch] epoch has {len(sampler)} balanced batches of size {cfg.batch_size}")
    else:
        loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)

    loader_val  = DataLoader(ds_val,  batch_size=cfg.batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

    def evaluate(loader):
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.cpu().numpy())
        L = np.concatenate(all_logits, axis=0)
        Y = np.concatenate(all_y, axis=0)
        preds = L.argmax(axis=1)  # ARGMAX ONLY (no tau) to avoid HAARYE bias
        acc = float((preds == Y).mean())
        mf1 = macro_f1(Y, preds, num_classes=3)
        return acc, mf1, preds, Y

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    patience = 20

    print(f"[DEBUG] [Sanity] Data: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}; "
          f"batch_size={cfg.batch_size}, balanced_mode={cfg.balanced_one_batch}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for xb, yb in loader_train:
            xb = xb.to(device); yb = yb.to(device)

            # Augmentations
            fs = 2000  # Hz
            max_w = int((cfg.time_mask_max_ms / 1000.0) * fs) if cfg.time_mask_max_ms else 0
            if max_w > 0 or cfg.time_mask_prob > 0:
                xb = apply_time_mask(xb, cfg.time_mask_prob, max_w, rng)
            if cfg.channel_drop_prob > 0:
                xb = apply_channel_drop(xb, cfg.channel_drop_prob, rng)

            # Forward
            logits = model(xb)

            # Mixup (loss as convex combination of CE targets)
            if cfg.mixup_alpha and cfg.mixup_alpha > 0.0:
                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                idx = torch.randperm(xb.size(0), device=xb.device)
                loss = lam * F.cross_entropy(logits, yb, weight=class_weights, label_smoothing=cfg.label_smoothing) + \
                       (1 - lam) * F.cross_entropy(logits, yb[idx], weight=class_weights, label_smoothing=cfg.label_smoothing)
            else:
                loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            steps += 1

        # Validation
        val_acc, val_f1, _, _ = evaluate(loader_val)
        print(f"[DEBUG] Epoch {epoch}: Val Macro-F1 = {val_f1*100:.2f}%")

        if cfg.scheduler == "cosine":
            sch.step()
        else:
            sch.step(val_f1)

        # Early stopping
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[DEBUG] Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_f1, test_pred, test_true = evaluate(loader_test)
    cm = confusion_matrix(test_true, test_pred)

    return {
        "test_acc": test_acc,
        "test_f1": test_f1,
        "cm": cm,
        "y_true": test_true,
        "y_pred": test_pred,
    }


# ---------------------- CV wrapper ----------------------

def stratified_kfold_indices(y: np.ndarray, k: int, seed: int = 42,
                             val_frac_of_rest: float = 0.15) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns list of (train_idx, val_idx, test_idx) for each fold.
    Strategy:
      - Split into k folds stratified for TEST.
      - From the remaining, take a stratified VAL ≈ val_frac_of_rest of remaining
        (with a target minimum of ~8 per class when available).
    """
    rng = np.random.RandomState(seed)
    y = y.astype(int)
    N = len(y)
    classes = np.unique(y)
    per_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(per_class[c])

    folds_per_class = {c: np.array_split(per_class[c], k) for c in classes}
    folds = []
    for f in range(k):
        test_idx = np.concatenate([folds_per_class[c][f] for c in classes])
        rest_idx = np.setdiff1d(np.arange(N), test_idx, assume_unique=False)

        # Build validation
        val_parts = []
        for c in classes:
            remaining_c = np.array([i for i in rest_idx if y[i] == c])
            rng.shuffle(remaining_c)
            take = max(8, int(math.ceil(val_frac_of_rest * len(remaining_c)))) if len(remaining_c) > 0 else 0
            take = min(take, len(remaining_c))
            val_parts.append(remaining_c[:take])
        val_idx = np.concatenate(val_parts) if len(val_parts) > 0 else np.array([], dtype=int)

        train_idx = np.setdiff1d(rest_idx, val_idx, assume_unique=False)
        folds.append((train_idx, val_idx, test_idx))
    return folds


def train(cfg: TrainConfig):
    # load processed data
    data_path = Path("processed_data") / f"{cfg.patient_id}_classification_data.npy"
    arr = np.load(str(data_path), allow_pickle=True)

    # unwrap
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        X = np.stack([d["signals"] for d in arr], axis=0).astype(np.float32)  # [N,T,C]
        y_names = [d["label"] for d in arr]
        y = np.array([CLS2IDX.get(lbl if isinstance(lbl, str) else str(lbl), 2) for lbl in y_names], dtype=np.int64)
    else:
        raise RuntimeError("Unsupported processed_data format")

    print(f"[DEBUG] Built dataset: X=({X.shape[0]}, {X.shape[1]}, {X.shape[2]}), y=({y.shape[0]},), num_channels={X.shape[2]}")
    uniq, cnts = np.unique(y, return_counts=True)
    dist = {IDX2CLS[int(k)]: int(v) for k, v in zip(uniq, cnts)}
    print(f"[DEBUG] Label distribution: {dist}")

    print(f"[INFO] [CV] Running Stratified {cfg.cv_folds}-Fold cross-validation")
    folds = stratified_kfold_indices(y, cfg.cv_folds, seed=42, val_frac_of_rest=cfg.val_frac_of_rest)

    all_acc, all_f1 = [], []
    sum_cm = np.zeros((3, 3), dtype=int)

    for i, (tr, va, te) in enumerate(folds, 1):
        print(f"[INFO] [CV] Fold {i}/{cfg.cv_folds} — train={len(tr)}, val={len(va)}, test={len(te)}")
        res = train_one_split(cfg, X, y, tr, va, te)
        all_acc.append(res["test_acc"])
        all_f1.append(res["test_f1"])
        sum_cm += res["cm"]

        print(f"Fold {i} — Test Accuracy: {res['test_acc']*100:.2f}%  |  Macro-F1: {res['test_f1']*100:.2f}%")
        print("Confusion Matrix (fold):")
        print(res["cm"])

    acc_mean = float(np.mean(all_acc))
    acc_std  = float(np.std(all_acc))
    f1_mean  = float(np.mean(all_f1))
    f1_std   = float(np.std(all_f1))

    print("\n===== Cross-Validation Summary =====")
    print(f"Folds: {cfg.cv_folds}")
    print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"Macro-F1: {f1_mean*100:.2f}% ± {f1_std*100:.2f}%")
    print("Summed Confusion Matrix over folds:")
    print(sum_cm)
