# =============================
# train_classification.py
# =============================
import os
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ------------ LOG HELPERS ------------
def _normpath(p: str) -> str:
    try: return os.path.normpath(p)
    except: return p
def LOG_DEBUG(msg: str): print(f"[DEBUG] {msg}")
def LOG_INFO(msg: str):  print(f"[INFO] {msg}")
def LOG_WARN(msg: str):  print(f"[WARN] {msg}")

# ------------ LABELS ------------
LABEL_NAMES: List[str] = ["HAARYE", "TUT", "OTHER"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

# ------------ MODEL ------------
class ConvBlock1D(nn.Module):
    """Conv1D -> Norm -> ReLU (optional MaxPool)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1,
                 pool: bool = False, norm_type: str = "batch"):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1,
                              padding=padding, dilation=dilation)
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_ch)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        else:
            raise ValueError("norm_type must be 'batch' or 'group'")
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else None

    def forward(self, x):  # [B, C, T]
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class LSTMClassifier(nn.Module):
    """CNN + BiLSTM + (optional) MHA + additive attention head. Input: x [B,T,C] -> logits [B,num_classes]"""
    def __init__(self, num_channels: int = 48, num_classes: int = 3,
                 lstm_hidden: int = 128, lstm_layers: int = 1,
                 attn_hidden: int = 64, dropout: float = 0.5,
                 use_mha: bool = False, mha_heads: int = 4,
                 conv_norm: str = "batch",
                 k1: int = 15, k2: int = 5, use_dilation: bool = False):
        super().__init__()
        d1 = 1
        d2 = (2 if use_dilation else 1)

        # כמו במקור: שתי קונבולוציות עם (15,5), BN, ופולינג פעם אחת אחרי Conv2
        self.conv1 = ConvBlock1D(num_channels, num_channels, kernel_size=k1, dilation=d1, pool=False, norm_type=conv_norm)
        self.conv2 = ConvBlock1D(num_channels, num_channels, kernel_size=k2, dilation=d2, pool=True,  norm_type=conv_norm)

        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        enc_dim = lstm_hidden * 2

        self.use_mha = use_mha
        if use_mha:
            self.mha    = nn.MultiheadAttention(embed_dim=enc_dim, num_heads=mha_heads, batch_first=True)
            self.mha_ln = nn.LayerNorm(enc_dim)

        self.attn = nn.Sequential(nn.Linear(enc_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1))
        self.fc_hidden  = nn.Linear(enc_dim, 128)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_out     = nn.Linear(128, num_classes)

    def temporal_attention_pool(self, h):  # h: [B,T,D] -> [B,D]
        scores  = self.attn(h)                               # [B,T,1]
        weights = torch.softmax(scores.squeeze(-1), dim=-1)  # [B,T]
        ctx     = torch.einsum("btd,bt->bd", h, weights)     # [B,D]
        return ctx

    def forward(self, x):  # x: [B,T,C]
        x = x.transpose(1, 2)      # [B,C,T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)      # [B,T,C]

        out, _ = self.lstm(x)      # [B,T,2H]
        if self.use_mha:
            mha_out, _ = self.mha(out, out, out)
            out = self.mha_ln(out + mha_out)

        feat = self.temporal_attention_pool(out)
        z = F.relu(self.fc_hidden(feat))
        z = self.dropout_fc(z)
        logits = self.fc_out(z)
        return logits

# ------------ UTILS ------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_channel_norm_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert X.ndim == 3, f"Expected (N,T,C), got {X.shape}"
    mu = X.mean(axis=(0, 1)); sigma = X.std(axis=(0, 1))
    sigma[sigma < 1e-6] = 1.0
    return mu, sigma

def apply_channel_norm(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu.reshape(1, 1, -1)) / sigma.reshape(1, 1, -1)

class NPWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))   # [N,T,C]
        self.y = torch.from_numpy(y.astype(np.int64))     # [N]
        assert self.X.ndim == 3 and self.y.ndim == 1 and self.X.shape[0] == self.y.shape[0]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ------------ LOSSES ------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.register_buffer('alpha', alpha if alpha is not None else None)
        self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else (loss.sum() if self.reduction == 'sum' else loss)

def class_balanced_alpha(labels: np.ndarray, num_classes: int, beta: float = 0.9999) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    eff[eff == 0.0] = 1e-6
    w = (counts.sum() / (num_classes * eff))
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)

# ------------ DATA LOADING ------------
def load_patient_npy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    LOG_DEBUG(f"Loading { _normpath(path) }")
    raw = np.load(path, allow_pickle=True)
    LOG_DEBUG(f"Raw top-level: type={type(raw)}, shape={getattr(raw,'shape',None)}, dtype={getattr(raw,'dtype',None)}")

    if isinstance(raw, np.ndarray) and raw.dtype == object:
        X_list, y_list = [], []
        for item in raw.tolist():
            if not isinstance(item, dict):
                raise ValueError('Expected dict items in object array')
            sig = item.get('signals'); lab = item.get('label')
            if isinstance(lab, str): lab = LABEL_TO_ID.get(lab, LABEL_TO_ID['OTHER'])
            elif isinstance(lab, (np.integer, int)): lab = int(lab)
            else: raise ValueError('Unsupported label type in npy list-of-dicts')
            X_list.append(sig); y_list.append(lab)
        X = np.stack(X_list, axis=0); y = np.array(y_list, dtype=np.int64)
    elif isinstance(raw, dict) or (hasattr(raw, 'item') and isinstance(raw.item(), dict)):
        d = raw if isinstance(raw, dict) else raw.item()
        X, y = d['X'], d['y']
    else:
        raise ValueError('Unsupported npy format. Expect list-of-dicts {signals,label} or dict {X,y}.')

    LOG_DEBUG(f"After unwrap: X.shape={X.shape}, y.shape={y.shape}, X.ndim={X.ndim}, y.ndim={y.ndim}")
    if X.shape[1] < X.shape[2]:
        LOG_WARN("Detected (N,C,T); transposing to (N,T,C).")
        X = np.transpose(X, (0, 2, 1))
        LOG_DEBUG(f"After transpose: X.shape={X.shape}")
    return X, y

# ------------ TRAIN / EVAL ------------
@dataclass
class TrainConfig:
    data_path: str
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 60
    patience: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Defaults to mirror your best run:
    use_sampler: bool = True          # YES sampler
    use_focal: bool = False           # CE only
    gamma_focal: float = 2.0
    beta_cb: float = 0.9999
    grad_clip: float = 1.0
    tau_logit_adjust: float = 0.0     # NO logit adj
    eval_test_each_epoch: bool = True
    save_dir: str = 'Classification/models'
    save_name: str = 'best_cls_lstm.pth'
    # Model options (mirroring your original arch)
    conv_norm: str = "batch"
    k1: int = 15
    k2: int = 5
    use_dilation: bool = False
    use_mha: bool = False
    lstm_layers: int = 1

def build_datasets(cfg: TrainConfig):
    X, y = load_patient_npy(cfg.data_path)

    # Label distribution (global)
    counts = np.bincount(y, minlength=len(LABEL_NAMES))
    label_dist = {LABEL_NAMES[i]: int(counts[i]) for i in range(len(LABEL_NAMES))}
    LOG_DEBUG(f"Built dataset: X={X.shape}, y={y.shape}, num_channels={X.shape[-1]}")
    LOG_DEBUG(f"Label distribution: {label_dist}")

    # Split (80/10/10)
    N = X.shape[0]
    LOG_DEBUG(f"Total windows: {N}")
    idx = np.arange(N); rng = np.random.default_rng(cfg.seed); rng.shuffle(idx)
    n_train = int(0.8 * N); n_val = int(0.1 * N)
    train_idx = idx[:n_train]; val_idx = idx[n_train:n_train + n_val]; test_idx = idx[n_train + n_val:]
    LOG_DEBUG(f"Train windows: {len(train_idx)}, Val windows: {len(val_idx)}, Test windows: {len(test_idx)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Normalization (train μ,σ)
    LOG_DEBUG("Normalizing each channel (zero mean, unit std)...")
    mu, sigma = compute_channel_norm_stats(X_train)
    X_train_n = apply_channel_norm(X_train, mu, sigma)
    X_val_n   = apply_channel_norm(X_val,   mu, sigma)
    X_test_n  = apply_channel_norm(X_test,  mu, sigma)

    # Print per-channel stats AFTER applying (for parity with your logs)
    X_all_n = np.concatenate([X_train_n, X_val_n, X_test_n], axis=0)
    C = X_all_n.shape[-1]
    for c in range(C):
        m = X_all_n[:, :, c].mean(); s = X_all_n[:, :, c].std()
        LOG_DEBUG(f"Channel {c}: mean={m:+0.3f}, std={s:0.3f}")

    ds_train = NPWindowDataset(X_train_n, y_train)
    ds_val   = NPWindowDataset(X_val_n,   y_val)
    ds_test  = NPWindowDataset(X_test_n,  y_test)

    stats = {'mu': mu.tolist(), 'sigma': sigma.tolist(),
             'class_counts': np.bincount(y_train, minlength=len(LABEL_NAMES)).tolist()}
    return ds_train, ds_val, ds_test, stats

def make_loader(dataset: Dataset, cfg: TrainConfig, class_counts: Optional[np.ndarray] = None, shuffle: bool = True):
    if cfg.use_sampler and class_counts is not None:
        class_counts = np.asarray(class_counts, dtype=np.float64)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        sample_weights = class_weights[dataset.y.numpy()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
        LOG_DEBUG(f"Class weights (balanced sampler): {class_weights.tolist()}")
        return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, tau: float, priors_log: Optional[torch.Tensor] = None):
    model.eval()
    all_logits, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        if priors_log is not None and tau is not None and tau > 1e-9:
            logits = logits - tau * priors_log
        all_logits.append(logits.cpu()); all_targets.append(yb.cpu())
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds   = logits.argmax(dim=1)
    acc     = accuracy_score(targets.numpy(), preds.numpy())
    macro_f1= f1_score(targets.numpy(), preds.numpy(), average='macro')
    return acc, macro_f1, targets.numpy(), preds.numpy()

def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    ds_train, ds_val, ds_test, stats = build_datasets(cfg)
    class_counts = np.bincount(ds_train.y.numpy(), minlength=len(LABEL_NAMES))

    # Priors (for optional logit adjust – default off)
    priors = class_counts / class_counts.sum()
    priors_log = torch.log(torch.tensor(priors, dtype=torch.float32)).to(cfg.device)

    train_loader = make_loader(ds_train, cfg, class_counts, shuffle=True)
    val_loader   = make_loader(ds_val,   cfg, class_counts=None, shuffle=False)
    test_loader  = make_loader(ds_test,  cfg, class_counts=None, shuffle=False)

    model = LSTMClassifier(
        num_channels=ds_train.X.shape[-1], num_classes=len(LABEL_NAMES),
        lstm_hidden=128, lstm_layers=cfg.lstm_layers,
        use_mha=cfg.use_mha, conv_norm=cfg.conv_norm,
        k1=cfg.k1, k2=cfg.k2, use_dilation=cfg.use_dilation
    ).to(cfg.device)

    LOG_DEBUG(f"Model Architecture: {model.__class__.__name__}(\n  (conv1): Conv1d({ds_train.X.shape[-1]}, {ds_train.X.shape[-1]}, kernel_size=({cfg.k1},), stride=(1,))\n  (bn1): BatchNorm1d({ds_train.X.shape[-1]})\n  (conv2): Conv1d({ds_train.X.shape[-1]}, {ds_train.X.shape[-1]}, kernel_size=({cfg.k2},), stride=(1,))\n  (bn2): BatchNorm1d({ds_train.X.shape[-1]})\n  (pool): MaxPool1d(kernel_size=2, stride=2)\n  (lstm): LSTM({ds_train.X.shape[-1]}, 128, batch_first=True, bidirectional=True)\n  (attn): Sequential(\n    (0): Linear(in_features=256, out_features=64, bias=True)\n    (1): Tanh()\n    (2): Linear(in_features=64, out_features=1, bias=True)\n  )\n  (fc_hidden): Linear(in_features=256, out_features=128, bias=True)\n  (dropout_fc): Dropout(p=0.5, inplace=False)\n  (fc_out): Linear(in_features=128, out_features={len(LABEL_NAMES)}, bias=True)\n)")

    # Loss: כש-sampler פעיל -> בלי משקולות בכרוס-אנטרופי (כדי לא להכפיל איזון)
    alpha_cb = class_balanced_alpha(ds_train.y.numpy(), num_classes=len(LABEL_NAMES), beta=cfg.beta_cb).to(cfg.device)
    if cfg.use_sampler:
        criterion = nn.CrossEntropyLoss(weight=None)
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha_cb)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_f1 = -1.0
    epochs_no_improve = 0
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device); yb = yb.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(ds_train)
        print(f"Epoch [{epoch}/{cfg.max_epochs}]  Loss: {train_loss:.4f}")

        # Logs כמו במקור
        if cfg.eval_test_each_epoch:
            test_acc_e, _, _, _ = evaluate(model, test_loader, cfg.device, tau=cfg.tau_logit_adjust, priors_log=priors_log)
            LOG_DEBUG(f"Epoch {epoch}: Test Accuracy = {test_acc_e*100:.2f}%")
        _, val_f1_e, _, _ = evaluate(model, val_loader, cfg.device, tau=cfg.tau_logit_adjust, priors_log=priors_log)
        LOG_DEBUG(f"Epoch {epoch}: Val Macro-F1 = {val_f1_e*100:.2f}%")

        scheduler.step(val_f1_e)

        if val_f1_e > best_f1 + 1e-6:
            best_f1 = val_f1_e
            epochs_no_improve = 0
            torch.save({'model_state': model.state_dict(), 'cfg': cfg.__dict__, 'stats': stats, 'label_names': LABEL_NAMES}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                LOG_DEBUG(f"Early stopping at epoch {epoch} (no improvement in {cfg.patience} epochs).")
                break

    print(f"Saved checkpoint to: { _normpath(ckpt_path) }")

    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state['model_state'])

    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, cfg.device, tau=cfg.tau_logit_adjust, priors_log=priors_log)
    correct = int((y_true == y_pred).sum()); total = int(len(y_true))
    print(f"Test Accuracy: {test_acc*100:.2f}% ({correct}/{total} windows correct)")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    return ckpt_path
