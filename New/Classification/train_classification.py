#train_classification.py
import os
from typing import Tuple, Dict, Any, List

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ===============================
# Model (48→48 convs, LSTM input=48)
# ===============================
class LSTMClassifier(nn.Module):
    def __init__(self, in_channels: int, lstm_hidden: int = 128, num_classes: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, 48, kernel_size=15, stride=1, padding=0)
        self.bn1   = nn.BatchNorm1d(48)
        self.conv2 = nn.Conv1d(48, 48, kernel_size=5, stride=1, padding=0)
        self.bn2   = nn.BatchNorm1d(48)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=48,          # keep input=48 to match best run
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc_hidden  = nn.Linear(lstm_hidden * 2, 128)
        self.dropout_fc = nn.Dropout(p=dropout_p)
        self.fc_out     = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x):  # x: (B, T, C)
        # conv expects (B, C, T)
        x = x.permute(0, 2, 1)                    # (B, C, T)
        x = self.relu(self.bn1(self.conv1(x)))    # (B, 48, T-14)
        x = self.relu(self.bn2(self.conv2(x)))    # (B, 48, T-18)
        x = self.pool(x)                          # (B, 48, T')
        x = x.permute(0, 2, 1)                    # (B, T', 48)

        lstm_out, _ = self.lstm(x)                # (B, T', 2H)
        scores  = self.attn(lstm_out).squeeze(-1) # (B, T')
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * weights, dim=1)  # (B, 2H)

        z = self.relu(self.fc_hidden(context))
        z = self.dropout_fc(z)
        logits = self.fc_out(z)
        return logits


# ===============================
# Data loading utilities
# ===============================
EXPECTED_T = 300
EXPECTED_C = 48
KNOWN_LABELS = ["HAARYE", "TUT", "OTHER"]


def _unwrap_scalar_ndarray(o: Any, max_depth: int = 20) -> Any:
    depth = 0
    while isinstance(o, np.ndarray) and o.shape == ():
        o = o.item()
        depth += 1
        if depth >= max_depth:
            break
    return o


def _collect_arrays(obj: Any, sink: List[np.ndarray]):
    obj = _unwrap_scalar_ndarray(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_arrays(v, sink)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_arrays(v, sink)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.shape == ():
                _collect_arrays(obj.item(), sink)
            else:
                for v in obj.flat:
                    _collect_arrays(v, sink)
        else:
            sink.append(obj)


def _pick_window_from_container(sample: Any) -> Tuple[np.ndarray, Any]:
    s = _unwrap_scalar_ndarray(sample)

    if isinstance(s, dict):
        keys = {k.lower(): k for k in s.keys()}
        x_keys = ["x", "data", "window", "windows", "features", "feat"]
        y_keys = ["y", "label", "labels", "target", "class"]

        X = None
        y = None
        for k in x_keys:
            if k in keys:
                X = _unwrap_scalar_ndarray(s[keys[k]])
                break
        for k in y_keys:
            if k in keys:
                y = _unwrap_scalar_ndarray(s[keys[k]])
                break
        if X is None:
            arrays: List[np.ndarray] = []
            _collect_arrays(s, arrays)
            arrays2d = [a for a in arrays if isinstance(a, np.ndarray) and a.ndim >= 2]
            if arrays2d:
                X = arrays2d[0]
        return (np.asarray(X) if X is not None else None, y)

    if isinstance(s, (list, tuple)) and len(s) >= 2:
        a = _unwrap_scalar_ndarray(s[0])
        b = _unwrap_scalar_ndarray(s[1])
        arr_a = a if isinstance(a, np.ndarray) else None
        arr_b = b if isinstance(b, np.ndarray) else None

        if arr_a is not None and arr_a.ndim >= 2 and (arr_b is None or arr_b.ndim < 2):
            return np.asarray(arr_a), b
        if arr_b is not None and arr_b.ndim >= 2 and (arr_a is None or arr_a.ndim < 2):
            return np.asarray(arr_b), a

        if arr_a is not None and arr_b is not None:
            if arr_a.size >= arr_b.size:
                return np.asarray(arr_a), b
            else:
                return np.asarray(arr_b), a

    if isinstance(s, np.ndarray) and s.ndim >= 2:
        return np.asarray(s), None

    return None, None


def _extract_from_object_samples(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    windows: List[np.ndarray] = []
    labels: List[Any] = []

    for sample in arr:
        X_i, y_i = _pick_window_from_container(sample)
        if X_i is None:
            continue
        windows.append(np.asarray(X_i))
        labels.append(y_i)

    if len(windows) == 0:
        raise ValueError("Found a 1-D object array but could not extract any (window,label) pairs.")

    fixed = []
    for w in windows:
        w = np.asarray(w)
        if w.ndim == 3 and w.shape[0] == 1:
            w = w[0]
        if w.ndim != 2:
            raise ValueError(f"Each window must be 2D per sample; got shape {w.shape}")
        fixed.append(w.astype(np.float32))
    windows = fixed

    try:
        X = np.stack(windows, axis=0)
    except Exception:
        a_shapes = [w.shape for w in windows]
        fixed2 = []
        for w in windows:
            if EXPECTED_T in w.shape and EXPECTED_C in w.shape:
                if w.shape == (EXPECTED_C, EXPECTED_T):
                    w = w.T
            fixed2.append(w)
        X = np.stack(fixed2, axis=0)

    if any(l is None for l in labels):
        raise ValueError("Some samples did not include labels; cannot build y vector.")
    y = np.array(labels)

    return X, y


def _extract_X_y_from_any(obj: Any) -> Tuple[np.ndarray, np.ndarray]:
    obj = _unwrap_scalar_ndarray(obj)

    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 1 and obj.size >= 2:
        return _extract_from_object_samples(obj)

    if isinstance(obj, dict):
        keys = {k.lower(): k for k in obj.keys()}
        X_keys = ["x", "data", "features", "windows"]
        y_keys = ["y", "labels", "target", "targets"]

        X = y = None
        for k in X_keys:
            if k in keys:
                X = _unwrap_scalar_ndarray(obj[keys[k]])
                break
        for k in y_keys:
            if k in keys:
                y = _unwrap_scalar_ndarray(obj[keys[k]])
                break
        if X is not None and y is not None:
            return np.asarray(X), np.asarray(y)

    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        a = _unwrap_scalar_ndarray(obj[0])
        b = _unwrap_scalar_ndarray(obj[1])
        A = np.asarray(a) if isinstance(a, (list, tuple, np.ndarray)) else None
        B = np.asarray(b) if isinstance(b, (list, tuple, np.ndarray)) else None
        if A is not None and B is not None:
            if (A.ndim >= 3 and B.ndim <= 2) or (A.size >= B.size):
                return A, B
            else:
                return B, A

    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return _extract_X_y_from_any(obj.item())

    arrays: List[np.ndarray] = []
    _collect_arrays(obj, arrays)
    three_d = [a for a in arrays if a.ndim == 3]
    if three_d:
        X_candidate = max(three_d, key=lambda a: a.size)
        N = X_candidate.shape[0]
        y_candidates = [a for a in arrays if a.ndim == 1 and a.shape[0] == N]
        if y_candidates:
            return np.asarray(X_candidate), np.asarray(y_candidates[0])

    two_d = [a for a in arrays if a.ndim == 2]
    if two_d:
        X_candidate = max(two_d, key=lambda a: a.size)
        N = X_candidate.shape[0]
        y_candidates = [a for a in arrays if a.ndim == 1 and a.shape[0] == N]
        if y_candidates:
            return np.asarray(X_candidate), np.asarray(y_candidates[0])

    raise ValueError("Could not extract X and y from the provided file. "
                     "Consider saving as a dict {'X': X, 'y': y} with np.save(..., allow_pickle=True).")


def _maybe_fix_axes(X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise ValueError(f"X must be a numpy array, got {type(X)}")
    if X.ndim == 2:
        N, D = X.shape
        if D == EXPECTED_T * EXPECTED_C:
            print(f"[WARN] Detected flattened windows (N,{D}); reshaping to (N,{EXPECTED_T},{EXPECTED_C}).")
            return X.reshape(N, EXPECTED_T, EXPECTED_C)
        raise ValueError(f"X must be 3D (N,T,C) or (N,C,T); got 2D {X.shape}.")
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,T,C) or (N,C,T), got {X.shape}")
    n, a, b = X.shape
    if a == EXPECTED_T and b == EXPECTED_C:
        return X
    if a == EXPECTED_C and b == EXPECTED_T:
        print("[WARN] Detected (N,C,T); transposing to (N,T,C).")
        return np.transpose(X, (0, 2, 1))
    return X


def _coerce_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[Any, int]]:
    if y.ndim != 1:
        y = y.reshape(-1)
    if np.issubdtype(y.dtype, np.integer):
        classes = sorted(np.unique(y).tolist())
        mapping = {c: int(c) for c in classes}
        return y.astype(np.int64), mapping
    uniq = list(map(lambda z: z if isinstance(z, str) else str(z), np.unique(y)))
    order = KNOWN_LABELS if set(uniq) == set(KNOWN_LABELS) else sorted(uniq)
    str2id = {s: i for i, s in enumerate(order)}
    y_int = np.array([str2id[str(v)] for v in y], dtype=np.int64)
    return y_int, str2id


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[Any, int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    print(f"[DEBUG] Loading {path}")
    print(f"[DEBUG] Raw top-level: type={type(loaded)}, "
          f"shape={getattr(loaded, 'shape', None)}, dtype={getattr(loaded, 'dtype', None)}")

    if isinstance(loaded, np.lib.npyio.NpzFile):
        X = _unwrap_scalar_ndarray(loaded["X"])
        y = _unwrap_scalar_ndarray(loaded["y"])
        loaded.close()
    else:
        X, y = _extract_X_y_from_any(loaded)

    if isinstance(X, (list, tuple)):
        X = np.array(X)
    if isinstance(y, (list, tuple)):
        y = np.array(y)

    X = np.asarray(X)
    y = np.asarray(y)
    print(f"[DEBUG] After unwrap: X.shape={getattr(X, 'shape', None)}, y.shape={getattr(y, 'shape', None)}, "
          f"X.ndim={getattr(X, 'ndim', None)}, y.ndim={getattr(y, 'ndim', None)}")

    X = _maybe_fix_axes(X)
    y_int, mapping = _coerce_labels(y)

    if X.shape[0] != y_int.shape[0]:
        raise ValueError(f"N mismatch: X={X.shape[0]} vs y={y_int.shape[0]}")

    X = X.astype(np.float32)
    y_int = y_int.astype(np.int64)
    return X, y_int, mapping


# ===============================
# Torch dataset / training
# ===============================
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean=None, std=None):
        self.X = X
        self.y = y
        if mean is None or std is None:
            mean = X.mean(axis=(0, 1))
            std = X.std(axis=(0, 1)) + 1e-8
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = (self.X[idx] - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


def _build_sampler(y_train: np.ndarray):
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weight_per_class = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weight_per_class[c] for c in y_train], dtype=np.float32)
    weights_list = [weight_per_class[c] for c in sorted(weight_per_class.keys())]
    print(f"[DEBUG] Class weights (balanced sampler): {weights_list}")
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m, y_true, y_pred


def _print_channel_norm_debug(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    print("[DEBUG] Normalizing each channel (zero mean, unit std)...")
    normed = (X - mean) / std
    C = normed.shape[2]
    for c in range(C):
        mu = normed[:, :, c].mean()
        sg = normed[:, :, c].std()
        print(f"[DEBUG] Channel {c}: mean={mu:+.3f}, std={sg:.3f}")


def _stratified_80_10_10(X: np.ndarray, y: np.ndarray, seed: int):
    n = X.shape[0]
    val_cnt = int(np.ceil(n * 0.10))
    test_cnt = int(np.ceil(n * 0.10))
    tv_cnt = val_cnt + test_cnt

    sss = StratifiedShuffleSplit(n_splits=1, test_size=tv_cnt, random_state=seed)
    train_idx, tv_idx = next(sss.split(X, y))

    # Split tv_idx into val and test with exact counts
    X_tv = X[tv_idx]
    y_tv = y[tv_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_cnt, random_state=seed)
    val_idx_local, test_idx_local = next(sss2.split(X_tv, y_tv))
    val_idx = tv_idx[val_idx_local]
    test_idx = tv_idx[test_idx_local]
    return train_idx, val_idx, test_idx


def run_training(
    patient_id: str,
    npy_path: str,
    batch_size: int = 64,
    lr: float = 5e-4,
    epochs: int = 60,
    device: str = "cpu",
    save_path: str = os.path.join("Classification", "models", "best_cls_lstm.pth"),
    seed: int = 42,
    early_stop_patience: int = 10
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[DEBUG] Training on patient: {patient_id}")
    X, y, mapping = load_dataset(npy_path)
    n, T, C = X.shape
    print(f"[DEBUG] Built dataset: X={X.shape}, y={y.shape}, num_channels={C}")

    inv_map = {v: k for k, v in mapping.items()}
    labels_print = [inv_map[int(k)] if int(k) in inv_map else int(k) for k in np.unique(y)]
    counts = [int((y == k).sum()) for k in np.unique(y)]
    print(f"[DEBUG] Label distribution: {dict(zip(labels_print, counts))}")
    print(f"[DEBUG] Total windows: {n}")

    train_idx, val_idx, test_idx = _stratified_80_10_10(X, y, seed=seed)
    print(f"[DEBUG] Train windows: {len(train_idx)}, Val windows: {len(val_idx)}, Test windows: {len(test_idx)}")

    # Train stats
    train_mean = X[train_idx].mean(axis=(0, 1))
    train_std  = X[train_idx].std(axis=(0, 1)) + 1e-8

    # Debug print per-channel normalization like the best run
    _print_channel_norm_debug(X, train_mean, train_std)

    ds_train = WindowDataset(X[train_idx], y[train_idx], mean=train_mean, std=train_std)
    ds_val   = WindowDataset(X[val_idx],   y[val_idx],   mean=train_mean, std=train_std)
    ds_test  = WindowDataset(X[test_idx],  y[test_idx],  mean=train_mean, std=train_std)

    sampler  = _build_sampler(y[train_idx])
    dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = LSTMClassifier(in_channels=C, lstm_hidden=128, num_classes=len(np.unique(y)), dropout_p=0.5).to(device)
    print("[DEBUG] Model Architecture:", model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_state = None
    last_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(ds_train)

        test_acc, _, _, _ = _evaluate(model, dl_test, device)
        val_acc, val_f1, _, _ = _evaluate(model, dl_val, device)

        print(f"Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}")
        print(f"[DEBUG] Epoch {epoch}: Test Accuracy = {test_acc*100:.2f}%")
        print(f"[DEBUG] Epoch {epoch}: Val Macro-F1 = {val_f1*100:.2f}%")

        if val_f1 > best_f1 + 1e-8:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            last_improve = epoch

        if epoch - last_improve >= early_stop_patience:
            print(f"[DEBUG] Early stopping at epoch {epoch} (no improvement in {early_stop_patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to: {save_path}")

    test_acc, _, y_true, y_pred = _evaluate(model, dl_test, device)
    cm = confusion_matrix(y_true, y_pred)

    # Pretty target names in the same order as class indices
    class_indices = sorted(np.unique(y_true).tolist())
    target_names = [str(inv_map[c]) if c in inv_map else str(c) for c in class_indices]
    report = classification_report(y_true, y_pred, digits=3, target_names=target_names)

    correct = int(test_acc * len(y_true))
    print(f"Test Accuracy: {test_acc*100:.2f}% ({correct}/{len(y_true)} windows correct)")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    return test_acc, best_f1
