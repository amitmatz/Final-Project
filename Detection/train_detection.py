# Detection/train_detection.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from defines import PATIENT_ID, PROCESSED_DATA_DIR
import copy

# חלק מהפרמטרים למודל (2 שכבות LSTM, 128 יחידות חבויות, Dropout 0.3)
TOTAL_HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
# שיעור Dropout עבור השכבה הצפופה (רגולריזציה נוספת)
FC_DROPOUT = 0.5
# מקדם דעיכת משקל (Weight Decay) עבור האופטימייזר (רגולריזציית L2)
WEIGHT_DECAY = 1e-4

# פרמטרים עבור שכבות הקונבולוציה (Conv1D) והשכבה הצפופה
CONV_CHANNELS = 64
CONV_KERNEL_SIZE1 = 15
CONV_KERNEL_SIZE2 = 5
POOL_KERNEL_SIZE = 2
FC_HIDDEN_SIZE = 128

# ניתן להפעיל מצב Debug כדי לקבל הדפסות מפורטות במהלך הריצה
DEBUG = True

# מודל LSTM משולב עם שכבות Conv1D לזיהוי דיבור (פלט 1 = דיבור, 0 = אין דיבור)
class LSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size=TOTAL_HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(LSTMDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # שכבות Conv1D לחילוץ מאפיינים מקומיים מהאות
        self.conv1 = nn.Conv1d(input_size, CONV_CHANNELS, CONV_KERNEL_SIZE1)
        self.conv2 = nn.Conv1d(CONV_CHANNELS, CONV_CHANNELS, CONV_KERNEL_SIZE2)
        self.pool = nn.MaxPool1d(POOL_KERNEL_SIZE)
        # שכבת LSTM: input_size = CONV_CHANNELS (לאחר ה-Conv), hidden_size = גודל hidden state, num_layers = מספר שכבות LSTM
        self.lstm = nn.LSTM(CONV_CHANNELS, hidden_size, num_layers, batch_first=True, dropout=DROPOUT)
        # שכבת fully connected בגודל 128 עם ReLU לפני הפלט
        self.fc_hidden = nn.Linear(hidden_size, FC_HIDDEN_SIZE)
        self.relu = nn.ReLU()
        # שכבת Dropout לאחר השכבה הצפופה לצורך רגולריזציה
        self.dropout_fc = nn.Dropout(FC_DROPOUT)
        # שכבה מלאה לפלט בינארי יחיד
        self.fc_out = nn.Linear(FC_HIDDEN_SIZE, 1)

    def forward(self, x):
        # x בצורה: (batch, seq_len, input_size)
        # העברת האות דרך שכבות הקונבולוציה (Conv1D) וה-Pooling
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        # אתחל hidden state ו-cell state התחלתי לאפסים עבור ה-LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # העבר את הקלט דרך ה-LSTM
        out, _ = self.lstm(x, (h0, c0))
        # קח את ה-hidden state האחרון של ה-LSTM (האחרון בציר הזמן) עבור כל batch
        last_out = out[:, -1, :]
        # העבר דרך השכבה הצפופה והפעל ReLU
        x = self.fc_hidden(last_out)
        x = self.relu(x)
        # החלת Dropout לצורך רגולריזציה לפני הפלט
        x = self.dropout_fc(x)
        # העבר דרך שכבת הפלט והחזר את הלוגיט (ערך לפני סיגמואיד)
        logit = self.fc_out(x)
        return logit

def train(config):
    # שליפת הגדרות הפציינט הנבחר מתוך ה-config
    patient_info = config["patients"][PATIENT_ID]
    csv_dir = os.path.join(PROCESSED_DATA_DIR, "csvs")  # תיקיית ה-CSV עם הערוצים (נוצרת ב-preprocessing)
    labels_csv = os.path.join(PROCESSED_DATA_DIR, "labels_aligned.csv")
    # טען פרמטרים מהתצורה, עם ברירת מחדל אם לא מוגדר
    sample_rate = patient_info.get("sample_rate", None)
    window_size = patient_info.get("window_size", 300)
    # חישוב stride מתוך overlap אם מוגדר
    if "overlap" in patient_info:
        overlap_frac = patient_info["overlap"]
        stride = int(window_size * (1 - overlap_frac))
    else:
        stride = patient_info.get("stride", window_size)
    batch_size = patient_info.get("batch_size", 32)
    learning_rate = patient_info.get("learning_rate", 0.001)
    num_epochs = patient_info.get("epochs", 20)

    if DEBUG:
        print(f"[DEBUG] Training on patient: {PATIENT_ID}")
        print(f"[DEBUG] Parameters - window_size: {window_size}, stride: {stride}, batch: {batch_size}, lr: {learning_rate}, epochs: {num_epochs}")

    # 1. טען את כל קבצי ה-CSV של הערוצים וצרו מערך נתונים מאוחד
    channel_dfs = []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        ch_path = os.path.join(csv_dir, fname)
        df_ch = pd.read_csv(ch_path)
        # ודא שהקובץ מכיל עמודות "time" ו-"signal"
        if "time" not in df_ch.columns or "signal" not in df_ch.columns:
            continue
        channel_dfs.append(df_ch["signal"])
        # ב-debug, נדפיס את גודל הערוץ וכמות הדגימות
        if DEBUG:
            print(f"[DEBUG] Loaded channel {fname} with {len(df_ch)} samples")
    if len(channel_dfs) == 0:
        raise RuntimeError(f"No CSV files found in {csv_dir}")
    # איחוד הערוצים: ערימת כל סדרות האותות כעמודות (numpy array)
    signals = np.column_stack([np.array(ch) for ch in channel_dfs])  # צורה: (N_samples, N_channels)
    time = pd.read_csv(os.path.join(csv_dir, sorted(os.listdir(csv_dir))[0]))["time"].values  # וקטור זמנים (זהה בכל הערוצים)
    num_channels = signals.shape[1]
    if DEBUG:
        print(f"[DEBUG] Combined signals shape: {signals.shape} (samples x channels)")

    # נרמול כל ערוץ (ממוצע 0, סטיית תקן 1 לכל ערוץ)
    if DEBUG:
        print("[DEBUG] Normalizing each channel (zero mean, unit std)...")
    for ch in range(num_channels):
        mean_val = np.mean(signals[:, ch])
        std_val = np.std(signals[:, ch])
        if std_val != 0:
            signals[:, ch] = (signals[:, ch] - mean_val) / std_val
        else:
            signals[:, ch] = signals[:, ch] - mean_val
        if DEBUG:
            new_mean = signals[:, ch].mean()
            new_std = signals[:, ch].std()
            print(f"[DEBUG] Channel {ch}: mean={new_mean:.3f}, std={new_std:.3f}")

    # 2. טען את קובץ התוויות המיושר לזמן
    df_labels = pd.read_csv(labels_csv)
    # ודא שקיימות העמודות הנדרשות בתוויות
    if "start_adj" not in df_labels.columns or "end_adj" not in df_labels.columns or "label" not in df_labels.columns:
        raise RuntimeError("labels_aligned.csv does not contain required columns")

    # 3. חלץ חלונות מכל קטע מתויג לפי הגודל והחפיפה שהוגדרו
    X_windows = []
    y_labels = []
    for _, row in df_labels.iterrows():
        start_time = row["start_adj"]
        end_time = row["end_adj"]
        label_str = str(row["label"])  # ודא שהתוית היא מחרוזת
        # קבע האם קטע זה כולל דיבור (label שאינה "none"/"nospeech" נחשב כדיבור)
        is_speech = label_str.lower() not in ["none", "nospeech"]
        # מצא אינדקס התחלה וסיום במערך הזמנים (מניחים שהזמן ממויין בסדר עולה)
        start_idx = np.searchsorted(time, start_time, side="left")
        end_idx = np.searchsorted(time, end_time, side="right") - 1
        if start_idx >= len(time) or end_idx < 0:
            continue  # טווח מחוץ לגבולות האות
        if end_idx <= start_idx:
            continue  # טווח ריק או שלילי
        segment_length = end_idx - start_idx + 1
        if segment_length < window_size:
            # דלג על קטעים קצרים מדי שאינם מכסים חלון שלם
            continue
        # חלץ תת-חלונות בגודל window בתוך הטווח [start_idx, end_idx] עם חפיפה (stride) שהוגדר
        for i in range(start_idx, end_idx - window_size + 1, stride):
            j = i + window_size  # חלון [i, j)
            window_data = signals[i:j, :]
            X_windows.append(window_data)
            y_labels.append(1 if is_speech else 0)
    X_windows = np.array(X_windows)
    y_labels = np.array(y_labels)
    if DEBUG:
        print(f"[DEBUG] Total windows extracted: {len(X_windows)}")
    if len(X_windows) == 0:
        raise RuntimeError("No windows were extracted for training – check window_size or labels coverage")

    # 4. פצל לסט אימון/בדיקה לפי יחס (ברירת מחדל 90%/10% אם לא הוגדר אחרת)
    train_ratio = 0.9
    train_ratio = config.get("train_split", train_ratio)  # ניתן להגדיר יחס גלובלי בקובץ התצורה
    num_samples = len(X_windows)
    train_size = int(num_samples * train_ratio)
    # ערבוב אקראי של האינדקסים לפני הפיצול
    indices = np.random.permutation(num_samples)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    X_train = X_windows[train_idx]
    X_test = X_windows[test_idx]
    y_train = y_labels[train_idx]
    y_test = y_labels[test_idx]
    if DEBUG:
        print(f"[DEBUG] Train windows: {len(X_train)}, Test windows: {len(X_test)}")

    # חשב משקולות עבור אי-איזון בין מחלקות (Neg/Pos)
    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    if DEBUG:
        print(f"[DEBUG] pos_count: {pos_count}, neg_count: {neg_count}, pos_weight: {pos_weight:.2f}")

    # 5. המר את הנתונים למבני PyTorch (טנזורים)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # הוסף מימד נוסף לווקטור התוויות
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # צור DataLoader עבור אימון (מיני-באצ'ים)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 6. הגדרת המודל, פונקציית עלות ואופטימייזר
    model = LSTMDetector(input_size=num_channels, hidden_size=TOTAL_HIDDEN_SIZE, num_layers=NUM_LAYERS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    if DEBUG:
        print(f"[DEBUG] Model Architecture: {model}")

    # הגדרת משתנים לעצירת אימון מוקדמת (Early Stopping)
    best_accuracy = 0.0
    best_state = None
    epochs_no_improve = 0
    patience = 10

    # 7. לולאת אימון
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()                   # אפס את הגרדיאנטים הקודמים
            outputs = model(batch_X)                # העבר קדימה (forward pass)
            loss = criterion(outputs, batch_y)      # חשב את פונקציית העלות
            loss.backward()                         # בצע backpropagation
            optimizer.step()                        # עדכן את משקלי המודל
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        if DEBUG or epoch % 5 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # הערכה על סט הבדיקה עבור עצירת אימון מוקדמת
        model.eval()
        with torch.no_grad():
            outputs_test = model(X_test_tensor)
            probs_test = torch.sigmoid(outputs_test)
            preds_test = (probs_test >= 0.5).float()
            correct_test = (preds_test == y_test_tensor).sum().item()
            total_test = y_test_tensor.size(0)
            accuracy_test = correct_test / total_test if total_test > 0 else 0.0
        if DEBUG:
            print(f"[DEBUG] Epoch {epoch}: Test Accuracy = {accuracy_test * 100:.2f}%")
        model.train()

        # עדכון סטטוס Early Stopping
        if accuracy_test > best_accuracy:
            best_accuracy = accuracy_test
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if DEBUG:
                print(f"[DEBUG] Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
            break

    # אם נעצר מוקדם, טען את המודל הטוב ביותר לצורך הערכה סופית
    if best_state is not None:
        model.load_state_dict(best_state)

    # 8. הערכת המודל על סט הבדיקה והדפסת הביצועים
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct = (preds == y_test_tensor).sum().item()
        total = y_test_tensor.size(0)
        accuracy = correct / total if total > 0 else 0.0
        # חשב מדדי ביצוע: True/False Positive/Negative
        y_true = y_test_tensor
        y_pred = preds
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total} windows correct)")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%")
