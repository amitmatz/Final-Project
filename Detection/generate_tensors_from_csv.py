import os
import numpy as np
import pandas as pd
import torch
from defines import PROCESSED_DATA_DIR

# ============================
# הגדרת נתיבים
# ============================
LABELS_PATH = os.path.join(PROCESSED_DATA_DIR, "labels_aligned.csv")
CSV_DIR = os.path.join(PROCESSED_DATA_DIR, "csvs")
OUTPUT_DIR = os.path.join("Detection", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# פרמטרים
# ============================
FIXED_LENGTH = 300

# ============================
# טעינת תוויות
# ============================
print(f"[INFO] Loading labels from {LABELS_PATH}")
labels_df = pd.read_csv(LABELS_PATH)

X = []
y = []
skipped = 0

# ============================
# קריאת כל קבצי ה-CSV לפי ערוצים
# ============================
for fname in sorted(os.listdir(CSV_DIR)):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(CSV_DIR, fname))

    for _, row in labels_df.iterrows():
        label_str = row["label"]
        start, end = row["start_adj"], row["end_adj"]
        is_speech = label_str.lower() != "none"

        seg = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
        signal = seg["signal"].values

        if len(signal) < FIXED_LENGTH:
            skipped += 1
            continue

        for i in range(0, len(signal) - FIXED_LENGTH + 1, FIXED_LENGTH):
            chunk = signal[i:i + FIXED_LENGTH]
            X.append(chunk)
            y.append(1 if is_speech else 0)

print(f"[INFO] Total segments: {len(X)}, Skipped: {skipped}")

X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
X_tensor = X_tensor.unsqueeze(-1)  # ← הוספת מימד לערוץ
y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

# שמירה
torch.save(X_tensor, os.path.join(OUTPUT_DIR, "X_tensor.pt"))
torch.save(y_tensor, os.path.join(OUTPUT_DIR, "y_tensor.pt"))
print("✅ Saved X_tensor.pt and y_tensor.pt to Detection/models/")
