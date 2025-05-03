# Detection/prepare_detection_data.py

import os
import numpy as np
import torch
from defines import PROCESSED_DATA_DIR

# קובץ הנתונים המעובד
processed_path = os.path.join(PROCESSED_DATA_DIR, "Patient_01_data.npy")
print(f"[INFO] Loading processed data from: {processed_path}")

# טעינת הנתונים
data = np.load(processed_path, allow_pickle=True)
print(f"[INFO] Loaded {len(data)} labeled segments")

# הדפסת הדוגמה הראשונה לצורך ניתוח מבנה
print(f"[DEBUG] First sample: {data[0]}")

# פרמטרים קבועים
fixed_length = 300  # מספר דגימות לכל חלון
# המשמעות: אם הדגימה הוקלטה ב-48000Hz, אז זה 300/48000 ≈ 6.25ms של מידע ב"זמן מוחי" – חלון קטן שמתאים לזיהוי נקודתי

X = []
y = []
skipped_short = 0

for sample in data:
    signals = sample["signals"]  # שימוש בשם השדה הנכון (matrix: channels x time)
    label = sample["label"]

    signal = signals[0]  # ערוץ 0 בלבד – ערוץ יחיד לבינתיים

    if len(signal) < fixed_length:
        skipped_short += 1
        continue

    # חלוקה למקטעים באורך קבוע (לא חופפים)
    for j in range(0, len(signal) - fixed_length + 1, fixed_length):
        chunk = signal[j:j + fixed_length]
        X.append(chunk)
        y.append(1 if label != "none" else 0)

X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

# שמירה
output_dir = "Detection"
os.makedirs(output_dir, exist_ok=True)
torch.save(X_tensor, os.path.join(output_dir, "X_tensor.pt"))
torch.save(y_tensor, os.path.join(output_dir, "y_tensor.pt"))

# הדפסות עזר
print(f"[INFO] Total samples (label segments): {len(data)}")
print(f"[INFO] Skipped segments (too short): {skipped_short}")
print(f"[INFO] Final dataset shape: X={X_tensor.shape}, y={y_tensor.shape}")
print("✅ Detection data saved to Detection/X_tensor.pt and Detection/y_tensor.pt")
