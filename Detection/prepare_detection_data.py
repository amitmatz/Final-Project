# Detection/prepare_detection_data.py

import os
import numpy as np
import torch
from defines import PROCESSED_DATA_DIR

# ============================================================
# Script to load preprocessed LFP segments and prepare them
# for speech detection training using fixed-length windows
# ============================================================

# קובץ הנתונים המעובד (תוצאה של preprocessing)
processed_path = os.path.join(PROCESSED_DATA_DIR, "Patient_01_data.npy")
print(f"[INFO] Loading processed data from: {processed_path}")

# טעינת הנתונים המעובדים (רשימת מילונים: {'signals': ..., 'label': ...})
data = np.load(processed_path, allow_pickle=True)
print(f"[INFO] Loaded {len(data)} labeled segments")
print("[DEBUG] First sample:", data[0])

# פרמטרים קבועים
fixed_length = 300  # מספר דגימות (לדוגמה, 300 דגימות = 6.25 מ"ש ב-48kHz)

X = []
y = []
skipped_short = 0

for sample in data:
    signal_matrix = sample["signals"]  # shape: (num_channels, num_samples)
    label = sample["label"]

    # נבחר כרגע רק ערוץ אחד (הראשון), נוכל להרחיב בעתיד
    signal = signal_matrix[0]  # shape: (num_samples,)

    if len(signal) < fixed_length:
        skipped_short += 1
        continue

    # חלוקה למקטעים באורך קבוע (בלי חפיפות)
    for i in range(0, len(signal) - fixed_length + 1, fixed_length):
        chunk = signal[i:i + fixed_length]
        X.append(chunk)
        y.append(1 if label != "none" else 0)

# המרה ל-tensor: מומלץ קודם np.array כדי למנוע אזהרה של PyTorch
X_tensor = torch.tensor(np.array(X), dtype=torch.float32)  # <== מהיר ובטוח יותר
y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

# שמירה
output_dir = os.path.join("Detection", "models")
os.makedirs(output_dir, exist_ok=True)
torch.save(X_tensor, os.path.join(output_dir, "X_tensor.pt"))
torch.save(y_tensor, os.path.join(output_dir, "y_tensor.pt"))

# הדפסות עזר
print(f"[INFO] Total samples (label segments): {len(data)}")
print(f"[INFO] Skipped segments (too short): {skipped_short}")
print(f"[INFO] Final dataset shape: X={X_tensor.shape}, y={y_tensor.shape}")
print("✅ Detection data saved to Detection/models/X_tensor.pt and Detection/models/y_tensor.pt")
