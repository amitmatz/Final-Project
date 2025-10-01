import numpy as np
import os
from sklearn.model_selection import train_test_split

INPUT_PATH = "processed_data/Patient_03_detection_data_labeled.npy"
OUTPUT_DIR = "Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(INPUT_PATH, allow_pickle=True)
speech_data = [d for d in data if d["is_speech"]]

X = np.array([d["signals"] for d in speech_data], dtype=np.float32)
X = X[:, :, np.newaxis]

unique_labels = sorted(set(d["label"] for d in speech_data))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
y = np.array([label_to_idx[d["label"]] for d in speech_data], dtype=np.int64)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

np.save(os.path.join(OUTPUT_DIR, "Patient_01_train.npy"), {"X": X_train, "y": y_train})
np.save(os.path.join(OUTPUT_DIR, "Patient_01_val.npy"), {"X": X_val, "y": y_val})
np.save(os.path.join(OUTPUT_DIR, "Patient_01_test.npy"), {"X": X_test, "y": y_test})

print("✅ Classification data saved to Data/")
print(f"🔢 Labels map: {label_to_idx}")
