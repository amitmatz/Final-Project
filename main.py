# main.py (root level)
import os
import json
from Classification.train import train_model
from Classification.validate import evaluate


# שלב 1: הרצת Preprocessing
print("[STEP 1] Running preprocessing for all patients...")
os.system("python -m preprocessing.preprocessing")

# שלב 2: הרצת Detection
print("[STEP 2] Generating tensors for detection training...")
os.system("python -m Detection.generate_tensors_from_csv")
print("[STEP 3] Training detection model...")
os.system("python -m Detection.train_detection")
print("[STEP 4] Applying detection model on full segments...")
os.system("python apply_detection_model.py")

# שלב 3: הכנת נתוני classification עם איחוד label "Haarye" ל-"Arye"
def normalize_label(label):
    if label.lower() == "haarye":
        return "Arye"
    return label

print("[STEP 5] Preparing classification data from detection results...")
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_PATH = "processed_data/Patient_01_detection_data_labeled.npy"
OUTPUT_DIR = "Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(INPUT_PATH, allow_pickle=True)
speech_data = [d for d in data if d["is_speech"]]

X = np.array([d["signals"] for d in speech_data], dtype=np.float32)
X = X[:, :, np.newaxis]

unique_labels = sorted(set(normalize_label(d["label"]) for d in speech_data))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
y = np.array([label_to_idx[normalize_label(d["label"])] for d in speech_data], dtype=np.int64)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

np.save(os.path.join(OUTPUT_DIR, "Patient_01_train.npy"), {"X": X_train, "y": y_train})
np.save(os.path.join(OUTPUT_DIR, "Patient_01_val.npy"), {"X": X_val, "y": y_val})
np.save(os.path.join(OUTPUT_DIR, "Patient_01_test.npy"), {"X": X_test, "y": y_test})
print("✅ Classification data saved to Data/")
print(f"🔢 Labels map: {label_to_idx}")

# שלב 4: הרצת אימון ו־Evaluation של classification
print("[STEP 6] Training classification model...")
with open("classification/classification_config.json") as f:
    config = json.load(f)
train_model(config)

print("[STEP 7] Final evaluation on test set...")
evaluate(config)
