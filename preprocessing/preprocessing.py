
import os
import json
import numpy as np
import pandas as pd
import h5py
from defines import BASE_DATA_PATH, PATIENTS_CONFIG_PATH, PROCESSED_DATA_DIR

def load_offset(mat_path):
    with h5py.File(mat_path, 'r') as f:
        offset_array = f['new_start_end_times_micsec'][:]
        start_micro = offset_array[0][0]
        offset_sec = start_micro / 1e6
        return offset_sec

def load_labels(label_path):
    df = pd.read_csv(label_path, delimiter="\t", names=["start", "end", "label"])
    return df

def export_lfp_csvs(lfp_folder, out_dir):
    for fname in sorted(os.listdir(lfp_folder)):
        if not fname.endswith(".mat"):
            continue
        path = os.path.join(lfp_folder, fname)
        with h5py.File(path, 'r') as f:
            signal = np.array(f["#refs#/g"]).flatten()
            sr = float(f["#refs#/f/rowTimes"]["sampleRate"][()])
            times = np.arange(len(signal)) / sr
            df = pd.DataFrame({"time": times, "signal": signal})
            out_path = os.path.join(out_dir, fname.replace(".mat", ".csv"))
            df.to_csv(out_path, index=False)
            print(f"[EXPORT] {fname}: {len(signal)} samples → {times[-1]:.3f}s → {out_path}")

def export_detection_windows(lfp_dir, df_labels, out_path, fixed_length=300):
    print(f"[INFO] Creating labeled detection windows from CSV files...")
    all_segments = []
    skipped_segments = 0

    for fname in sorted(os.listdir(lfp_dir)):
        if not fname.endswith(".csv"):
            continue

        ch_name = fname.replace(".csv", "")
        path = os.path.join(lfp_dir, fname)
        df = pd.read_csv(path)

        for _, row in df_labels.iterrows():
            label_str = row["label"]
            start, end = row["start_adj"], row["end_adj"]
            is_speech = label_str.lower() != "none"
            seg = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
            signal = seg["signal"].values

            if len(signal) < fixed_length:
                skipped_segments += 1
                continue

            for i in range(0, len(signal) - fixed_length + 1, fixed_length):
                chunk = signal[i:i + fixed_length]
                all_segments.append({
                    "channel": ch_name,
                    "signals": chunk,
                    "label": label_str,
                    "is_speech": is_speech
                })

    np.save(out_path, all_segments)
    print(f"[INFO] Saved {len(all_segments)} segments, skipped {skipped_segments} (too short).")
    print(f"✅ Detection dataset saved to: {out_path}")

def export_labels_to_csv(df_labels):
    out_path = os.path.join(PROCESSED_DATA_DIR, "labels_aligned.csv")
    df_labels.to_csv(out_path, index=False)
    print(f"[EXPORT] Labels saved to: {out_path}")

def process_patient(patient_name, patient_info):
    print(f"===== Processing {patient_name} =====")
    label_path = os.path.join(BASE_DATA_PATH, patient_info["labels_file"])
    offset_path = os.path.join(BASE_DATA_PATH, patient_info["offset_file"])
    lfp_dir = os.path.join(BASE_DATA_PATH, patient_info["lfp_folder"])
    csv_out_dir = os.path.join(PROCESSED_DATA_DIR, "csvs")
    os.makedirs(csv_out_dir, exist_ok=True)

    offset = load_offset(offset_path)
    df_labels = load_labels(label_path)
    df_labels["start_adj"] = df_labels["start"] + offset
    df_labels["end_adj"] = df_labels["end"] + offset

    export_lfp_csvs(lfp_dir, csv_out_dir)
    export_labels_to_csv(df_labels)

    detection_out_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_detection_data.npy")
    export_detection_windows(csv_out_dir, df_labels, detection_out_path)

def process_all_patients():
    with open(PATIENTS_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    for name, info in config["patients"].items():
        process_patient(name, info)

#
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from detection_lstm import SpeechLSTM
#
# # ======================================================
# # קובץ זה מאמן את מודל ה-LSTM לזיהוי קטעים עם דיבור
# # ======================================================
#
# # נתיבי קבצי הקלט
# X_PATH = os.path.join("Detection", "models", "X_tensor.pt")
# Y_PATH = os.path.join("Detection", "models", "y_tensor.pt")
#
# # נתיב לשמירת המודל המאומן
# MODEL_PATH = os.path.join("Detection", "models", "detection_lstm.pth")
#
# # פרמטרים למודל
# INPUT_SIZE = 1
# SEQ_LENGTH = 300
# HIDDEN_SIZE = 64
# OUTPUT_SIZE = 1
# NUM_EPOCHS = 10
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
#
# print("[INFO] Loading processed tensors...")
#
# X = torch.load(X_PATH)
# y = torch.load(Y_PATH)
#
# # וידוא מימדים
# assert X.dim() == 3 and X.shape[1] == SEQ_LENGTH and X.shape[2] == INPUT_SIZE
# assert y.dim() == 1 or y.dim() == 2
#
# # המרת התוויות למימד מתאים
# y = y.view(-1, 1)
#
# # יצירת מודל
# model = SpeechLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# print("[INFO] Starting training...")
#
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     permutation = torch.randperm(X.size(0))
#     epoch_loss = 0
#     correct = 0
#     total = 0
#
#     for i in range(0, X.size(0), BATCH_SIZE):
#         indices = permutation[i:i + BATCH_SIZE]
#         batch_X, batch_y = X[indices], y[indices]
#
#         # אין צורך ב־unsqueeze – הנתונים כבר 3D
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#         predictions = (torch.sigmoid(outputs) > 0.5).float()
#         correct += (predictions == batch_y).sum().item()
#         total += batch_y.size(0)
#
#     accuracy = correct / total
#     print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f}")
#
# # שמירת המודל
# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# torch.save(model.state_dict(), MODEL_PATH)
# print(f"✅ Model saved to {MODEL_PATH}")
