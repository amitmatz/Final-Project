import torch
import numpy as np
import os
from Detection.detection_lstm import SpeechLSTM

INPUT_SIZE = 1
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
SEQ_LENGTH = 300
MODEL_PATH = "Detection/models/detection_lstm.pth"

INPUT_FILE = "processed_data/Patient_01_detection_data.npy"
OUTPUT_FILE = "processed_data/Patient_01_detection_data_labeled.npy"

model = SpeechLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

data = np.load(INPUT_FILE, allow_pickle=True)
print(f"[INFO] Running detection model on {len(data)} segments...")

for d in data:
    sig = d["signals"][:SEQ_LENGTH]
    sig_tensor = torch.tensor(sig, dtype=torch.float32).view(1, SEQ_LENGTH, 1)
    with torch.no_grad():
        pred = torch.sigmoid(model(sig_tensor)).item()
        d["is_speech"] = pred > 0.5

np.save(OUTPUT_FILE, data)
print(f"✅ Updated detection saved to: {OUTPUT_FILE}")
