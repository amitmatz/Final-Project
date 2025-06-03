# Detection/train_detection.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from detection_lstm import SpeechLSTM

# ======================================================
# קובץ זה מאמן את מודל ה-LSTM לזיהוי קטעים עם דיבור
# ======================================================

# נתיבי קבצי הקלט
X_PATH = os.path.join("Detection", "models", "X_tensor.pt")
Y_PATH = os.path.join("Detection", "models", "y_tensor.pt")

# נתיב לשמירת המודל המאומן
MODEL_PATH = os.path.join("Detection", "models", "detection_lstm.pth")

# פרמטרים למודל
INPUT_SIZE = 1     # גודל וקטור הקלט בכל צעד זמן (כעת: ערך אחד בכל שלב)
SEQ_LENGTH = 300   # מספר צעדי זמן (אורך רצף = 300)
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1    # פלט בינארי: דיבור או לא
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

print("[INFO] Loading processed tensors...")

# טעינת הנתונים
X = torch.load(X_PATH)
y = torch.load(Y_PATH)

# וידוא צורת קלט
assert len(X.shape) == 2 and X.shape[1] == SEQ_LENGTH
assert len(y.shape) == 1

# המרת התוויות למימד מתאים
y = y.view(-1, 1)

# יצירת מודל
model = SpeechLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("[INFO] Starting training...")

# אימון
for epoch in range(NUM_EPOCHS):
    model.train()
    permutation = torch.randperm(X.size(0))
    epoch_loss = 0
    correct = 0
    total = 0

    for i in range(0, X.size(0), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_X, batch_y = X[indices], y[indices]

        # הוספת מימד עבור input_size (כעת input_size=1)
        batch_X = batch_X.unsqueeze(-1)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f}")

# שמירת המודל
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
