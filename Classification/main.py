import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import EEGEncoderDecoder
from utils import load_data, organize_data, train_model, test_model, plot_results, plot_confusion_matrix

# === הגדרות כלליות ===
PATIENT_ID = 0
EPOCHS = 30
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
NUM_FC_LAYERS = 2
DROPOUT_PROB = 0.2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
MODEL_NAME = f'model_patient_{PATIENT_ID}.pth'

# === טענת נתונים ===
mat_path = f'data/matrices_{PATIENT_ID}.pt'
lab_path = f'data/labels_{PATIENT_ID}.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

matrices, labels = load_data(mat_path, lab_path)
train_dl, eval_dl, test_dl = organize_data(matrices, labels, device, BATCH_SIZE)

# === הגדרת המודל ===
model = EEGEncoderDecoder(input_size=matrices.shape[1],  # num_channels
                          hidden_size=HIDDEN_SIZE,
                          num_lstm_layers=NUM_LSTM_LAYERS,
                          num_fc_layers=NUM_FC_LAYERS,
                          num_classes=3,
                          dropout_prob=DROPOUT_PROB).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# === אימון ===
train_losses, eval_losses, train_accs, eval_accs = train_model(
    model, train_dl, eval_dl, criterion, optimizer, EPOCHS, MODEL_NAME
)

# === בדיקה ===
test_acc, test_f1, precision, recall, y_pred, y_true = test_model(model, test_dl, MODEL_NAME)

# === תצוגה ===
print(f"\n--- תוצאות על פציינט #{PATIENT_ID} ---")
print(f"Accuracy: {test_acc:.3f}, F1: {test_f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

plot_results(PATIENT_ID, train_losses, eval_losses, train_accs, eval_accs, EPOCHS)
plot_confusion_matrix(y_true, y_pred, PATIENT_ID)
