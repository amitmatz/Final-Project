import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import EEGEncoderDecoder
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# === טוען את הנתונים מהפרויקט שלך ===
def load_npy_data(data_path, labels_path):
    X = np.load(data_path)         # shape: (samples, channels, time)
    y = np.load(labels_path)       # shape: (samples,)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# === מחלק את הנתונים ל־train/val/test ===
def split_data(X, y, val_frac=0.1, test_frac=0.1):
    dataset = TensorDataset(X, y)
    total = len(dataset)
    test_size = int(test_frac * total)
    val_size = int(val_frac * total)
    train_size = total - test_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

# === פונקציית אימון בודדת ===
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # הערכת דיוק על validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {acc:.4f}")

# === פונקציית בדיקה ===
def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return acc, f1, precision, recall

# === הרצת הסקריפט ===
if __name__ == "__main__":
    # 🧠 עדכן כאן את הנתיבים לנתונים שלך:
    data_path = "Patient_01_data.npy"
    labels_path = "Patient_01_labels.npy"

    # 🧠 עדכן מספר קטגוריות בהתאם למילים שלך (למשל: "האריה", "אהב", "תות")
    num_classes = 3

    # === הגדרות המודל ===
    input_size = 12  # מספר ערוצים (channels) – ודא שזה נכון!
    hidden_size = 128
    num_lstm_layers = 2
    num_fc_layers = 2
    dropout_prob = 0.2
    batch_size = 16
    epochs = 30
    learning_rate = 0.001
    weight_decay = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === טעינה ===
    X, y = load_npy_data(data_path, labels_path)
    train_ds, val_ds, test_ds = split_data(X, y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # === בניית המודל והאימון ===
    model = EEGEncoderDecoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        num_fc_layers=num_fc_layers,
        num_classes=num_classes,
        dropout_prob=dropout_prob
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

    # === בדיקה על סט בדיקה ===
    test_model(model, test_loader, device)

    # === שמירת המודל ===
    torch.save(model.state_dict(), "eeg_decoder_model.pt")
