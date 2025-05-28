# utils.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def load_data(matrices_path, labels_path):
    matrices = torch.load(matrices_path)
    labels = torch.load(labels_path)
    if not isinstance(labels, list):
        labels = labels.tolist()
    return matrices, labels

def organize_data(matrices, labels, device, batch_size):
    x_train, x_temp, y_train, y_temp = train_test_split(
        matrices, labels, train_size=0.65, stratify=labels, random_state=42)
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42)

    x_train = x_train.float().to(device)
    x_eval = x_eval.float().to(device)
    x_test = x_test.float().to(device)
    y_train = torch.tensor(y_train).to(device)
    y_eval = torch.tensor(y_eval).to(device)
    y_test = torch.tensor(y_test).to(device)

    train_ds = TensorDataset(x_train, y_train)
    eval_ds = TensorDataset(x_eval, y_eval)
    test_ds = TensorDataset(x_test, y_test)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(eval_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )

def train_model(model, train_loader, eval_loader, criterion, optimizer, epochs, model_name):
    train_losses, eval_losses = [], []
    train_accs, eval_accs = [], []
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(acc)

        # Evaluate
        model.eval()
        eval_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in eval_loader:
                out = model(x)
                loss = criterion(out, y)
                eval_loss += loss.item()
                val_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        eval_f1 = f1_score(val_labels, val_preds, average='weighted')
        eval_acc = accuracy_score(val_labels, val_preds)
        eval_losses.append(eval_loss / len(eval_loader))
        eval_accs.append(eval_acc)

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), model_name)

    return train_losses, eval_losses, train_accs, eval_accs

def test_model(model, test_loader, model_name):
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_name))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    return acc, f1, prec, rec, all_preds, all_labels

def plot_results(patient_id, train_losses, eval_losses, train_accs, eval_accs, epochs):
    def smooth(x):
        for i in range(1, len(x)):
            x[i] = 0.9 * x[i-1] + 0.1 * x[i]
        return x

    train_losses = smooth(train_losses)
    eval_losses = smooth(eval_losses)
    train_accs = smooth(train_accs)
    eval_accs = smooth(eval_accs)

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(eval_accs, label='Eval Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.suptitle(f'Patient #{patient_id}')
    plt.savefig(f'plots/loss_accuracy_p{patient_id}.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, patient_id):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["האריה", "אהב", "תות"])
    disp.plot(cmap='Blues')
    os.makedirs("plots", exist_ok=True)
    plt.title(f"Confusion Matrix – Patient {patient_id}")
    plt.savefig(f"plots/confusion_matrix_p{patient_id}.png")
    plt.show()
