# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Classification.dataset import get_dataloaders
from Classification.model.lstm_model import LSTMSpeechClassifier
from Classification.utils import save_model, set_seed
import os


def train_model(config):
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(config)

    model = LSTMSpeechClassifier(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    writer = SummaryWriter(log_dir=config['log_dir'])

    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        acc = correct / total
        val_acc = validate(model, val_loader, device)
        writer.add_scalars("accuracy", {"train": acc, "val": val_acc}, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, config['save_path'])

        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss:.4f}, Acc: {acc:.4f}, Val_Acc: {val_acc:.4f}")

    writer.close()


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
