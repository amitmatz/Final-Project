# validate.py
import torch
from Classification.utils import load_model
from Classification.dataset import get_test_loader
from Classification.model.lstm_model import LSTMSpeechClassifier


def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSpeechClassifier(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    ).to(device)

    load_model(model, config['save_path'])

    test_loader = get_test_loader(config)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Test Accuracy: {correct / total:.4f}")
