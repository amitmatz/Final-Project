# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LFPSpeechDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(config):
    train_dataset = LFPSpeechDataset(config['train_data_path'])
    val_dataset = LFPSpeechDataset(config['val_data_path'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader


def get_test_loader(config):
    test_dataset = LFPSpeechDataset(config['test_data_path'])
    return DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
