# utils.py
import torch
import random
import numpy as np


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# classification_config.json
{
  "train_data_path": "Data/Patient_01_train.npy",
  "val_data_path": "Data/Patient_01_val.npy",
  "test_data_path": "Data/Patient_01_test.npy",
  "input_dim": 64,
  "hidden_dim": 128,
  "output_dim": 3,
  "batch_size": 32,
  "epochs": 50,
  "lr": 0.001,
  "seed": 42,
  "save_path": "classification/saved_models/lstm_model.pth",
  "log_dir": "classification/logs"
}
