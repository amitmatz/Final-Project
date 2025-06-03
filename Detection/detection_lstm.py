# Detection/detection_lstm.py

import torch
import torch.nn as nn

class SpeechLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpeechLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(2)  # reshape to (batch_size, sequence_length, input_size=1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # use last hidden state
        out = self.fc(out)
        return out
