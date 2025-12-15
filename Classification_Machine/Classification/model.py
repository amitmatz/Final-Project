# Classification/model.py

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Simple LSTM-based classifier for sequence data.

    Input shape:  (batch_size, seq_len, input_dim)
    Output shape: (batch_size, num_classes)  (logits)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        out, _ = self.lstm(x)  # out: (B, T, H * num_directions)
        last = out[:, -1, :]   # last time step: (B, H * num_directions)
        last = self.dropout(last)
        logits = self.fc(last)  # (B, num_classes)
        return logits
