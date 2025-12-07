# classification/model.py
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Tiny (optionally bidirectional) LSTM-based classifier for sequence data.

    Expected input:
        x: Tensor of shape (batch_size, seq_len, input_dim)
           where typically:
             - seq_len   = T_bins  (from preprocessing)
             - input_dim = F       (number of features per time step)

    Output:
        logits: Tensor of shape (batch_size, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_classes: int = 3,
        bidirectional: bool = True,
        lstm_dropout: float = 0.0,
        fc_dropout: float = 0.4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM backbone (uni- or bi-directional).
        # For num_layers == 1 we disable internal LSTM dropout.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * num_directions

        # Tiny FC head: LayerNorm -> Linear -> ReLU -> Dropout -> Linear
        fc_hidden_dim = 64  # small head

        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # out: (batch_size, seq_len, hidden_dim * num_directions)
        out, _ = self.lstm(x)

        # Take the last time step representation
        last = out[:, -1, :]  # (batch_size, hidden_dim * num_directions)

        logits = self.fc(last)  # (batch_size, num_classes)
        return logits


def build_tiny_bilstm_classifier(
    input_dim: int,
    num_classes: int,
) -> LSTMClassifier:
    """
    Convenience builder for the Tiny BiLSTM with recommended hyperparameters.

    Args:
        input_dim: feature dimension per time step (F).
        num_classes: number of output classes.

    Returns:
        An instance of LSTMClassifier configured as Tiny BiLSTM.
    """
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=32,       # tiny hidden size
        num_layers=1,        # single layer
        num_classes=num_classes,
        bidirectional=True,  # BiLSTM
        lstm_dropout=0.0,    # no internal dropout for 1 layer
        fc_dropout=0.4,      # strong regularization in the head
    )
    return model
