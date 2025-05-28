import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs: [batch, time, hidden]
        weights = self.attention(lstm_outputs)                      # [batch, time, 1]
        weights = torch.softmax(weights, dim=1)                     # softmax over time
        weighted_output = torch.bmm(weights.permute(0, 2, 1), lstm_outputs)  # [batch, 1, hidden]
        return weighted_output.squeeze(1)                           # [batch, hidden]

class EEGEncoder(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, dropout_prob=0.0):
        super(EEGEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.attention = Attention(hidden_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # input: [batch, channels, time] -> [batch, time, channels]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, time, hidden]
        attention_out = self.attention(lstm_out)  # [batch, hidden]
        attention_out = self.batch_norm(attention_out)
        return attention_out

class EEGDecoder(nn.Module):
    def __init__(self, hidden_size=128, num_classes=3, num_layers=2, dropout_prob=0.0):
        super(EEGDecoder, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(hidden_size, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)  # [batch, num_classes]

class EEGEncoderDecoder(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_lstm_layers=2, num_fc_layers=2, num_classes=3,
                 dropout_prob=0.0):
        super(EEGEncoderDecoder, self).__init__()
        self.encoder = EEGEncoder(input_size, hidden_size, num_lstm_layers, dropout_prob)
        self.decoder = EEGDecoder(hidden_size, num_classes, num_fc_layers, dropout_prob)

    def forward(self, x):
        x = self.encoder(x)  # [batch, hidden]
        x = self.decoder(x)  # [batch, num_classes]
        return x
