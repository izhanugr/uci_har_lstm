import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, dropout, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer,
                            dropout=dropout, bidirectional=bidirectional, batch_first=True,)
        self.fc = nn.Linear(hidden_dim *(2 if bidirectional else 1), output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:]) #use last timestep
        return out