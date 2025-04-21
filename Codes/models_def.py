# models_def.py

import torch
import torch.nn as nn

class TL_LipLanguageModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, num_classes=10, freeze_first_lstm=False):
        super(TL_LipLanguageModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        if freeze_first_lstm:
            for name, param in self.lstm.named_parameters():
                if 'l0' in name:  
                    param.requires_grad = False

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        return self.classifier(out)
