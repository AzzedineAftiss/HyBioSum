import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NoamScheduler:
    def __init__(self, optimizer, warmup, model_size):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup = warmup
        self.model_size = model_size

    def step(self):
        self.epoch += 1
        new_lr = self.model_size**(-0.5) * min(self.epoch**(-0.5), self.epoch * self.warmup**(-1.5))
        for param in self.optimizer.param_groups:
            param["lr"] = new_lr

class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff=2048, nheads=8, num_encoders=2, dropout=0.1, max_len=512):
        super().__init__()
        self.positional_enc = PositionalEncoding(d_model, dropout, max_len)
        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=d_ff),
            num_layers=num_encoders
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.positional_enc(x)
        x = self.encoders(x)
        x = self.layer_norm(x)
        logit = self.linear(x)
        sentences_scores = self.sigmoid(logit)
        return sentences_scores.squeeze(-1), logit.squeeze(-1)

def padToSize(to_pad_list, pad_size, filler):
    return to_pad_list + [filler]*(pad_size-len(to_pad_list))