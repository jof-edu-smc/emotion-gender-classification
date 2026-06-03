import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchaudio
from torchaudio.transforms import Resample, Spectrogram, TimeStretch, TimeMasking, FrequencyMasking, MelScale
import numpy as np

# ---------------------------------------------------------------------------
# 2. Temporal Convolutional Network (TCN)
# ---------------------------------------------------------------------------

# Conv1d: [B, C, L]
# Conv2d: [B, C, H, W]
# Linear: usually [B, F]
# RNN/LSTM: [L, B, F] or [B, L, F] depending on batch_first
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout1d(p=dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout1d(p=dropout_rate)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_elu = nn.ELU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.elu2(out)
        out = out + res
        out = self.final_elu(out)
        return out

class TCNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3, dilation=1, dropout_rate=0.2):
        super().__init__()
        self.tcn_block = TCNBlock(in_channels, 64, kernel_size, dilation, dropout_rate)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        tcn_out = self.tcn_block(x)
        # Global average pooling over time dimension
        gap_out = tcn_out.mean(dim=2)
        logits = self.classifier(gap_out)
        return logits