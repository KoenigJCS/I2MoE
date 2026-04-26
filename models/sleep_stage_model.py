import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard transformer sinusoidal positional encoding added to token features.
    Expects input of shape (B, T, D). Adds PE(T, D).
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerEncoderNS(nn.Module):
    """
    Stacked Transformer encoder layers with LayerNorm output.
    Input/Output: (B, T, D)
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 4_096, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.norm(x)


class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise conv followed by pointwise conv for 1D sequences.
    Expects input (B, C, T). Returns same shape.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        return self.dropout(x)


class MultiScaleAttentionFusion(nn.Module):
    """
    Implements C(x) = sum_k Point_k(Depth_k(x)) with kernel sizes 1,3,5, plus self-attention A(x):
    MSA(x) = C(x) + A(x)

    Input expects features shaped as (B, T, D_total) for attention, and (B, C, T) for conv.
    We'll bridge by reshaping between the two views.
    """

    def __init__(self, channels_for_conv: int, d_model_total: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        # Multi-scale depthwise separable conv branch
        self.conv1 = DepthwiseSeparableConv1D(channels_for_conv, 1, dropout)
        self.conv3 = DepthwiseSeparableConv1D(channels_for_conv, 3, dropout)
        self.conv5 = DepthwiseSeparableConv1D(channels_for_conv, 5, dropout)

        # Self-attention branch operating on (B, T, D_total)
        self.attn_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model_total, nhead=nhead, batch_first=True, norm_first=True,
                                        dim_feedforward=max(4 * d_model_total, 512), dropout=dropout),
            num_layers=1,
        )
        self.ln = nn.LayerNorm(d_model_total)

    def forward(self, x_conv: torch.Tensor, x_attn: torch.Tensor) -> torch.Tensor:
        # x_conv: (B, C, T) where C = channels_for_conv
        # x_attn: (B, T, D_total)
        c1 = self.conv1(x_conv)
        c3 = self.conv3(x_conv)
        c5 = self.conv5(x_conv)
        c = c1 + c3 + c5  # (B, C, T)

        a = self.attn_layer(x_attn)
        a = self.ln(a)  # (B, T, D_total)

        # Bring attention back to conv view and sum
        b, t, d = a.shape
        a_conv = a.transpose(1, 2)  # (B, D_total, T)
        # If D_total != C, project to C
        if a_conv.size(1) != c.size(1):
            proj = getattr(self, 'proj_to_c', None)
            if proj is None:
                self.proj_to_c = nn.Conv1d(a_conv.size(1), c.size(1), kernel_size=1)
                self.proj_to_c = self.proj_to_c.to(a_conv.device)
            a_conv = self.proj_to_c(a_conv)
        fused_conv = c + a_conv  # (B, C, T)
        return fused_conv, a  # type: ignore # return both views


class SleepStageClassifier(nn.Module):
    """Simplified, robust 2D-CNN classifier for spectrogram inputs.

    This implementation replaces the heavier transformer-based model with a
    compact 2D convolutional encoder that treats the M modalities as input
    channels. It reduces collapse-to-one-class behaviour by using batch-norm,
    adaptive pooling and a small MLP head.
    """

    def __init__(self, num_modalities: int, num_classes: int, freq_bins: int, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.freq_bins = freq_bins

        # encoder: input (B, M, F, T) -> convs
        self.encoderblock1 = nn.Sequential(
            nn.Conv2d(num_modalities, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.encoderblock2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

        self.encoderblock3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

        self.encoderblock4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

        self.encoderblock5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

        self.encoderblock6 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

        self.encoderblock7 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )


        self.dropout = nn.Dropout(dropout)

        self.sinusoidal_pe = SinusoidalPositionalEncoding(d_model=base_channels * 4, max_len=500)
        

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias) # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, F, T)
        b, m, f, t = x.shape
        assert m == self.num_modalities, f"Expected {self.num_modalities} modalities, got {m}"
        # encoder expects (B, C, H, W) where C=num_modalities, H=F, W=T
        out = self.encoderblock1(x)
        out = self.encoderblock2(out)
        out = self.encoderblock3(out)
        out = self.encoderblock4(out)
        out = self.encoderblock5(out)
        out = self.encoderblock6(out)
        out = self.encoderblock7(out)
        out = self.dropout(out)
        # print(out.shape)
        # out = self.sinusoidal_pe(out)
        # print(out.shape)
        logits = self.head(out)
        return logits
