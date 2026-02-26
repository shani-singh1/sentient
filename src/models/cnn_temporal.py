from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualConvBlock(64),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualConvBlock(128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage2(h)
        h = self.stage3(h)
        return self.proj(h)


class CNNLSTMRegressor(nn.Module):
    def __init__(self, in_channels: int, aux_dim: int, emb_dim: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, emb_dim=emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.aux = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, C, H, W]
        b, t, c, h, w = x_seq.shape
        x = x_seq.reshape(b * t, c, h, w)
        e = self.encoder(x).reshape(b, t, -1)
        o, _ = self.lstm(e)
        z = self.aux(x_aux)
        y = self.head(torch.cat([o[:, -1, :], z], dim=1)).squeeze(1)
        return y


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(0.2),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CNNTCNRegressor(nn.Module):
    def __init__(self, in_channels: int, aux_dim: int, emb_dim: int = 128, tcn_channels: int = 128) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, emb_dim=emb_dim)
        self.proj = nn.Linear(emb_dim, tcn_channels)
        self.tcn = nn.Sequential(
            TemporalConvBlock(tcn_channels, dilation=1),
            TemporalConvBlock(tcn_channels, dilation=2),
            TemporalConvBlock(tcn_channels, dilation=4),
        )
        self.aux = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(tcn_channels + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, C, H, W]
        b, t, c, h, w = x_seq.shape
        x = x_seq.reshape(b * t, c, h, w)
        e = self.encoder(x).reshape(b, t, -1)
        z = self.proj(e).transpose(1, 2)  # [B, Ch, T]
        z = self.tcn(z)[:, :, -1]
        a = self.aux(x_aux)
        y = self.head(torch.cat([z, a], dim=1)).squeeze(1)
        return y
