"""
temporal/lstm_model.py
========================
Temporal models for turbulence characterization and wavefront
prediction: stacked LSTM and Transformer encoder, operating on Zernike
coefficient time series.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ZernikeTimeSeries(nn.Module):
    """
    Stacked LSTM for next-frame Zernike coefficient prediction.

    Parameters
    ----------
    input_size : int
        Number of input features (n_zernike).
    hidden_size : int
    n_layers : int
    output_size : int
        Number of output features (n_zernike).
    dropout : float
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, n_zernike)

        Returns
        -------
        out : torch.Tensor, shape (B, output_size)
        """
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer inputs.

    Parameters
    ----------
    d_model : int
    max_len : int
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, d_model)

        Returns
        -------
        out : torch.Tensor, same shape as x
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class TemporalTransformer(nn.Module):
    """
    Transformer-encoder-based temporal model for next-frame Zernike
    coefficient prediction.

    Parameters
    ----------
    d_model : int
    nhead : int
    n_encoder_layers : int
    n_zernike : int
        Input/output feature dimension.
    seq_len : int
        Expected input sequence length (used for positional encoding
        max length).
    dropout : float
    """

    def __init__(self, d_model: int, nhead: int, n_encoder_layers: int, n_zernike: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_zernike, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(seq_len + 1, 16))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.output_head = nn.Linear(d_model, n_zernike)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, n_zernike)

        Returns
        -------
        out : torch.Tensor, shape (B, n_zernike)
        """
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        last_hidden = h[:, -1, :]
        return self.output_head(last_hidden)
