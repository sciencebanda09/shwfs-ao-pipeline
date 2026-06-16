"""
reconstruction/cnn_model.py
============================
Neural-network wavefront reconstructors: UNet-style encoder/decoder,
a simpler CNN baseline, weighted/physics-informed loss functions, and
an MC-Dropout variant for uncertainty quantification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU, with optional residual connection."""

    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        if self.residual:
            out = out + x
        return out


class EncoderBlock(nn.Module):
    """ConvBlock followed by MaxPool2d. Returns (feature_map, pooled)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        pooled = self.pool(feat)
        return feat, pooled


class DecoderBlock(nn.Module):
    """ConvTranspose2d upsample, concatenate skip connection, then ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetReconstructor(nn.Module):
    """
    UNet-style encoder-decoder wavefront reconstructor.

    Encoder: 4 levels with doubling filters (32, 64, 128, 256).
    Bottleneck: 512 filters.
    Decoder mirrors the encoder with skip connections.
    Head: global average pool -> MLP -> n_zernike outputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 2, for slopes_x/slopes_y).
    n_zernike : int
        Number of output Zernike coefficients.
    base_filters : int
        Number of filters at the first encoder level.
    """

    def __init__(self, in_channels: int = 2, n_zernike: int = 36, base_filters: int = 32):
        super().__init__()
        f = base_filters

        self.enc1 = EncoderBlock(in_channels, f)
        self.enc2 = EncoderBlock(f, f * 2)
        self.enc3 = EncoderBlock(f * 2, f * 4)
        self.enc4 = EncoderBlock(f * 4, f * 8)

        self.bottleneck = ConvBlock(f * 8, f * 16)

        self.dec4 = DecoderBlock(f * 16, f * 8, f * 8)
        self.dec3 = DecoderBlock(f * 8, f * 4, f * 4)
        self.dec2 = DecoderBlock(f * 4, f * 2, f * 2)
        self.dec1 = DecoderBlock(f * 2, f, f)

        self.dropout = nn.Dropout(0.2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f, f * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(f * 2, n_zernike),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, n_sub, n_sub)

        Returns
        -------
        out : torch.Tensor, shape (B, n_zernike)
        """
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)
        b = self.dropout(b)

        d4 = self.dec4(b, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        return self.head(d1)


class CNNReconstructor(nn.Module):
    """
    Simple baseline CNN reconstructor: 4x Conv2d(ReLU) -> Flatten ->
    FC(512) -> FC(n_zernike).

    Parameters
    ----------
    in_channels : int
    n_sub : int
        Subaperture grid size (n_sub x n_sub input).
    n_zernike : int
    """

    def __init__(self, in_channels: int = 2, n_sub: int = 10, n_zernike: int = 36):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        flat_size = 64 * n_sub * n_sub
        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, n_zernike)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Weighted MSE loss between predicted and target Zernike
    coefficients.

    Parameters
    ----------
    pred, target : torch.Tensor, shape (B, n_zernike)
    weights : torch.Tensor, shape (n_zernike,), optional
        Per-mode weights (e.g. higher weight on tip/tilt/focus, which
        dominate wavefront error).

    Returns
    -------
    loss : torch.Tensor, scalar
    """
    sq_err = (pred - target) ** 2
    if weights is not None:
        sq_err = sq_err * weights.unsqueeze(0)
    return sq_err.mean()


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss combining Zernike-coefficient MSE with a
    sensor forward-model consistency term:

    L_total = L_MSE(pred, target) + lambda * ||D @ pred - measured_slopes||^2

    where D is the modal interaction matrix (slopes = D @ zernike).

    Parameters
    ----------
    interaction_matrix : array-like, shape (2*n_valid_sub, n_zernike)
        Modal interaction matrix D.
    """

    def __init__(self, interaction_matrix, lam: float = 0.1):
        super().__init__()
        D = torch.as_tensor(interaction_matrix, dtype=torch.float32)
        self.register_buffer("D", D)
        self.lam = lam

    def forward(self, pred_zernike: torch.Tensor, target_zernike: torch.Tensor, measured_slopes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_zernike, target_zernike : torch.Tensor, shape (B, n_zernike)
        measured_slopes : torch.Tensor, shape (B, 2*n_valid_sub)

        Returns
        -------
        loss : torch.Tensor, scalar
        """
        mse_loss = F.mse_loss(pred_zernike, target_zernike)

        D = self.D.to(pred_zernike.device)
        predicted_slopes = pred_zernike @ D.T
        physics_loss = F.mse_loss(predicted_slopes, measured_slopes)

        return mse_loss + self.lam * physics_loss


class UNetReconstructorMCDropout(UNetReconstructor):
    """
    UNet reconstructor variant supporting Monte-Carlo Dropout for
    uncertainty quantification.

    Parameters
    ----------
    in_channels : int
    n_zernike : int
    base_filters : int
    """

    def __init__(self, in_channels: int = 2, n_zernike: int = 36, base_filters: int = 32):
        super().__init__(in_channels, n_zernike, base_filters)

    def _enable_dropout_only(self) -> None:
        """Set the model to eval mode but force Dropout layers to train mode."""
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run ``n_samples`` stochastic forward passes (with dropout
        active) and return the per-mode mean and standard deviation of
        the predicted Zernike coefficients.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, n_sub, n_sub)
        n_samples : int

        Returns
        -------
        mean_coeffs, std_coeffs : torch.Tensor, shape (B, n_zernike)
        """
        self._enable_dropout_only()

        preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)  # (S, B, n_zernike)
        mean_coeffs = preds.mean(dim=0)
        std_coeffs = preds.std(dim=0)

        self.eval()
        return mean_coeffs, std_coeffs

    @staticmethod
    def uncertainty_gate(std_coeffs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Boolean gate mask: True where the predicted standard deviation
        is below ``threshold`` (i.e. the correction is confident).

        Parameters
        ----------
        std_coeffs : torch.Tensor, shape (..., n_zernike)
        threshold : float

        Returns
        -------
        gate : torch.Tensor of bool, same shape as std_coeffs
        """
        return std_coeffs < threshold
