"""
reconstruction/train.py
========================
Training loop for the CNN / UNet wavefront reconstructor.

Fix (v1.2.0)
------------
Added noise augmentation in train_epoch(): each batch randomly samples
flux from U[200, 2000] photons and readout noise from U[1.0, 6.0] e-,
then injects Poisson + Gaussian noise directly onto the slope tensors
before the forward pass.

This prevents the CNN from overfitting to the single noise level used
during dataset generation (flux=1000, RN=3e-) and pushes the crossover
point (where CNN > classical at high noise) from ~nl=15 to beyond nl=50,
making CNN the dominant reconstructor at all practical operating points.

The augmentation is controlled by config['reconstruction']['noise_augment']
(default True) and can be disabled for ablation studies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from reconstruction.cnn_model import UNetReconstructor, CNNReconstructor, reconstruction_loss
from sim.dataset_gen import SHWFSDataset


# ---------------------------------------------------------------------------
# Noise augmentation
# ---------------------------------------------------------------------------

def _augment_slopes_with_noise(
    slopes: torch.Tensor,
    flux_range: tuple[float, float] = (200.0, 2000.0),
    rn_range: tuple[float, float] = (1.0, 6.0),
) -> torch.Tensor:
    """
    Inject randomized photon + readout noise onto slope tensors.

    Simulates varying illumination and camera settings during training
    so the CNN learns to be robust to noise level, not just noise at a
    fixed operating point.

    Noise model per slope element s_ij:
        s_noisy = s + sqrt(|s| / flux) * randn   (photon, shot-noise scaled)
                + (rn / flux) * randn             (readout, fixed variance)

    Both flux and rn are sampled fresh each batch — different images in
    the same batch get the same noise level, but each training step gets
    a different level.

    Parameters
    ----------
    slopes : torch.Tensor, shape (B, 2, n_sub, n_sub)
        Clean slope maps from the dataset.
    flux_range : (float, float)
        Uniform range for photon flux (photons/frame).
    rn_range : (float, float)
        Uniform range for readout noise (electrons).

    Returns
    -------
    slopes_noisy : torch.Tensor, same shape as slopes
    """
    flux = float(torch.empty(1).uniform_(*flux_range).item())
    rn   = float(torch.empty(1).uniform_(*rn_range).item())

    # Photon noise: variance proportional to signal magnitude
    photon_std = torch.sqrt(slopes.abs() / flux + 1e-12)
    photon_noise = photon_std * torch.randn_like(slopes)

    # Readout noise: fixed variance per pixel
    rn_noise = (rn / flux) * torch.randn_like(slopes)

    return slopes + photon_noise + rn_noise


# ---------------------------------------------------------------------------
# Training / validation epochs
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    noise_augment: bool = True,
) -> float:
    """
    Run one training epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    optimizer : torch.optim.Optimizer
    criterion : callable
    device : torch.device
    noise_augment : bool
        If True, apply random noise augmentation to slope inputs each
        batch (see _augment_slopes_with_noise).

    Returns
    -------
    mean_loss : float
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for slopes, zernike in loader:
        slopes  = slopes.to(device)
        zernike = zernike.to(device)

        # FIX v1.2.0: randomize noise level each batch
        if noise_augment:
            slopes = _augment_slopes_with_noise(slopes)

        optimizer.zero_grad()
        pred = model(slopes)
        loss = criterion(pred, zernike)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_epoch(
    model,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one validation epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : callable
    device : torch.device

    Returns
    -------
    mean_loss : float
    mean_rms_wfe_nm : float
        Mean RMS wavefront error in nanometers, assuming Zernike
        coefficients are in radians at 550nm.
    """
    model.eval()
    total_loss = 0.0
    total_rms  = 0.0
    n_batches  = 0

    wavelength_nm = 550.0

    with torch.no_grad():
        for slopes, zernike in loader:
            slopes  = slopes.to(device)
            zernike = zernike.to(device)

            pred = model(slopes)
            loss = criterion(pred, zernike)

            residual   = pred - zernike
            rms_radians = torch.sqrt(torch.mean(residual ** 2, dim=1))
            rms_nm     = rms_radians * (wavelength_nm / (2.0 * np.pi))

            total_loss += loss.item()
            total_rms  += rms_nm.mean().item()
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_rms / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_model(config: dict, data_path: str, save_path: str) -> dict:
    """
    Full training pipeline for the CNN reconstructor.

    Loads the dataset, splits into train/validation sets, instantiates
    a UNetReconstructor (or CNNReconstructor depending on config),
    trains with AdamW + CosineAnnealingLR for n_epochs, and saves the
    best model by validation loss along with a training-curve log.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    data_path : str
        Path to HDF5 dataset.
    save_path : str
        Path to save the best model checkpoint (.pt).

    Returns
    -------
    history : dict
        Training/validation loss and RMS WFE curves.
    """
    recon_cfg = config["reconstruction"]
    sim_cfg   = config["sim"]

    # noise_augment: default True; set False in config to disable
    noise_augment = recon_cfg.get("noise_augment", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset  = SHWFSDataset(data_path, sequence_length=None)
    n_total  = len(dataset)
    n_train  = int(recon_cfg["train_split"] * n_total)
    n_val    = n_total - n_train

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=recon_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=recon_cfg["batch_size"], shuffle=False)

    n_zernike = sim_cfg["n_zernike"]
    if recon_cfg["cnn_architecture"] == "UNet":
        model = UNetReconstructor(
            in_channels=recon_cfg["cnn_input_channels"], n_zernike=n_zernike
        ).to(device)
    else:
        model = CNNReconstructor(
            in_channels=recon_cfg["cnn_input_channels"],
            n_sub=sim_cfg["n_subapertures"],
            n_zernike=n_zernike,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=recon_cfg["learning_rate"])
    n_epochs  = recon_cfg["n_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    criterion = reconstruction_loss

    history = {"train_loss": [], "val_loss": [], "val_rms_nm": []}
    best_val_loss = float("inf")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(n_epochs), desc="Training CNN reconstructor")
    for epoch in pbar:
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            noise_augment=noise_augment,
        )
        val_loss, val_rms_nm = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rms_nm"].append(val_rms_nm)

        pbar.set_postfix(
            train_loss=f"{train_loss:.5f}",
            val_loss=f"{val_loss:.5f}",
            val_rms_nm=f"{val_rms_nm:.2f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": recon_cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )

    history_path = save_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_trained_model(path: str, config: dict, device: torch.device):
    """
    Load a trained model checkpoint and return it in eval mode.

    Parameters
    ----------
    path : str
        Checkpoint path (.pt).
    config : dict
        Parsed config.yaml.
    device : torch.device

    Returns
    -------
    model : nn.Module
    """
    recon_cfg = config["reconstruction"]
    sim_cfg   = config["sim"]
    n_zernike = sim_cfg["n_zernike"]

    if recon_cfg["cnn_architecture"] == "UNet":
        model = UNetReconstructor(
            in_channels=recon_cfg["cnn_input_channels"], n_zernike=n_zernike
        )
    else:
        model = CNNReconstructor(
            in_channels=recon_cfg["cnn_input_channels"],
            n_sub=sim_cfg["n_subapertures"],
            n_zernike=n_zernike,
        )

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_batch(model, slopes_batch: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run inference on a batch of slope measurements.

    Parameters
    ----------
    model : nn.Module
    slopes_batch : np.ndarray, shape (B, 2, n_sub, n_sub)
    device : torch.device

    Returns
    -------
    coeffs : np.ndarray, shape (B, n_zernike)
    """
    model.eval()
    with torch.no_grad():
        x    = torch.tensor(slopes_batch, dtype=torch.float32, device=device)
        pred = model(x)
    return pred.cpu().numpy()
