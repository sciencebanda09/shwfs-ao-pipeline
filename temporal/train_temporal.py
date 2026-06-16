"""
temporal/train_temporal.py
============================
Training loop for the LSTM / Transformer temporal prediction model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from temporal.lstm_model import ZernikeTimeSeries, TemporalTransformer
from sim.dataset_gen import load_dataset


def prepare_sequences(zernike_data: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (input, target) sequence pairs from a Zernike
    coefficient time series.

    Parameters
    ----------
    zernike_data : np.ndarray, shape (n_frames, n_zernike)
    seq_len : int
        Input sequence length.
    horizon : int
        Prediction horizon (frames ahead of the last input frame).

    Returns
    -------
    X : np.ndarray, shape (n_samples, seq_len, n_zernike)
    y : np.ndarray, shape (n_samples, n_zernike)
        Target at t = start + seq_len + horizon - 1.
    """
    n_frames, n_zernike = zernike_data.shape
    n_samples = n_frames - seq_len - horizon + 1
    if n_samples <= 0:
        raise ValueError("Sequence too short for given seq_len and horizon")

    X = np.zeros((n_samples, seq_len, n_zernike), dtype=np.float32)
    y = np.zeros((n_samples, n_zernike), dtype=np.float32)

    for i in range(n_samples):
        X[i] = zernike_data[i: i + seq_len]
        y[i] = zernike_data[i + seq_len + horizon - 1]

    return X, y


def train_temporal_model(config: dict, data_path: str, save_path: str) -> dict:
    """
    Full training pipeline for the temporal prediction model.

    Loads the dataset, prepares sliding-window sequences from the
    Zernike time series, splits into train/validation sets,
    instantiates an LSTM or Transformer per config, trains with AdamW +
    cosine LR schedule, and saves the best model by validation loss.

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
        Training/validation loss curves.
    """
    temp_cfg = config["temporal"]
    sim_cfg = config["sim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(data_path)
    zernike_data = data["zernike_coeffs"]

    seq_len = temp_cfg["sequence_length"]
    horizon = temp_cfg["predict_horizon"]

    X, y = prepare_sequences(zernike_data, seq_len, horizon)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=temp_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=temp_cfg["batch_size"], shuffle=False)

    n_zernike = sim_cfg["n_zernike"]

    if temp_cfg["model"] == "lstm":
        model = ZernikeTimeSeries(
            input_size=n_zernike,
            hidden_size=temp_cfg["hidden_size"],
            n_layers=temp_cfg["n_layers"],
            output_size=n_zernike,
        ).to(device)
    else:
        model = TemporalTransformer(
            d_model=temp_cfg["hidden_size"],
            nhead=4,
            n_encoder_layers=temp_cfg["n_layers"],
            n_zernike=n_zernike,
            seq_len=seq_len,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=temp_cfg["learning_rate"])
    n_epochs = temp_cfg["n_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(n_epochs), desc="Training temporal model")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        pbar.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": temp_cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )

    history_path = save_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def load_trained_temporal_model(path: str, config: dict, device: "torch.device"):
    """
    Load a trained temporal model checkpoint saved by train_temporal_model.

    Mirrors reconstruction/train.py::load_trained_model.

    Parameters
    ----------
    path : str
        Path to .pt checkpoint file.
    config : dict
        Parsed config.yaml.
    device : torch.device

    Returns
    -------
    model : nn.Module
        The loaded model in eval() mode on ``device``.
    """
    temp_cfg = config["temporal"]
    sim_cfg = config["sim"]
    n_zernike = sim_cfg["n_zernike"]
    seq_len = temp_cfg["sequence_length"]

    if temp_cfg["model"] == "lstm":
        model = ZernikeTimeSeries(
            input_size=n_zernike,
            hidden_size=temp_cfg["hidden_size"],
            n_layers=temp_cfg["n_layers"],
            output_size=n_zernike,
        )
    else:
        model = TemporalTransformer(
            d_model=temp_cfg["hidden_size"],
            nhead=4,
            n_encoder_layers=temp_cfg["n_layers"],
            n_zernike=n_zernike,
            seq_len=seq_len,
        )

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_prediction_accuracy(
    model, test_sequences: tuple[np.ndarray, np.ndarray], dt: float, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-mode prediction RMSE for a trained temporal model and
    compare to a persistence baseline (predict t+1 = t).

    Parameters
    ----------
    model : nn.Module
    test_sequences : tuple[np.ndarray, np.ndarray]
        (X, y) as produced by prepare_sequences.
    dt : float
        Unused directly, kept for API consistency.
    config : dict

    Returns
    -------
    model_rmse : np.ndarray, shape (n_zernike,)
        Per-mode RMSE of model predictions.
    baseline_rmse : np.ndarray, shape (n_zernike,)
        Per-mode RMSE of the persistence baseline (last input frame).
    """
    X, y = test_sequences
    device = next(model.parameters()).device

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()

    model_rmse = np.sqrt(np.mean((preds - y) ** 2, axis=0))

    persistence = X[:, -1, :]  # last frame of each input sequence
    baseline_rmse = np.sqrt(np.mean((persistence - y) ** 2, axis=0))

    return model_rmse, baseline_rmse
