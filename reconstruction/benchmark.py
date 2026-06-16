"""
reconstruction/benchmark.py
=============================
Side-by-side benchmarking of classical (modal/SVD) vs neural (CNN/UNet)
wavefront reconstruction, including noise- and dropout-robustness
sweeps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reconstruction.classical import ModalReconstructor
from reconstruction.train import load_trained_model, predict_batch
from reconstruction.zernike import zernike_basis
from sim.dataset_gen import load_dataset
from sim.noise import apply_sensor_noise
from sim.phase_screen import compute_rms_wavefront_error, compute_strehl_ratio, get_aperture_mask, zernike_reconstruct
from sim.shwfs import SHWFSSensor, _k_shift


def _rms_wfe_from_coeffs(pred_coeffs: np.ndarray, true_coeffs: np.ndarray, N: int) -> tuple[float, float]:
    """Compute RMS WFE (radians) and Strehl ratio from coefficient error."""
    diff = pred_coeffs - true_coeffs
    rms = float(np.sqrt(np.mean(diff ** 2)))
    strehl = compute_strehl_ratio(rms)
    return rms, strehl


def benchmark_reconstruction(data_path: str, cnn_model_path: str, config: dict, n_test_frames: int = 200) -> pd.DataFrame:
    """
    Run both ModalReconstructor (classical) and a trained CNN on the
    same test data and compute per-frame RMS WFE and Strehl ratio.

    Parameters
    ----------
    data_path : str
        Path to HDF5 dataset.
    cnn_model_path : str
        Path to trained CNN checkpoint.
    config : dict
    n_test_frames : int

    Returns
    -------
    df : pd.DataFrame, columns
        ['frame', 'rms_classical', 'rms_cnn', 'strehl_classical', 'strehl_cnn']
    """
    sim_cfg = config["sim"]
    N = sim_cfg["grid_size"]
    n_zernike = sim_cfg["n_zernike"]
    n_sub = sim_cfg["n_subapertures"]

    data = load_dataset(data_path)
    n_frames = min(n_test_frames, data["slopes"].shape[0])

    sensor = SHWFSSensor(
        n_subapertures=n_sub,
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    basis = zernike_basis(n_zernike, N)
    modal_recon = ModalReconstructor(sensor, basis, n_zernike, config["reconstruction"]["svd_condition_number"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = load_trained_model(cnn_model_path, config, device)

    rms_classical = np.zeros(n_frames)
    rms_cnn = np.zeros(n_frames)
    strehl_classical = np.zeros(n_frames)
    strehl_cnn = np.zeros(n_frames)

    slopes = data["slopes"][:n_frames]
    truth = data["zernike_coeffs"][:n_frames]

    cnn_preds = predict_batch(cnn_model, slopes.astype(np.float32), device)

    for k in range(n_frames):
        sx, sy = slopes[k, 0], slopes[k, 1]
        pred_classical = modal_recon.reconstruct(sx, sy)

        rms_classical[k], strehl_classical[k] = _rms_wfe_from_coeffs(pred_classical, truth[k], N)
        rms_cnn[k], strehl_cnn[k] = _rms_wfe_from_coeffs(cnn_preds[k], truth[k], N)

    return pd.DataFrame(
        {
            "frame": np.arange(n_frames),
            "rms_classical": rms_classical,
            "rms_cnn": rms_cnn,
            "strehl_classical": strehl_classical,
            "strehl_cnn": strehl_cnn,
        }
    )


def benchmark_noise_robustness(data_path: str, cnn_model_path: str, config: dict, noise_levels: list[float]) -> pd.DataFrame:
    """
    Sweep readout-noise levels and compute RMS WFE for both classical
    and CNN reconstructors.

    Parameters
    ----------
    data_path : str
    cnn_model_path : str
    config : dict
    noise_levels : list[float]
        Readout noise sigma values (electrons).

    Returns
    -------
    df : pd.DataFrame, columns ['noise_level', 'rms_classical', 'rms_cnn']
    """
    sim_cfg = config["sim"]
    N = sim_cfg["grid_size"]
    n_zernike = sim_cfg["n_zernike"]
    n_sub = sim_cfg["n_subapertures"]
    n_test_frames = 100

    data = load_dataset(data_path)
    n_frames = min(n_test_frames, data["slopes"].shape[0])
    slopes = data["slopes"][:n_frames]
    truth = data["zernike_coeffs"][:n_frames]

    sensor = SHWFSSensor(
        n_subapertures=n_sub,
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    basis = zernike_basis(n_zernike, N)
    modal_recon = ModalReconstructor(sensor, basis, n_zernike, config["reconstruction"]["svd_condition_number"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = load_trained_model(cnn_model_path, config, device)

    flux = config["noise"]["flux_photons_per_frame"]

    results = []
    for noise_level in noise_levels:
        rms_classical_list = []
        rms_cnn_list = []

        sensor_config = {
            "flux_photons_per_frame": flux,
            "readout_noise_e": noise_level,
            "pixel_to_slope": 1.0 / _k_shift(sim_cfg["detector_pixels_per_subaperture"]),
            "dropout_fraction": 0.0,
        }

        noisy_slopes_batch = np.zeros_like(slopes)
        for k in range(n_frames):
            sx, sy = slopes[k, 0], slopes[k, 1]
            nx, ny = apply_sensor_noise(sx, sy, sensor_config, seed=k)
            noisy_slopes_batch[k, 0] = nx
            noisy_slopes_batch[k, 1] = ny

            pred_classical = modal_recon.reconstruct(nx, ny)
            rms_c, _ = _rms_wfe_from_coeffs(pred_classical, truth[k], N)
            rms_classical_list.append(rms_c)

        cnn_preds = predict_batch(cnn_model, noisy_slopes_batch.astype(np.float32), device)
        for k in range(n_frames):
            rms_n, _ = _rms_wfe_from_coeffs(cnn_preds[k], truth[k], N)
            rms_cnn_list.append(rms_n)

        results.append(
            {
                "noise_level": noise_level,
                "rms_classical": float(np.mean(rms_classical_list)),
                "rms_cnn": float(np.mean(rms_cnn_list)),
            }
        )

    return pd.DataFrame(results)


def benchmark_dropout_robustness(data_path: str, cnn_model_path: str, config: dict, dropout_fracs: list[float] | None = None) -> pd.DataFrame:
    """
    Sweep subaperture dropout fractions and compute RMS WFE degradation
    for both classical and CNN reconstructors.

    Parameters
    ----------
    data_path : str
    cnn_model_path : str
    config : dict
    dropout_fracs : list[float], optional
        Defaults to [0, 0.1, 0.2, 0.3, 0.5].

    Returns
    -------
    df : pd.DataFrame, columns ['dropout_frac', 'rms_classical', 'rms_cnn']
    """
    if dropout_fracs is None:
        dropout_fracs = [0.0, 0.1, 0.2, 0.3, 0.5]

    sim_cfg = config["sim"]
    N = sim_cfg["grid_size"]
    n_zernike = sim_cfg["n_zernike"]
    n_sub = sim_cfg["n_subapertures"]
    n_test_frames = 100

    data = load_dataset(data_path)
    n_frames = min(n_test_frames, data["slopes"].shape[0])
    slopes = data["slopes"][:n_frames]
    truth = data["zernike_coeffs"][:n_frames]

    sensor = SHWFSSensor(
        n_subapertures=n_sub,
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    basis = zernike_basis(n_zernike, N)
    modal_recon = ModalReconstructor(sensor, basis, n_zernike, config["reconstruction"]["svd_condition_number"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = load_trained_model(cnn_model_path, config, device)

    flux = config["noise"]["flux_photons_per_frame"]
    readout = config["noise"]["readout_noise_e"]

    results = []
    for frac in dropout_fracs:
        sensor_config = {
            "flux_photons_per_frame": flux,
            "readout_noise_e": readout,
            "pixel_to_slope": 1.0 / _k_shift(sim_cfg["detector_pixels_per_subaperture"]),
            "dropout_fraction": frac,
        }

        rms_classical_list = []
        noisy_slopes_batch = np.zeros_like(slopes)

        for k in range(n_frames):
            sx, sy = slopes[k, 0], slopes[k, 1]
            nx, ny = apply_sensor_noise(sx, sy, sensor_config, seed=1000 + k)
            noisy_slopes_batch[k, 0] = nx
            noisy_slopes_batch[k, 1] = ny

            pred_classical = modal_recon.reconstruct(nx, ny)
            rms_c, _ = _rms_wfe_from_coeffs(pred_classical, truth[k], N)
            rms_classical_list.append(rms_c)

        cnn_preds = predict_batch(cnn_model, noisy_slopes_batch.astype(np.float32), device)
        rms_cnn_list = [
            _rms_wfe_from_coeffs(cnn_preds[k], truth[k], N)[0] for k in range(n_frames)
        ]

        results.append(
            {
                "dropout_frac": frac,
                "rms_classical": float(np.mean(rms_classical_list)),
                "rms_cnn": float(np.mean(rms_cnn_list)),
            }
        )

    return pd.DataFrame(results)


def print_benchmark_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table of benchmark results."""
    summary = df.describe().loc[["mean", "std", "min", "max"]]
    print("=" * 60)
    print("Benchmark summary")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)


def save_benchmark_results(df: pd.DataFrame, path: str) -> Path:
    """Save benchmark results DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
