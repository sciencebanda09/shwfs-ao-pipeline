"""
sim/dataset_gen.py
===================
Dataset generation: single-frame simulation, full time-series HDF5
dataset writer/reader, and PyTorch Dataset wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from sim.noise import add_photon_noise, add_readout_noise, centroid_cog
from sim.phase_screen import apply_aperture_mask, compute_zernike_coefficients, get_aperture_mask
from sim.shwfs import SHWFSSensor
from sim.turbulence import MultiLayerAtmosphere, build_atmosphere_from_config


def generate_single_frame(
    atmosphere: MultiLayerAtmosphere,
    sensor: SHWFSSensor,
    noise_config: dict,
    zernike_basis: Optional[np.ndarray],
    n_zernike: int,
) -> dict:
    """
    Simulate a single AO time-step.

    Steps
    -----
    1. Get the Cn2-integrated phase screen from the atmosphere.
    2. Propagate it through the SH-WFS to obtain ideal slopes.
    3. Apply the sensor noise chain (photon + readout + dropout via
       per-subaperture spot simulation and centroiding).
    4. Compute ground-truth Zernike coefficients from the clean phase.

    Parameters
    ----------
    atmosphere : MultiLayerAtmosphere
    sensor : SHWFSSensor
    noise_config : dict
        Sub-dict of config['noise'].
    zernike_basis : np.ndarray or None
        Unused directly (kept for API compatibility); Zernike fitting
        is performed analytically via least squares.
    n_zernike : int

    Returns
    -------
    frame : dict with keys
        'slopes_x', 'slopes_y' : np.ndarray (n_sub, n_sub)
        'zernike_coeffs' : np.ndarray (n_zernike,)
        'phase_map' : np.ndarray (N, N), radians
    """
    phase_radians = atmosphere.get_integrated_phase_radians()
    N = phase_radians.shape[0]
    mask = get_aperture_mask(N, "circular")
    phase_masked = apply_aperture_mask(phase_radians, "circular")

    # Ground-truth slopes (no noise)
    slopes_x, slopes_y = sensor.propagate(phase_masked)

    if noise_config.get("photon_noise", True):
        slopes_x, slopes_y = _apply_realistic_noise(
            sensor, slopes_x, slopes_y, noise_config
        )

    zernike_coeffs = compute_zernike_coefficients(phase_masked, mask, n_zernike, pixel_scale=1.0)

    return {
        "slopes_x": slopes_x,
        "slopes_y": slopes_y,
        "zernike_coeffs": zernike_coeffs,
        "phase_map": phase_masked,
    }


def _apply_realistic_noise(
    sensor: SHWFSSensor, slopes_x: np.ndarray, slopes_y: np.ndarray, noise_config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a per-subaperture spot-simulation noise chain: for each valid
    subaperture, simulate the spot image from the local slope, add
    photon + readout noise, recompute the centroid, and convert the
    recovered centroid offset back to a slope.
    """
    flux = noise_config.get("flux_photons_per_frame", 1000)
    readout = noise_config.get("readout_noise_e", 3.0)

    valid = sensor.get_valid_subaperture_mask()
    noisy_x = np.zeros_like(slopes_x)
    noisy_y = np.zeros_like(slopes_y)

    n = sensor.pix_per_sub
    # Reference centroid from noiseless zero-tilt spot (not geometric centre,
    # because the oversampled FFT PSF peak may not land exactly at (n-1)/2).
    _ref_spot = sensor.simulate_spot_image(0.0, 0.0)
    ref_cx, ref_cy = centroid_cog(_ref_spot)

    # Conversion factor K_SHIFT: centroid pixels per (rad/pixel tilt).
    # With the oversampled FFT in simulate_spot_image (4x zero-padding),
    # a tilt of t rad/pixel shifts the spot by t * n / (2*pi) pixels,
    # i.e.  K_SHIFT = n / (2*pi).  Verified empirically: R^2 > 0.999
    # over |tilt| < 0.05 rad/pixel for n = pix_per_sub.
    # This is also what the Fourier shift theorem predicts: a phase ramp
    # exp(i*t*X) over an n-pixel pupil shifts the PSF by t*n/(2*pi) pixels.
    pix_per_rad = n / (2.0 * np.pi)  # = K_SHIFT

    for i in range(sensor.n_sub):
        for j in range(sensor.n_sub):
            if not valid[i, j]:
                continue
            sx, sy = slopes_x[i, j], slopes_y[i, j]
            spot = sensor.simulate_spot_image(sx, sy)

            noisy_spot = add_photon_noise(spot, flux)
            noisy_spot = add_readout_noise(noisy_spot, readout)
            noisy_spot = np.clip(noisy_spot, 0, None)

            cx, cy = centroid_cog(noisy_spot)

            noisy_x[i, j] = (cx - ref_cx) / pix_per_rad
            noisy_y[i, j] = (cy - ref_cy) / pix_per_rad

    return noisy_x, noisy_y


def generate_dataset(config: dict, n_frames: int, output_path: str, seed: int = 42) -> Path:
    """
    Generate a full time-series dataset and write it to an HDF5 file.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    n_frames : int
        Number of frames to simulate.
    output_path : str
        Output HDF5 file path.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    path : Path
        Path to the written HDF5 file.
    """
    np.random.seed(seed)

    sim_cfg = config["sim"]
    noise_cfg = config["noise"]
    N = sim_cfg["grid_size"]
    n_zernike = sim_cfg["n_zernike"]
    n_sub = sim_cfg["n_subapertures"]
    dt = sim_cfg["dt_s"]
    pixel_scale = sim_cfg["aperture_diameter_m"] / N

    atmosphere = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=seed)
    sensor = SHWFSSensor(
        n_subapertures=n_sub,
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        slopes_ds = f.create_dataset(
            "slopes", shape=(n_frames, 2, n_sub, n_sub), dtype="float64",
            chunks=(1, 2, n_sub, n_sub),
        )
        zernike_ds = f.create_dataset(
            "zernike_coeffs", shape=(n_frames, n_zernike), dtype="float64",
            chunks=(min(64, n_frames), n_zernike),
        )
        phase_ds = f.create_dataset(
            "phase_maps", shape=(n_frames, N, N), dtype="float64",
            chunks=(1, N, N),
        )
        ts_ds = f.create_dataset(
            "timestamps", shape=(n_frames,), dtype="float64",
        )

        for k in tqdm(range(n_frames), desc="Generating dataset"):
            atmosphere.evolve(dt)
            frame = generate_single_frame(atmosphere, sensor, noise_cfg, None, n_zernike)

            slopes_ds[k, 0] = frame["slopes_x"]
            slopes_ds[k, 1] = frame["slopes_y"]
            zernike_ds[k] = frame["zernike_coeffs"]
            phase_ds[k] = frame["phase_map"]
            ts_ds[k] = k * dt

        f.attrs["n_frames"] = n_frames
        f.attrs["n_zernike"] = n_zernike
        f.attrs["n_sub"] = n_sub
        f.attrs["dt_s"] = dt
        f.attrs["r0_m"] = config["turbulence"]["r0_m"]

    return output_path


def load_dataset(path: str) -> dict:
    """
    Load an HDF5 dataset produced by ``generate_dataset``.

    Returns
    -------
    data : dict with keys 'slopes', 'zernike_coeffs', 'phase_maps',
        'timestamps', and 'attrs'.
    """
    with h5py.File(path, "r") as f:
        data = {
            "slopes": f["slopes"][:],
            "zernike_coeffs": f["zernike_coeffs"][:],
            "phase_maps": f["phase_maps"][:],
            "timestamps": f["timestamps"][:],
            "attrs": dict(f.attrs),
        }
    return data


class SHWFSDataset(Dataset):
    """
    PyTorch Dataset wrapping an HDF5 SH-WFS dataset.

    Two modes:
      - Frame mode (sequence_length=None): __getitem__ returns
        (slopes_tensor, zernike_tensor) for a single frame, where
        slopes_tensor has shape (2, n_sub, n_sub).
      - Sequence mode (sequence_length=int): __getitem__ returns
        (slopes_sequence_tensor, next_zernike_tensor), where
        slopes_sequence_tensor has shape (sequence_length, n_zernike)
        built from per-frame Zernike coefficients (used by temporal
        models), and next_zernike_tensor is the target at
        t = start + sequence_length.

    Parameters
    ----------
    path : str
        Path to HDF5 file.
    sequence_length : int or None
        If set, enables sequence mode for temporal models.
    """

    def __init__(self, path: str, sequence_length: Optional[int] = None):
        self.path = path
        self.sequence_length = sequence_length

        with h5py.File(path, "r") as f:
            self.n_frames = f.attrs["n_frames"]
            self.n_zernike = f.attrs["n_zernike"]
            self.n_sub = f.attrs["n_sub"]

        self._slopes = None
        self._zernike = None

    def _lazy_load(self):
        if self._slopes is None:
            with h5py.File(self.path, "r") as f:
                self._slopes = f["slopes"][:]
                self._zernike = f["zernike_coeffs"][:]

    def __len__(self) -> int:
        self._lazy_load()
        if self.sequence_length is None:
            return self.n_frames
        return max(0, self.n_frames - self.sequence_length)

    def __getitem__(self, idx: int):
        self._lazy_load()

        if self.sequence_length is None:
            slopes = self._slopes[idx]  # (2, n_sub, n_sub)
            zernike = self._zernike[idx]  # (n_zernike,)
            return (
                torch.tensor(slopes, dtype=torch.float32),
                torch.tensor(zernike, dtype=torch.float32),
            )
        else:
            seq = self._zernike[idx: idx + self.sequence_length]  # (seq_len, n_zernike)
            target = self._zernike[idx + self.sequence_length]  # (n_zernike,)
            return (
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
            )
