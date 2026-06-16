"""
sim/phase_screen.py
====================
Utility functions for phase screen manipulation: aperture masking,
Zernike decomposition/reconstruction, and quality metrics (RMS WFE,
Strehl ratio).
"""

from __future__ import annotations

import numpy as np

from reconstruction.zernike import zernike_basis, fit_zernike


def apply_aperture_mask(phase: np.ndarray, aperture_type: str = "circular") -> np.ndarray:
    """
    Apply a circular or square aperture mask to a phase map, zeroing
    pixels outside the aperture.

    Parameters
    ----------
    phase : np.ndarray, shape (N, N)
    aperture_type : str
        'circular' or 'square'.

    Returns
    -------
    masked_phase : np.ndarray, shape (N, N)
    """
    N = phase.shape[0]
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)

    if aperture_type == "circular":
        mask = (xx ** 2 + yy ** 2) <= 1.0
    elif aperture_type == "square":
        mask = (np.abs(xx) <= 1.0) & (np.abs(yy) <= 1.0)
    else:
        raise ValueError(f"Unknown aperture_type: {aperture_type}")

    return phase * mask


def get_aperture_mask(N: int, aperture_type: str = "circular") -> np.ndarray:
    """Return a boolean N x N aperture mask."""
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)
    if aperture_type == "circular":
        return (xx ** 2 + yy ** 2) <= 1.0
    elif aperture_type == "square":
        return (np.abs(xx) <= 1.0) & (np.abs(yy) <= 1.0)
    raise ValueError(f"Unknown aperture_type: {aperture_type}")


def compute_zernike_coefficients(
    phase: np.ndarray, mask: np.ndarray, n_terms: int, pixel_scale: float
) -> np.ndarray:
    """
    Decompose a phase map into Zernike polynomial coefficients via
    least-squares fit.

    Parameters
    ----------
    phase : np.ndarray, shape (N, N), radians
    mask : np.ndarray of bool, shape (N, N)
        Valid-aperture mask.
    n_terms : int
        Number of Zernike modes (Noll indices 1..n_terms).
    pixel_scale : float
        Unused directly (coordinates are normalized to the unit disk),
        kept for API compatibility.

    Returns
    -------
    coeffs : np.ndarray, shape (n_terms,), radians
    """
    N = phase.shape[0]
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    coeffs = fit_zernike(
        phase.flatten(), rho.flatten(), theta.flatten(), n_terms, mask.flatten()
    )
    return coeffs


def zernike_reconstruct(coeffs: np.ndarray, shape: tuple[int, int], pixel_scale: float) -> np.ndarray:
    """
    Reconstruct a phase map from Zernike coefficients.

    phase = sum_j coeffs[j] * Z_j

    Parameters
    ----------
    coeffs : np.ndarray, shape (n_terms,)
    shape : tuple[int, int]
        Output (N, N) shape; must be square (N, N).
    pixel_scale : float
        Unused, kept for API compatibility.

    Returns
    -------
    phase : np.ndarray, shape (N, N), radians
    """
    N = shape[0]
    n_terms = coeffs.shape[0]
    basis = zernike_basis(n_terms, N, normalize=True)
    phase = np.tensordot(coeffs, basis, axes=(0, 0))
    return phase


def compute_rms_wavefront_error(
    phase_true: np.ndarray, phase_reconstructed: np.ndarray, mask: np.ndarray
) -> float:
    """
    RMS wavefront error (radians) between true and reconstructed phase
    maps, evaluated over the valid aperture.
    """
    residual = (phase_true - phase_reconstructed)[mask]
    return float(np.sqrt(np.mean(residual ** 2)))


def compute_strehl_ratio(rms_wfe_radians: float) -> float:
    """
    Maréchal approximation for Strehl ratio:

    S = exp(-(2*pi*sigma/lambda)^2)  ->  simplified to exp(-sigma^2)
    where sigma is the RMS wavefront error in radians.
    Clipped to 1e-6 floor to avoid 0.0000 for large turbulence.
    """
    return float(np.maximum(np.exp(-(rms_wfe_radians ** 2)), 1e-6))
