"""
sim/noise.py
============
Sensor noise models: photon (shot) noise, readout noise, and
centroiding algorithms (CoG / weighted CoG) plus analytical
centroiding-error formulas and full slope noise injection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def add_photon_noise(spot_image: np.ndarray, flux: float) -> np.ndarray:
    """
    Scale a normalized spot image to a given photon flux and apply
    Poisson (shot) noise.

    Parameters
    ----------
    spot_image : np.ndarray
        Normalized intensity image (sums to ~1).
    flux : float
        Total number of photons in the frame.

    Returns
    -------
    noisy_image : np.ndarray
        Image in photon-count units with Poisson noise applied.
    """
    scaled = spot_image * flux
    scaled = np.clip(scaled, 0, None)
    noisy = np.random.poisson(scaled).astype(float)
    return noisy


def add_readout_noise(spot_image: np.ndarray, sigma_e: float) -> np.ndarray:
    """
    Add Gaussian readout noise with standard deviation ``sigma_e``
    (electrons) to a spot image.

    Parameters
    ----------
    spot_image : np.ndarray
    sigma_e : float
        Readout noise standard deviation (electrons).

    Returns
    -------
    noisy_image : np.ndarray
    """
    noise = np.random.normal(0.0, sigma_e, size=spot_image.shape)
    return spot_image + noise


def centroid_cog(spot_image: np.ndarray) -> tuple[float, float]:
    """
    Centre-of-Gravity (CoG) centroid of a spot image.

    Background is estimated from the mean of the four corner pixels and
    subtracted; pixels below 3*background are thresholded to zero
    before computing the intensity-weighted mean coordinates.

    Parameters
    ----------
    spot_image : np.ndarray, shape (n, n)

    Returns
    -------
    cx, cy : float
        Centroid coordinates (pixels), in array (col, row) = (x, y)
        convention.
    """
    n_rows, n_cols = spot_image.shape

    corners = np.array(
        [
            spot_image[0, 0],
            spot_image[0, -1],
            spot_image[-1, 0],
            spot_image[-1, -1],
        ]
    )
    background = corners.mean()

    img = spot_image - background
    threshold = 3.0 * background
    img = np.where(spot_image > threshold, img, 0.0)
    img = np.clip(img, 0, None)

    total = img.sum()
    if total <= 0:
        return (n_cols - 1) / 2.0, (n_rows - 1) / 2.0

    y_idx, x_idx = np.indices(spot_image.shape)
    cx = (img * x_idx).sum() / total
    cy = (img * y_idx).sum() / total
    return float(cx), float(cy)


def centroid_wcog(spot_image: np.ndarray, threshold_frac: float = 0.3) -> tuple[float, float]:
    """
    Weighted Centre-of-Gravity (WCoG) centroid with a relative
    thresholding scheme to reduce noise bias.

    Pixels below ``threshold_frac * max(spot_image)`` are zeroed before
    computing the intensity-weighted mean coordinates.

    Parameters
    ----------
    spot_image : np.ndarray, shape (n, n)
    threshold_frac : float
        Fraction of the peak intensity below which pixels are zeroed.

    Returns
    -------
    cx, cy : float
        Centroid coordinates (pixels).
    """
    n_rows, n_cols = spot_image.shape
    peak = spot_image.max()
    if peak <= 0:
        return (n_cols - 1) / 2.0, (n_rows - 1) / 2.0

    threshold = threshold_frac * peak
    img = np.where(spot_image >= threshold, spot_image, 0.0)

    total = img.sum()
    if total <= 0:
        return (n_cols - 1) / 2.0, (n_rows - 1) / 2.0

    y_idx, x_idx = np.indices(spot_image.shape)
    cx = (img * x_idx).sum() / total
    cy = (img * y_idx).sum() / total
    return float(cx), float(cy)


def compute_centroiding_error(
    n_photons: float, readout_noise: float, fwhm_pixels: float
) -> float:
    """
    Analytical centroid measurement error (pixels), following the
    standard SH-WFS centroiding noise formula:

    sigma_c = (pi/2) * sigma_det * sqrt(sigma_PSF^2 + sigma_readout^2 / N_photons)

    where sigma_det is the detector pixel scale contribution (taken as
    1 pixel) and sigma_PSF is derived from the spot FWHM.

    Parameters
    ----------
    n_photons : float
        Number of detected photons.
    readout_noise : float
        Readout noise (electrons RMS).
    fwhm_pixels : float
        Spot full-width-half-maximum (pixels).

    Returns
    -------
    sigma_c : float
        Centroid error standard deviation (pixels).
    """
    sigma_psf = fwhm_pixels / 2.3548  # FWHM -> sigma for a Gaussian
    sigma_det = 1.0
    if n_photons <= 0:
        n_photons = 1e-6
    sigma_c = (np.pi / 2.0) * sigma_det * np.sqrt(
        sigma_psf ** 2 + (readout_noise ** 2) / n_photons
    )
    return float(sigma_c)


def apply_sensor_noise(
    slopes_x: np.ndarray,
    slopes_y: np.ndarray,
    sensor_config: dict,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a full sensor noise chain to ideal slope measurements.

    Adds Gaussian noise to each slope proportional to the analytical
    centroiding error (converted from pixels to slope units via the
    sensor focal length / pitch), and supports random subaperture
    dropout (vignetting simulation).

    Parameters
    ----------
    slopes_x, slopes_y : np.ndarray, shape (n_sub, n_sub)
        Ideal slope measurements (radians/pixel-equivalent units).
    sensor_config : dict
        Must contain keys:
          - 'flux_photons_per_frame'
          - 'readout_noise_e'
          - 'fwhm_pixels' (optional, default 2.0)
          - 'pixel_to_slope' : scale factor converting a 1-pixel
             centroid error to slope units (rad). If absent, defaults
             to 1.0.
          - 'dropout_fraction' (optional, default 0.0)
    seed : int, optional

    Returns
    -------
    noisy_slopes_x, noisy_slopes_y : np.ndarray, same shape as inputs
    """
    rng = np.random.default_rng(seed)

    flux = sensor_config.get("flux_photons_per_frame", 1000.0)
    readout_noise = sensor_config.get("readout_noise_e", 3.0)
    fwhm_pixels = sensor_config.get("fwhm_pixels", 2.0)
    pixel_to_slope = sensor_config.get("pixel_to_slope", 1.0)
    dropout_fraction = sensor_config.get("dropout_fraction", 0.0)

    sigma_pix = compute_centroiding_error(flux, readout_noise, fwhm_pixels)
    sigma_slope = sigma_pix * pixel_to_slope

    noisy_x = slopes_x + rng.normal(0.0, sigma_slope, size=slopes_x.shape)
    noisy_y = slopes_y + rng.normal(0.0, sigma_slope, size=slopes_y.shape)

    if dropout_fraction > 0:
        n_sub = slopes_x.shape[0] * slopes_x.shape[1]
        n_drop = int(round(dropout_fraction * n_sub))
        if n_drop > 0:
            flat_idx = rng.choice(n_sub, size=n_drop, replace=False)
            mask = np.ones(n_sub, dtype=bool)
            mask[flat_idx] = False
            mask = mask.reshape(slopes_x.shape)
            noisy_x = noisy_x * mask
            noisy_y = noisy_y * mask

    return noisy_x, noisy_y
