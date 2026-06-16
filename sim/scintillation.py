"""
sim/scintillation.py
=====================
Intensity scintillation via the Rytov weak-scintillation
approximation, plus Fresnel split-step propagation to estimate
pupil-plane intensity fluctuations.
"""

from __future__ import annotations

import numpy as np


def rytov_variance(r0: float, wavelength: float, altitude_m: float, D: float) -> float:
    """
    Rytov scintillation index (weak-scintillation regime), approximated
    using a single-layer Cn^2(h) integral collapsed via the Fried
    parameter:

    sigma^2_chi = 0.124 * k^(7/6) * Cn2_eff * h^(5/6)

    where ``Cn2_eff`` is derived from r0 via
    r0^(-5/3) = 0.423 * k^2 * Cn2_eff * h  (single-layer approximation),
    so Cn2_eff = r0^(-5/3) / (0.423 * k^2 * h).

    Parameters
    ----------
    r0 : float
        Fried parameter (m).
    wavelength : float
        Wavelength (m).
    altitude_m : float
        Layer altitude / propagation path length (m). Must be > 0.
    D : float
        Aperture diameter (m), unused in the scalar Rytov index but
        kept for API consistency with aperture-averaging extensions.

    Returns
    -------
    sigma2_chi : float
        Rytov variance (dimensionless).
    """
    k = 2.0 * np.pi / wavelength
    h = max(altitude_m, 1.0)

    cn2_eff = r0 ** (-5.0 / 3.0) / (0.423 * k ** 2 * h)
    sigma2_chi = 0.124 * k ** (7.0 / 6.0) * cn2_eff * h ** (5.0 / 6.0)
    return float(sigma2_chi)


def simulate_intensity_fluctuations(
    phase_screen: np.ndarray,
    wavelength: float,
    propagation_distance: float,
    N: int,
) -> np.ndarray:
    """
    Fresnel (split-step) propagation of a wavefront with a given phase
    screen, returning the resulting intensity map at the pupil plane.

    Parameters
    ----------
    phase_screen : np.ndarray, shape (N, N), radians
    wavelength : float
        Wavelength (m).
    propagation_distance : float
        Propagation distance (m).
    N : int
        Grid size; phase_screen is assumed to be N x N and span 1 m
        (unit aperture) -> pixel scale = 1/N.

    Returns
    -------
    intensity : np.ndarray, shape (N, N)
        Intensity map, normalized to unit mean.
    """
    pixel_scale = 1.0 / N
    k = 2.0 * np.pi / wavelength

    field = np.exp(1j * phase_screen)

    fx = np.fft.fftfreq(N, d=pixel_scale)
    FX, FY = np.meshgrid(fx, fx)

    # Fresnel transfer function
    H = np.exp(-1j * np.pi * wavelength * propagation_distance * (FX ** 2 + FY ** 2))

    field_ft = np.fft.fft2(field)
    propagated_ft = field_ft * H
    propagated_field = np.fft.ifft2(propagated_ft)

    intensity = np.abs(propagated_field) ** 2
    mean_i = intensity.mean()
    if mean_i > 0:
        intensity = intensity / mean_i
    return intensity


def compute_scintillation_index(intensity_map: np.ndarray, mask: np.ndarray) -> float:
    """
    Scintillation index sigma_I^2 = var(I) / mean(I)^2, evaluated over
    the valid aperture mask.

    Parameters
    ----------
    intensity_map : np.ndarray, shape (N, N)
    mask : np.ndarray of bool, shape (N, N)

    Returns
    -------
    sigma2_I : float
    """
    vals = intensity_map[mask]
    mean_i = vals.mean()
    if mean_i <= 0:
        return 0.0
    var_i = vals.var()
    return float(var_i / mean_i ** 2)


def scintillation_per_subaperture(intensity_map: np.ndarray, sensor_geometry) -> np.ndarray:
    """
    Compute the scintillation index sigma_I^2 within each valid
    subaperture of ``intensity_map``.

    Parameters
    ----------
    intensity_map : np.ndarray, shape (N, N)
    sensor_geometry : SHWFSSensor-like object
        Must expose ``n_sub`` and ``get_valid_subaperture_mask()``.

    Returns
    -------
    sigma2_per_sub : np.ndarray, shape (n_sub, n_sub)
        NaN for invalid subapertures.
    """
    n_sub = sensor_geometry.n_sub
    N = intensity_map.shape[0]
    tile = N // n_sub
    valid = sensor_geometry.get_valid_subaperture_mask()

    out = np.full((n_sub, n_sub), np.nan)
    for i in range(n_sub):
        for j in range(n_sub):
            if not valid[i, j]:
                continue
            y0, y1 = i * tile, (i + 1) * tile
            x0, x1 = j * tile, (j + 1) * tile
            tile_vals = intensity_map[y0:y1, x0:x1]
            mean_i = tile_vals.mean()
            if mean_i <= 0:
                out[i, j] = 0.0
                continue
            out[i, j] = tile_vals.var() / mean_i ** 2
    return out


def rytov_to_cn2_integral(
    sigma_rytov: float,
    wavelength: float,
    altitudes: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Invert the Rytov formula to estimate the path integral
    ``\\int Cn2(h) h^(5/6) dh`` from a measured Rytov variance.

    sigma^2_chi = 0.124 * k^(7/6) * I,   I = int Cn2(h) h^(5/6) dh

    => I = sigma^2_chi / (0.124 * k^(7/6))

    The ``altitudes`` and ``weights`` arguments describe how the total
    integral is expected to be distributed across discrete layers
    (weights summing to 1), allowing a per-layer Cn2*h^(5/6)
    decomposition to be returned implicitly via the total integral
    value.

    Parameters
    ----------
    sigma_rytov : float
        Measured Rytov variance (dimensionless).
    wavelength : float
        Wavelength (m).
    altitudes : np.ndarray
        Layer altitudes (m) -- unused directly but kept for API
        completeness / future per-layer decomposition.
    weights : np.ndarray
        Relative Cn2 weights per layer (sums to 1) -- unused directly.

    Returns
    -------
    cn2_integral : float
        Estimate of int Cn2(h) h^(5/6) dh.
    """
    k = 2.0 * np.pi / wavelength
    if sigma_rytov < 0:
        sigma_rytov = 0.0
    cn2_integral = sigma_rytov / (0.124 * k ** (7.0 / 6.0))
    return float(cn2_integral)
