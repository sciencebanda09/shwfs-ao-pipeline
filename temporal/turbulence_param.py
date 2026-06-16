"""
temporal/turbulence_param.py
==============================
Estimate atmospheric turbulence parameters (r0, Greenwood frequency,
wind velocity, Cn^2 profile) from reconstructed Zernike coefficient
time series and SH-WFS slope sequences.
"""

from __future__ import annotations

import numpy as np

from reconstruction.bayesian import KolmogorovCovariance
from reconstruction.zernike import noll_to_zernike
from profiling.temporal_psd import compute_greenwood_frequency


def estimate_r0_from_zernike(zernike_coeffs_sequence: np.ndarray, wavelength: float, D: float) -> float:
    """
    Estimate the Fried parameter r0 from a Zernike coefficient time
    series using Noll's variance formula.

    sigma^2_j = coeff_j * (D/r0)^(5/3)
    => log(sigma^2_j) = log(coeff_j) + (5/3) * log(D/r0)
    => log(sigma^2_j) - log(coeff_j) = (5/3) * (log D - log r0)

    We fit r0 via least squares on modes j=2..n (excluding piston),
    using the relation: log(sigma^2_j / coeff_j) = (5/3) * log(D/r0),
    so the average of log(sigma^2_j/coeff_j) gives (5/3)*log(D/r0).

    Parameters
    ----------
    zernike_coeffs_sequence : np.ndarray, shape (n_frames, n_zernike), radians
    wavelength : float
        Wavelength (m), unused directly but kept for API consistency.
    D : float
        Aperture diameter (m).

    Returns
    -------
    r0_m : float
    """
    n_modes = zernike_coeffs_sequence.shape[1]
    variances = np.var(zernike_coeffs_sequence, axis=0)

    # FIX: skip piston (j=1, idx=0) AND tip/tilt (j=2,3, idx=1,2).
    # Tip/tilt is often removed by a steering mirror before the WFS,
    # and their Noll coefficients (0.448) dominate the fit.  Using only
    # higher-order modes (j>=4, idx>=3) gives a physically correct r0.
    # Also guard against near-zero variance (corrected or saturated modes).
    log_ratios = []
    for idx in range(3, n_modes):  # skip piston+tip+tilt (Noll j=1,2,3)
        j = idx + 1
        coeff = KolmogorovCovariance.noll_variance(j, D, 1.0)  # coeff_j * D^(5/3)
        if coeff <= 0 or variances[idx] <= 0:
            continue
        ratio = variances[idx] / coeff  # = r0^(-5/3)
        if ratio <= 0:
            continue
        # Sanity: ignore modes where variance is suspiciously near zero
        # (happens when a mode is fully corrected in closed loop).
        if variances[idx] < 1e-8:
            continue
        log_ratios.append(np.log(ratio))

    if not log_ratios:
        return 0.15  # fallback default

    mean_log_ratio = np.mean(log_ratios)  # = -5/3 * log(r0)
    log_r0 = -mean_log_ratio * (3.0 / 5.0)
    r0_est = np.exp(log_r0)
    return float(r0_est)


def estimate_greenwood_frequency(zernike_coeffs_sequence: np.ndarray, dt: float, D: float, r0: float) -> float:
    """
    Estimate the Greenwood frequency from the temporal PSD of tip/tilt
    Zernike modes, fitting an f^(-17/3) high-frequency slope.

    Parameters
    ----------
    zernike_coeffs_sequence : np.ndarray, shape (n_frames, n_zernike)
    dt : float
    D : float
        Aperture diameter (m), unused directly.
    r0 : float
        Fried parameter (m), unused directly.

    Returns
    -------
    fg_hz : float
    """
    return compute_greenwood_frequency(zernike_coeffs_sequence, dt, modes=(2, 3))


def estimate_wind_velocity_frozen_flow(zernike_coeffs_sequence: np.ndarray, dt: float, n_subapertures: int) -> tuple[float, float]:
    """
    Estimate a single effective wind vector (speed, direction) using
    temporal cross-correlation of the tip and tilt Zernike mode time
    series (proxy for slope time series under frozen flow).

    The temporal lag at which the cross-correlation between tip
    (mode index 1, Noll j=2) and tilt (mode index 2, Noll j=3) peaks is
    used to infer a relative phase delay; combined with the known
    aperture geometry this yields a coarse wind-speed/direction
    estimate.

    Parameters
    ----------
    zernike_coeffs_sequence : np.ndarray, shape (n_frames, n_zernike)
    dt : float
    n_subapertures : int
        Used to set a characteristic length scale for speed estimation.

    Returns
    -------
    speed_ms, direction_deg : float, float
    """
    tip = zernike_coeffs_sequence[:, 1] - np.mean(zernike_coeffs_sequence[:, 1])
    tilt = zernike_coeffs_sequence[:, 2] - np.mean(zernike_coeffs_sequence[:, 2])

    n = tip.shape[0]
    corr = np.correlate(tip, tilt, mode="full")
    lags = np.arange(-(n - 1), n)

    peak_idx = np.argmax(np.abs(corr))
    lag = lags[peak_idx]

    # Characteristic length scale: subaperture pitch on a normalized
    # aperture -> use 1/n_subapertures as a proxy spatial scale.
    length_scale = 1.0 / n_subapertures

    if lag == 0:
        speed_ms = 0.0
    else:
        speed_ms = float(abs(length_scale / (lag * dt)))

    # Direction inferred from the relative sign/phase of tip vs tilt
    direction_deg = float(np.rad2deg(np.arctan2(np.mean(tilt), np.mean(tip) + 1e-12)) % 360.0)

    return speed_ms, direction_deg


def compute_cn2_profile(r0_per_layer: np.ndarray, altitudes: np.ndarray, wavelength: float) -> np.ndarray:
    """
    Compute a Cn^2(h) profile from per-layer r0 estimates using a
    simple scintillometry-style relation:

    Cn^2_i * delta_h_i ∝ r0_i^(-5/3)

    The per-layer Cn^2*delta_h contributions are normalized to sum to
    1, giving the relative turbulence-strength profile.

    Parameters
    ----------
    r0_per_layer : np.ndarray, shape (n_layers,)
        Per-layer effective Fried parameters (m).
    altitudes : np.ndarray, shape (n_layers,)
        Layer altitudes (m).
    wavelength : float
        Wavelength (m), unused directly but kept for API consistency.

    Returns
    -------
    cn2_profile : np.ndarray, shape (n_layers,)
        Normalized Cn^2 * delta_h weights.
    """
    weights = r0_per_layer ** (-5.0 / 3.0)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


class TurbulenceParameterEstimator:
    """
    Wraps all turbulence-parameter estimation functions into a single
    fit/report interface.

    Parameters
    ----------
    wavelength : float
        Wavelength (m).
    D : float
        Aperture diameter (m).
    dt : float
        Sampling interval (s).
    n_subapertures : int
    """

    def __init__(self, wavelength: float, D: float, dt: float, n_subapertures: int):
        self.wavelength = wavelength
        self.D = D
        self.dt = dt
        self.n_subapertures = n_subapertures
        self._results: dict = {}

    def fit(self, zernike_sequence: np.ndarray, slopes_sequence: np.ndarray | None = None) -> dict:
        """
        Run all estimators on the provided data.

        Parameters
        ----------
        zernike_sequence : np.ndarray, shape (n_frames, n_zernike)
        slopes_sequence : np.ndarray, optional, shape (n_frames, 2, n_sub, n_sub)
            Unused by current estimators but accepted for API
            completeness / future extensions.

        Returns
        -------
        results : dict
        """
        r0_est = estimate_r0_from_zernike(zernike_sequence, self.wavelength, self.D)
        fg_est = estimate_greenwood_frequency(zernike_sequence, self.dt, self.D, r0_est)
        speed_est, dir_est = estimate_wind_velocity_frozen_flow(zernike_sequence, self.dt, self.n_subapertures)

        # Bootstrap-based uncertainty estimates
        n_frames = zernike_sequence.shape[0]
        n_boot = 20
        rng = np.random.default_rng(0)
        r0_boot = []
        fg_boot = []
        for _ in range(n_boot):
            idx = rng.choice(n_frames, size=n_frames, replace=True)
            sample = zernike_sequence[idx]
            r0_boot.append(estimate_r0_from_zernike(sample, self.wavelength, self.D))
            fg_boot.append(estimate_greenwood_frequency(sample, self.dt, self.D, r0_est))

        self._results = {
            "r0_m": r0_est,
            "r0_m_std": float(np.std(r0_boot)),
            "greenwood_freq_hz": fg_est,
            "greenwood_freq_hz_std": float(np.std(fg_boot)),
            "wind_speed_ms": speed_est,
            "wind_direction_deg": dir_est,
        }
        return self._results

    def report(self) -> dict:
        """Return the dict of all estimated parameters with uncertainties."""
        return self._results
