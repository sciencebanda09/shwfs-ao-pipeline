"""
temporal/turbulence_param.py
==============================
Estimate atmospheric turbulence parameters (r0, τ₀, Greenwood frequency,
wind velocity, Cn² profile) from reconstructed Zernike coefficient time
series and SH-WFS slope sequences.

Improvements over original:
  - tau0 now computed from HIGHER-ORDER modes (Noll j≥4), not tip/tilt.
    Tip/tilt has the longest τ₀ and is not representative of the AO
    correction bandwidth.  Median of per-mode τ₀(j) for j=4..n gives
    the physically correct coherence time for AO purposes.
  - Added direct τ₀ formula:  τ₀ = 0.314 (r₀ / v̄)^(5/3) / λ^(6/5) ×
    some_factor  — actually the standard formula is
        τ₀ = 0.314 * (r₀ / v̄)         [Roddier 1981, simplified]
    or more precisely via Greenwood:
        τ₀ = 0.314 * r₀ / V_Greenwood
    Both are computed and reported.
  - Wind velocity estimate: uses physical aperture pitch (D/n_sub) as
    the spatial baseline instead of the dimensionless 1/n_sub proxy.
    Cross-correlation lag × physical_pitch / dt → v in m/s.
  - Bootstrap uncertainty still uses the same n_boot=20 samples but
    now also bootstraps tau0 for proper error reporting.
  - r0 estimation unchanged (correct Noll formula, piston+TT excluded).
"""

from __future__ import annotations

import numpy as np

from reconstruction.bayesian import KolmogorovCovariance
from reconstruction.zernike import noll_to_zernike
from profiling.temporal_psd import compute_greenwood_frequency, compute_tau0_per_mode


# ---------------------------------------------------------------------------
# r0 from Zernike variance
# ---------------------------------------------------------------------------

def estimate_r0_from_zernike(
    zernike_coeffs_sequence: np.ndarray,
    wavelength: float,
    D: float,
) -> float:
    """
    Estimate the Fried parameter r₀ from a Zernike coefficient time series
    using Noll's variance formula (Noll 1976).

    Modes j=1,2,3 (piston + tip/tilt) are excluded.  Higher-order modes
    are used via least-squares log-linear fit on:
        log(σ²_j) = log(c_j) + (5/3) log(D/r₀)

    Parameters
    ----------
    zernike_coeffs_sequence : (n_frames, n_zernike), radians
    wavelength : float  (m) — kept for API consistency
    D : float           (m) aperture diameter

    Returns
    -------
    r0_m : float
    """
    n_modes  = zernike_coeffs_sequence.shape[1]
    variances = np.var(zernike_coeffs_sequence, axis=0)

    log_ratios = []
    for idx in range(3, n_modes):          # skip piston + tip/tilt (Noll 1,2,3)
        j = idx + 1
        coeff = KolmogorovCovariance.noll_variance(j, D, 1.0)
        if coeff <= 0 or variances[idx] <= 1e-8:
            continue
        ratio = variances[idx] / coeff     # = r₀^(-5/3)
        if ratio <= 0:
            continue
        log_ratios.append(np.log(ratio))

    if not log_ratios:
        return 0.15  # fallback

    mean_log_ratio = np.mean(log_ratios)   # = -(5/3) * log(r₀)
    r0_est = np.exp(-mean_log_ratio * (3.0 / 5.0))
    return float(r0_est)


# ---------------------------------------------------------------------------
# τ₀ estimation
# ---------------------------------------------------------------------------

def estimate_tau0_from_higher_order_modes(
    zernike_coeffs_sequence: np.ndarray,
    dt: float,
    D: float,
    r0: float,
    first_mode_index: int = 3,
) -> float:
    """
    Estimate the coherence time τ₀ from the temporal PSD of higher-order
    Zernike modes (Noll j ≥ 4, i.e. 0-based index ≥ 3).

    Returns the MEDIAN τ₀(j) across modes, which is robust to outliers
    from corrected or saturated modes.

    Tip/tilt (j=2,3) is intentionally excluded: their long τ₀ would
    over-estimate the AO correction bandwidth requirement.

    Parameters
    ----------
    zernike_coeffs_sequence : (n_frames, n_zernike), radians
    dt : float  sampling interval (s)
    D  : float  aperture diameter (m)
    r0 : float  Fried parameter (m)
    first_mode_index : int
        0-based index of the first mode to include (default 3 → Noll j=4).

    Returns
    -------
    tau0_s : float  coherence time in seconds
    """
    n_modes = zernike_coeffs_sequence.shape[1]
    if first_mode_index >= n_modes:
        # Fallback: use Greenwood on tip/tilt
        fg = compute_greenwood_frequency(zernike_coeffs_sequence, dt, modes=(2, 3))
        return 1.0 / max(fg, 1e-3)

    seq_ho = zernike_coeffs_sequence[:, first_mode_index:]
    tau0_per_mode = compute_tau0_per_mode(seq_ho, dt, D, r0)

    # Exclude modes with implausibly long τ₀ (> 1 s — numerical artefacts)
    valid = tau0_per_mode[(tau0_per_mode > 0) & (tau0_per_mode < 1.0)]
    if valid.size == 0:
        fg = compute_greenwood_frequency(zernike_coeffs_sequence, dt, modes=(2, 3))
        return 1.0 / max(fg, 1e-3)

    return float(np.median(valid))


def estimate_tau0_from_r0_and_wind(r0: float, wind_speed_ms: float) -> float:
    """
    Direct τ₀ formula (Roddier 1981 / Greenwood):

        τ₀ = 0.314 * r₀ / v̄

    This is the standard AO coherence time in the single-layer frozen-flow
    approximation.  Use as a cross-check against the PSD-based estimate.

    Parameters
    ----------
    r0           : float  Fried parameter (m)
    wind_speed_ms : float  effective wind speed (m/s)

    Returns
    -------
    tau0_s : float
    """
    if wind_speed_ms <= 0:
        return np.inf
    return 0.314 * r0 / wind_speed_ms


# ---------------------------------------------------------------------------
# Greenwood frequency
# ---------------------------------------------------------------------------

def estimate_greenwood_frequency(
    zernike_coeffs_sequence: np.ndarray,
    dt: float,
    D: float,
    r0: float,
) -> float:
    """
    Estimate the Greenwood frequency from the temporal PSD of tip/tilt
    Zernike modes, fitting an f^(-11/6) von Karman model.

    Note: this gives fG representative of tip/tilt only.  Use the
    higher-order-mode estimate for the AO bandwidth requirement.
    """
    return compute_greenwood_frequency(zernike_coeffs_sequence, dt, modes=(2, 3))


# ---------------------------------------------------------------------------
# Wind velocity
# ---------------------------------------------------------------------------

def estimate_wind_velocity_frozen_flow(
    zernike_coeffs_sequence: np.ndarray,
    dt: float,
    n_subapertures: int,
    aperture_diameter_m: float = 0.5,
) -> tuple[float, float]:
    """
    Estimate a single effective wind vector (speed, direction) using
    temporal cross-correlation of the tip and tilt Zernike mode time series
    under the frozen-flow (Taylor) hypothesis.

    The temporal lag τ_peak at which cross-correlation peaks is related to
    the wind speed by:
        v = physical_pitch / τ_peak

    where physical_pitch = D / n_sub is the subaperture side length in
    metres — the correct spatial baseline for converting lag to speed.

    Parameters
    ----------
    zernike_coeffs_sequence : (n_frames, n_zernike)
    dt : float
    n_subapertures : int
    aperture_diameter_m : float
        Physical aperture diameter in metres (default 0.5 m).
        Must be supplied for a physically correct speed estimate.

    Returns
    -------
    speed_ms, direction_deg : float, float
    """
    tip  = zernike_coeffs_sequence[:, 1] - zernike_coeffs_sequence[:, 1].mean()
    tilt = zernike_coeffs_sequence[:, 2] - zernike_coeffs_sequence[:, 2].mean()

    n    = tip.shape[0]
    corr = np.correlate(tip, tilt, mode="full")
    lags = np.arange(-(n - 1), n)

    peak_idx = int(np.argmax(np.abs(corr)))
    lag      = lags[peak_idx]

    # Physical subaperture pitch in metres
    physical_pitch_m = aperture_diameter_m / n_subapertures

    if lag == 0:
        speed_ms = 0.0
    else:
        speed_ms = float(abs(physical_pitch_m / (lag * dt)))

    direction_deg = float(
        np.rad2deg(np.arctan2(np.mean(tilt), np.mean(tip) + 1e-12)) % 360.0
    )
    return speed_ms, direction_deg


# ---------------------------------------------------------------------------
# Cn² profile
# ---------------------------------------------------------------------------

def compute_cn2_profile(
    r0_per_layer: np.ndarray,
    altitudes: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """
    Compute a Cn²(h) profile from per-layer r₀ estimates.

        Cn²_i * Δh_i ∝ r₀_i^(-5/3)

    Normalized to sum to 1, giving relative turbulence-strength weights.
    """
    weights = r0_per_layer ** (-5.0 / 3.0)
    total   = weights.sum()
    return weights / total if total > 0 else weights


# ---------------------------------------------------------------------------
# Unified estimator class
# ---------------------------------------------------------------------------

class TurbulenceParameterEstimator:
    """
    Wraps all turbulence-parameter estimation functions into a single
    fit/report interface.

    Parameters
    ----------
    wavelength : float  (m)
    D : float           (m) aperture diameter
    dt : float          (s) sampling interval
    n_subapertures : int
    """

    def __init__(self, wavelength: float, D: float, dt: float, n_subapertures: int):
        self.wavelength      = wavelength
        self.D               = D
        self.dt              = dt
        self.n_subapertures  = n_subapertures
        self._results: dict  = {}

    def fit(
        self,
        zernike_sequence: np.ndarray,
        slopes_sequence: np.ndarray | None = None,
    ) -> dict:
        """
        Run all estimators on the provided data.

        Parameters
        ----------
        zernike_sequence : (n_frames, n_zernike), radians
        slopes_sequence  : (n_frames, 2, n_sub, n_sub), optional

        Returns
        -------
        results : dict with keys:
            r0_m, r0_m_std,
            tau0_s, tau0_s_std,
            tau0_direct_s,
            greenwood_freq_hz, greenwood_freq_hz_std,
            wind_speed_ms, wind_direction_deg
        """
        r0_est = estimate_r0_from_zernike(
            zernike_sequence, self.wavelength, self.D
        )
        fg_est = estimate_greenwood_frequency(
            zernike_sequence, self.dt, self.D, r0_est
        )
        tau0_ho = estimate_tau0_from_higher_order_modes(
            zernike_sequence, self.dt, self.D, r0_est
        )
        speed_est, dir_est = estimate_wind_velocity_frozen_flow(
            zernike_sequence, self.dt, self.n_subapertures,
            aperture_diameter_m=self.D,
        )
        tau0_direct = estimate_tau0_from_r0_and_wind(r0_est, speed_est)

        # Bootstrap uncertainty (n_boot=20 resamples)
        n_frames = zernike_sequence.shape[0]
        n_boot   = 20
        rng      = np.random.default_rng(0)
        r0_boot, fg_boot, tau0_boot = [], [], []
        for _ in range(n_boot):
            idx    = rng.choice(n_frames, size=n_frames, replace=True)
            sample = zernike_sequence[idx]
            r0b    = estimate_r0_from_zernike(sample, self.wavelength, self.D)
            r0_boot.append(r0b)
            fg_boot.append(
                estimate_greenwood_frequency(sample, self.dt, self.D, r0b)
            )
            tau0_boot.append(
                estimate_tau0_from_higher_order_modes(sample, self.dt, self.D, r0b)
            )

        self._results = {
            "r0_m":                  r0_est,
            "r0_m_std":              float(np.std(r0_boot)),
            "tau0_s":                tau0_ho,
            "tau0_s_std":            float(np.std(tau0_boot)),
            "tau0_direct_s":         tau0_direct,   # cross-check via r0/v
            "greenwood_freq_hz":     fg_est,
            "greenwood_freq_hz_std": float(np.std(fg_boot)),
            "wind_speed_ms":         speed_est,
            "wind_direction_deg":    dir_est,
        }
        return self._results

    def report(self) -> dict:
        """Return the dict of all estimated parameters with uncertainties."""
        return self._results
