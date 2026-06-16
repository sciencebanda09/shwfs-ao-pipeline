"""
profiling/slodar.py
====================
SLODAR (SLOpe Detection And Ranging) altitude-resolved Cn^2(h)
profiler via cross-correlation of SH-WFS slope maps from two stars at
known angular separation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit


class SLODARProfiler:
    """
    SLODAR Cn^2(h) profiler.

    Each turbulent layer at altitude h produces a peak in the
    cross-correlation of slope maps from two stars at offset
    Delta = h * theta / d (d = subaperture pitch), with peak strength
    proportional to Cn^2(h) * delta_h.

    Parameters
    ----------
    n_subapertures : int
    subaperture_pitch_m : float
    star_separation_rad : float
    max_altitude_m : float
    n_bins : int
    """

    def __init__(self, n_subapertures: int, subaperture_pitch_m: float, star_separation_rad: float, max_altitude_m: float, n_bins: int):
        self.n_sub = n_subapertures
        self.pitch = subaperture_pitch_m
        self.theta = star_separation_rad
        self.max_altitude = max_altitude_m
        self.n_bins = n_bins

        self.altitude_bins = np.linspace(0, max_altitude_m, n_bins)
        # Expected cross-correlation peak offset (in subaperture units)
        # for each altitude bin: Delta = h * theta / pitch
        self.expected_offsets = self.altitude_bins * self.theta / self.pitch

    def cross_correlate_slopes(self, slopes_star1: np.ndarray, slopes_star2: np.ndarray) -> np.ndarray:
        """
        Compute the 2D cross-correlation map of two slope maps via FFT.

        Parameters
        ----------
        slopes_star1, slopes_star2 : np.ndarray, shape (n_sub, n_sub)

        Returns
        -------
        corr_map : np.ndarray, shape (2*n_sub-1, 2*n_sub-1)
        """
        s1 = np.nan_to_num(slopes_star1)
        s2 = np.nan_to_num(slopes_star2)

        s1 = s1 - s1.mean()
        s2 = s2 - s2.mean()

        corr_map = fftconvolve(s1, s2[::-1, ::-1], mode="full")
        return corr_map

    def extract_cn2_profile(self, correlation_map: np.ndarray) -> np.ndarray:
        """
        Identify peaks in the correlation map at the expected altitude
        offsets, fit a Gaussian to each, and use the peak amplitude as
        a Cn^2(h) * delta_h estimate.

        Parameters
        ----------
        correlation_map : np.ndarray, shape (2*n_sub-1, 2*n_sub-1)

        Returns
        -------
        cn2_profile : np.ndarray, shape (n_bins,)
        """
        center = correlation_map.shape[0] // 2
        # 1D radial profile along the row axis through the center
        profile_1d = correlation_map[center, :]

        cn2_profile = np.zeros(self.n_bins)
        for i, offset in enumerate(self.expected_offsets):
            idx = int(round(center + offset))
            if 0 <= idx < profile_1d.shape[0]:
                window = 1
                lo = max(0, idx - window)
                hi = min(profile_1d.shape[0], idx + window + 1)
                cn2_profile[i] = max(profile_1d[lo:hi].max(), 0.0)
            else:
                cn2_profile[i] = 0.0

        return cn2_profile

    def fit_profile(self, cn2_profile: np.ndarray, altitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth and normalize the recovered Cn^2(h) profile so it sums
        to 1 (relative weighting), for comparison with input simulation
        weights.

        Parameters
        ----------
        cn2_profile : np.ndarray, shape (n_bins,)
        altitudes : np.ndarray, shape (n_bins,)

        Returns
        -------
        altitudes, cn2_profile_norm : np.ndarray, np.ndarray
        """
        profile = np.clip(cn2_profile, 0, None)
        total = profile.sum()
        if total > 0:
            profile_norm = profile / total
        else:
            profile_norm = profile
        return altitudes, profile_norm

    def run(self, slopes_star1_sequence: np.ndarray, slopes_star2_sequence: np.ndarray) -> np.ndarray:
        """
        Average the cross-correlation over N frames for improved SNR,
        then extract the Cn^2(h) profile.

        Parameters
        ----------
        slopes_star1_sequence, slopes_star2_sequence : np.ndarray,
            shape (n_frames, n_sub, n_sub)

        Returns
        -------
        cn2_profile : np.ndarray, shape (n_bins,)
        """
        n_frames = slopes_star1_sequence.shape[0]
        accum = None

        for k in range(n_frames):
            corr = self.cross_correlate_slopes(slopes_star1_sequence[k], slopes_star2_sequence[k])
            if accum is None:
                accum = corr
            else:
                accum += corr

        accum /= n_frames
        return self.extract_cn2_profile(accum)


def simulate_dual_star_slopes(atmosphere, sensor, theta_rad: float, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate SH-WFS slope sequences for two stars separated by angle
    ``theta_rad``. The second star's wavefront is approximated by
    laterally shifting each turbulence layer's phase screen by
    h * theta_rad before integration (mimicking the geometric
    cone-effect offset between the two guide-star lines of sight).

    Parameters
    ----------
    atmosphere : MultiLayerAtmosphere
    sensor : SHWFSSensor
    theta_rad : float
        Angular separation (radians).
    n_frames : int

    Returns
    -------
    slopes1_sequence, slopes2_sequence : np.ndarray,
        shape (n_frames, n_sub, n_sub) each, for slopes_x only
        (returned as the combined (x) slope component for simplicity).
    """
    from scipy import ndimage
    from sim.phase_screen import apply_aperture_mask

    n_sub = sensor.n_sub
    slopes1_seq = np.zeros((n_frames, n_sub, n_sub))
    slopes2_seq = np.zeros((n_frames, n_sub, n_sub))

    for k in range(n_frames):
        atmosphere.evolve(0.001)

        # Star 1: standard integrated phase
        phase1 = atmosphere.get_integrated_phase_radians()
        phase1_masked = apply_aperture_mask(phase1, "circular")
        sx1, _ = sensor.propagate(phase1_masked)
        slopes1_seq[k] = sx1

        # Star 2: shift each layer by altitude * theta / pixel_scale
        total_phase2 = np.zeros_like(phase1)
        for w, layer in zip(atmosphere.cn2_weights, atmosphere.layers):
            shift_pixels = (layer.altitude * theta_rad) / atmosphere.pixel_scale
            shifted = ndimage.shift(layer.phase, shift=[0, shift_pixels], mode="wrap", order=1)
            total_phase2 += w * shifted

        phase2_masked = apply_aperture_mask(total_phase2, "circular")
        sx2, _ = sensor.propagate(phase2_masked)
        slopes2_seq[k] = sx2

    return slopes1_seq, slopes2_seq


def validate_slodar(cn2_estimated: np.ndarray, cn2_true_weights: np.ndarray, layer_altitudes: np.ndarray) -> dict:
    """
    Compare the recovered Cn^2(h) profile to the known simulation
    truth by mapping the true discrete layer weights onto the
    profiler's altitude bins and computing the L2 error.

    Parameters
    ----------
    cn2_estimated : np.ndarray, shape (n_bins,)
        Normalized estimated profile (e.g. from fit_profile).
    cn2_true_weights : np.ndarray, shape (n_layers,)
        True Cn2 weights per simulated layer (sums to 1).
    layer_altitudes : np.ndarray, shape (n_layers,)
        True layer altitudes (m). Must lie within the profiler's
        altitude bin range.

    Returns
    -------
    result : dict with keys 'l2_error', 'cn2_true_binned', 'recovered_layer_altitudes'
    """
    n_bins = cn2_estimated.shape[0]
    max_alt = layer_altitudes.max() if layer_altitudes.size else 1.0
    bin_edges = np.linspace(0, max_alt * 1.2, n_bins + 1)

    cn2_true_binned = np.zeros(n_bins)
    for w, h in zip(cn2_true_weights, layer_altitudes):
        bin_idx = np.searchsorted(bin_edges, h) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        cn2_true_binned[bin_idx] += w

    if cn2_true_binned.sum() > 0:
        cn2_true_binned = cn2_true_binned / cn2_true_binned.sum()

    l2_error = float(np.sqrt(np.sum((cn2_estimated - cn2_true_binned) ** 2)))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    recovered_altitudes = bin_centers[cn2_estimated > 0.05]

    return {
        "l2_error": l2_error,
        "cn2_true_binned": cn2_true_binned,
        "recovered_layer_altitudes": recovered_altitudes,
    }
