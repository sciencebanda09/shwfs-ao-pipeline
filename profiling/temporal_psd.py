"""
profiling/temporal_psd.py
===========================
Mode-dependent temporal power spectral density (PSD) analysis: fit
von Karman temporal PSD models, extract Greenwood frequency, and
estimate per-mode coherence times tau_0(j).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def compute_temporal_psd(zernike_sequence: np.ndarray, mode_index: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT-based one-sided temporal PSD of a single Zernike
    mode's time series.

    Parameters
    ----------
    zernike_sequence : np.ndarray, shape (n_frames, n_zernike)
    mode_index : int
        Index into the n_zernike axis (0-based).
    dt : float
        Sampling interval (s).

    Returns
    -------
    freq : np.ndarray, shape (n_freq,)
        Positive frequency array (Hz).
    psd : np.ndarray, shape (n_freq,)
        Power spectral density (rad^2/Hz).
    """
    x = zernike_sequence[:, mode_index]
    n = x.shape[0]

    x = x - np.mean(x)
    fft_vals = np.fft.rfft(x)
    psd = (np.abs(fft_vals) ** 2) / (n / dt)

    freq = np.fft.rfftfreq(n, d=dt)

    # Exclude DC
    return freq[1:], psd[1:]


def _von_karman_temporal_model(f: np.ndarray, fg: float, sigma2: float) -> np.ndarray:
    """S(f) = sigma2 * (f^2 + fg^2)^(-11/6), normalized model shape."""
    return sigma2 * (f ** 2 + fg ** 2) ** (-11.0 / 6.0)


def fit_von_karman_temporal_psd(freq: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    """
    Fit S(f) ∝ (f^2 + fg^2)^(-11/6) to a measured temporal PSD.

    Parameters
    ----------
    freq : np.ndarray, shape (n_freq,)
    psd : np.ndarray, shape (n_freq,)

    Returns
    -------
    fg_hz : float
        Fitted Greenwood-like frequency (Hz).
    sigma2_fit : float
        Fitted amplitude scale.
    """
    valid = (freq > 0) & (psd > 0) & np.isfinite(psd)
    f = freq[valid]
    p = psd[valid]

    if f.size < 3:
        return 1.0, float(np.mean(p)) if p.size else 1.0

    fg0 = max(f[0], 0.1)
    sigma2_0 = p[0] * (fg0 ** (11.0 / 3.0))

    try:
        popt, _ = curve_fit(
            _von_karman_temporal_model, f, p,
            p0=[fg0, sigma2_0],
            maxfev=5000,
            bounds=([1e-3, 1e-12], [1e4, 1e6]),
        )
        fg_hz, sigma2_fit = popt
    except Exception:
        fg_hz, sigma2_fit = fg0, sigma2_0

    return float(fg_hz), float(sigma2_fit)


def compute_tau0_per_mode(zernike_sequence: np.ndarray, dt: float, D: float, r0: float) -> np.ndarray:
    """
    Compute the coherence time tau_0(j) for each Zernike mode by fitting
    the temporal PSD of each mode independently.

    tau_0(j) = 1 / fg(j)

    Parameters
    ----------
    zernike_sequence : np.ndarray, shape (n_frames, n_zernike)
    dt : float
    D : float
        Aperture diameter (m), unused directly (kept for API
        consistency with analytical models).
    r0 : float
        Fried parameter (m), unused directly.

    Returns
    -------
    tau0 : np.ndarray, shape (n_zernike,)
        Coherence time per mode (seconds).
    """
    n_modes = zernike_sequence.shape[1]
    tau0 = np.zeros(n_modes)

    for j in range(n_modes):
        freq, psd = compute_temporal_psd(zernike_sequence, j, dt)
        fg, _ = fit_von_karman_temporal_psd(freq, psd)
        fg = max(fg, 1e-3)
        tau0[j] = 1.0 / fg

    return tau0


def compute_greenwood_frequency(zernike_sequence: np.ndarray, dt: float, modes: tuple[int, int] = (2, 3)) -> float:
    """
    Compute the scalar Greenwood frequency from the temporal PSD of the
    tip/tilt modes (Noll indices 2 and 3 -> 0-based indices 1 and 2).

    Parameters
    ----------
    zernike_sequence : np.ndarray, shape (n_frames, n_zernike)
    dt : float
    modes : tuple[int, int]
        1-based Noll indices of tip/tilt modes (default (2, 3)).

    Returns
    -------
    fg_hz : float
    """
    fgs = []
    for noll_j in modes:
        idx = noll_j - 1
        if idx >= zernike_sequence.shape[1]:
            continue
        freq, psd = compute_temporal_psd(zernike_sequence, idx, dt)
        fg, _ = fit_von_karman_temporal_psd(freq, psd)
        fgs.append(fg)

    if not fgs:
        return 1.0
    return float(np.mean(fgs))


def plot_temporal_psd_fits(zernike_sequence: np.ndarray, dt: float, modes_to_plot: list[int], output_path: str) -> None:
    """
    Plot measured temporal PSD and fitted von Karman model for each mode
    in ``modes_to_plot``, annotating fg and tau_0, and marking the
    -17/3 high-frequency slope line.

    Parameters
    ----------
    zernike_sequence : np.ndarray, shape (n_frames, n_zernike)
    dt : float
    modes_to_plot : list[int]
        0-based mode indices.
    output_path : str
        Output image path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plots = len(modes_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, mode_idx in zip(axes, modes_to_plot):
        freq, psd = compute_temporal_psd(zernike_sequence, mode_idx, dt)
        fg, sigma2 = fit_von_karman_temporal_psd(freq, psd)
        tau0 = 1.0 / max(fg, 1e-3)

        model = _von_karman_temporal_model(freq, fg, sigma2)

        ax.loglog(freq, psd, ".", alpha=0.5, label="measured")
        ax.loglog(freq, model, "-", label="von Karman fit")

        # -17/3 slope reference line anchored at the highest frequency
        if freq.size > 1:
            f_ref = freq[-1]
            psd_ref = model[-1]
            slope_line = psd_ref * (freq / f_ref) ** (-17.0 / 3.0)
            ax.loglog(freq, slope_line, "--", color="gray", label="-17/3 slope")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (rad^2/Hz)")
        ax.set_title(f"Mode {mode_idx + 1}: fg={fg:.2f} Hz, tau0={tau0 * 1000:.2f} ms")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
