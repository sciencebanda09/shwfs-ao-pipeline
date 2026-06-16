"""
viz/plot_utils.py
==================
Matplotlib-based visualization utilities for phase maps, slope fields,
spot patterns, Zernike spectra, Strehl time series, robustness curves,
turbulence parameters, closed-loop animations, and benchmark tables.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


def plot_phase_map(phase: np.ndarray, title: str, mask: np.ndarray | None = None, colorbar: bool = True, units: str = "radians", ax=None):
    """
    Display a phase map with the RdBu_r colormap and symmetric
    vmin/vmax, optionally overlaying the aperture mask boundary.

    Parameters
    ----------
    phase : np.ndarray, shape (N, N)
    title : str
    mask : np.ndarray of bool, optional
    colorbar : bool
    units : str
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    data = phase.copy()
    if mask is not None:
        data = np.where(mask, data, np.nan)

    vmax = np.nanmax(np.abs(data)) if np.any(np.isfinite(data)) else 1.0
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")

    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=1)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(units)

    return ax


def plot_slope_field(slopes_x: np.ndarray, slopes_y: np.ndarray, valid_mask: np.ndarray, ax=None):
    """
    Quiver plot of SH-WFS slope vectors, colored by magnitude.

    Parameters
    ----------
    slopes_x, slopes_y : np.ndarray, shape (n_sub, n_sub)
    valid_mask : np.ndarray of bool, shape (n_sub, n_sub)
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    n_sub = slopes_x.shape[0]
    x, y = np.meshgrid(np.arange(n_sub), np.arange(n_sub))

    sx = np.where(valid_mask, slopes_x, np.nan)
    sy = np.where(valid_mask, slopes_y, np.nan)
    magnitude = np.sqrt(sx ** 2 + sy ** 2)

    q = ax.quiver(x, y, sx, sy, magnitude, cmap="viridis")
    plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="slope magnitude (rad)")

    ax.set_title("SH-WFS slope field")
    ax.set_xlabel("subaperture x")
    ax.set_ylabel("subaperture y")
    ax.set_aspect("equal")

    return ax


def plot_spot_pattern(spot_grid: np.ndarray, reference_grid: np.ndarray, ax=None):
    """
    Scatter plot of measured spot centroid positions vs reference
    positions, with displacement vectors.

    Parameters
    ----------
    spot_grid : np.ndarray, shape (n_sub, n_sub, 2)
        Measured centroid positions.
    reference_grid : np.ndarray, shape (n_sub, n_sub, 2)
        Reference centroid positions.
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ref = reference_grid.reshape(-1, 2)
    meas = spot_grid.reshape(-1, 2)

    ax.scatter(ref[:, 0], ref[:, 1], c="gray", marker="x", label="reference")
    ax.scatter(meas[:, 0], meas[:, 1], c="red", marker=".", label="measured")

    for r, m in zip(ref, meas):
        ax.annotate(
            "", xy=(m[0], m[1]), xytext=(r[0], r[1]),
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.5),
        )

    ax.set_title("Spot displacement pattern")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend()
    ax.set_aspect("equal")

    return ax


def plot_zernike_spectrum(coeffs: np.ndarray, labels: list[str] | None = None, ax=None):
    """
    Bar chart of Zernike coefficients vs Noll index.

    Parameters
    ----------
    coeffs : np.ndarray, shape (n_zernike,)
    labels : list[str], optional
        Custom x-axis labels; defaults to Noll indices 1..n.
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    n = coeffs.shape[0]
    x = np.arange(1, n + 1)

    ax.bar(x, coeffs, color="steelblue")
    ax.set_xlabel("Noll index j")
    ax.set_ylabel("Coefficient (radians)")
    ax.set_title("Zernike coefficient spectrum")

    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)

    return ax


def plot_strehl_timeseries(t: np.ndarray, strehl_open: np.ndarray, strehl_closed: np.ndarray, strehl_predicted: np.ndarray, ax=None):
    """
    Time-series comparison of open-loop, closed-loop, and predictive
    AO Strehl ratios, with shaded standard-deviation bands.

    Parameters
    ----------
    t : np.ndarray, shape (n_frames,)
    strehl_open, strehl_closed, strehl_predicted : np.ndarray, shape (n_frames,)
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    def _band(y, color, label):
        ax.plot(t, y, color=color, label=label)
        mean = np.nanmean(y)
        std = np.nanstd(y)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.1)

    _band(strehl_open, "gray", "open loop")
    _band(strehl_closed, "tab:blue", "closed loop")
    _band(strehl_predicted, "tab:green", "predictive AO")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("Strehl ratio")
    ax.set_title("Strehl ratio time series")
    ax.legend()
    ax.set_ylim(0, 1.05)

    return ax


def plot_noise_robustness(noise_levels: np.ndarray, rms_classical: np.ndarray, rms_cnn: np.ndarray, ax=None):
    """
    Dual-curve plot of RMS WFE vs noise level on log-log axes.

    Parameters
    ----------
    noise_levels, rms_classical, rms_cnn : np.ndarray
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.loglog(noise_levels, rms_classical, "o-", label="classical (SVD/modal)")
    ax.loglog(noise_levels, rms_cnn, "s-", label="CNN/UNet")

    ax.set_xlabel("readout noise (e-)")
    ax.set_ylabel("RMS WFE (radians)")
    ax.set_title("Noise robustness")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    return ax


def plot_turbulence_params(r0_est: np.ndarray, r0_true: np.ndarray, fg_est: np.ndarray, fg_true: np.ndarray, t: np.ndarray):
    """
    Two-panel plot of estimated vs true r0 and Greenwood frequency over
    time.

    Parameters
    ----------
    r0_est, r0_true, fg_est, fg_true, t : np.ndarray

    Returns
    -------
    fig : matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(t, r0_est, label="estimated")
    axes[0].plot(t, r0_true, "--", label="true")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("r0 (m)")
    axes[0].set_title("Fried parameter r0")
    axes[0].legend()

    axes[1].plot(t, fg_est, label="estimated")
    axes[1].plot(t, fg_true, "--", label="true")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("Greenwood frequency (Hz)")
    axes[1].set_title("Greenwood frequency")
    axes[1].legend()

    plt.tight_layout()
    return fig


def animate_closed_loop(phase_sequence: np.ndarray, dm_sequence: np.ndarray, residual_sequence: np.ndarray, output_path: str):
    """
    Save an MP4 animation with three panels: phase / DM surface /
    residual.

    Parameters
    ----------
    phase_sequence, dm_sequence, residual_sequence : np.ndarray,
        shape (n_frames, N, N)
    output_path : str

    Returns
    -------
    output_path : str
    """
    n_frames = phase_sequence.shape[0]
    vmax = np.nanmax(np.abs(phase_sequence))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Phase", "DM surface", "Residual"]
    sequences = [phase_sequence, dm_sequence, residual_sequence]

    ims = []
    for ax, title, seq in zip(axes, titles, sequences):
        im = ax.imshow(seq[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    def update(frame_idx):
        for im, seq in zip(ims, sequences):
            im.set_data(seq[frame_idx])
        return ims

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)

    writer = animation.FFMpegWriter(fps=20) if animation.writers.is_available("ffmpeg") else animation.PillowWriter(fps=20)
    try:
        anim.save(output_path, writer=writer)
    except Exception:
        # Fall back to GIF if MP4/ffmpeg unavailable
        gif_path = str(output_path).rsplit(".", 1)[0] + ".gif"
        anim.save(gif_path, writer=animation.PillowWriter(fps=20))
        output_path = gif_path

    plt.close(fig)
    return output_path


def plot_benchmark_table(df: pd.DataFrame):
    """
    Render a styled pandas DataFrame display with a color-coded
    "improvement" column (computed as classical - cnn, if both
    columns are present).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    styled : pandas.io.formats.style.Styler
    """
    df = df.copy()

    if "rms_classical" in df.columns and "rms_cnn" in df.columns:
        df["improvement"] = df["rms_classical"] - df["rms_cnn"]

        def _color(val):
            color = "green" if val > 0 else "red"
            return f"color: {color}"

        if hasattr(df.style, "map"):
            styled = df.style.map(_color, subset=["improvement"])
        else:
            styled = df.style.applymap(_color, subset=["improvement"])
        return styled

    return df.style
