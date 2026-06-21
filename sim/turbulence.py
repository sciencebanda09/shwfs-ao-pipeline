"""
sim/turbulence.py
==================
Atmospheric turbulence phase-screen generation using Kolmogorov / von
Karman power spectra, the subharmonic FFT method (Johansson & Gavel
1994), and frozen-flow (Taylor hypothesis) layer evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage


def kolmogorov_psd(freq: np.ndarray, r0: float) -> np.ndarray:
    """
    Kolmogorov power spectral density of phase fluctuations.

    PSD(f) = 0.023 * r0^(-5/3) * f^(-11/3)

    Parameters
    ----------
    freq : np.ndarray
        Spatial frequency magnitude (1/m), zeros are handled by masking.
    r0 : float
        Fried parameter (m).

    Returns
    -------
    psd : np.ndarray, same shape as freq
    """
    psd = np.zeros_like(freq, dtype=float)
    nonzero = freq > 0
    psd[nonzero] = 0.023 * r0 ** (-5.0 / 3.0) * freq[nonzero] ** (-11.0 / 3.0)
    return psd


def von_karman_psd(freq: np.ndarray, r0: float, L0: float) -> np.ndarray:
    """
    Von Karman power spectral density of phase fluctuations.

    PSD(f) = 0.023 * r0^(-5/3) * (f^2 + (1/L0)^2)^(-11/6)

    Parameters
    ----------
    freq : np.ndarray
        Spatial frequency magnitude (1/m).
    r0 : float
        Fried parameter (m).
    L0 : float
        Outer scale of turbulence (m).

    Returns
    -------
    psd : np.ndarray, same shape as freq
    """
    f0_sq = (1.0 / L0) ** 2
    psd = 0.023 * r0 ** (-5.0 / 3.0) * (freq ** 2 + f0_sq) ** (-11.0 / 6.0)
    return psd


def _psd_func(model: str, freq: np.ndarray, r0: float, L0: float) -> np.ndarray:
    if model == "kolmogorov":
        return kolmogorov_psd(freq, r0)
    elif model == "von_karman":
        return von_karman_psd(freq, r0, L0)
    else:
        raise ValueError(f"Unknown turbulence model: {model}")


def generate_phase_screen(
    N: int,
    pixel_scale: float,
    r0: float,
    L0: float,
    model: str = "von_karman",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an N x N atmospheric phase screen (radians) using the
    FFT-based method with subharmonic correction (Johansson & Gavel
    1994) for accurate low-frequency content.

    Steps
    -----
    1. Build a grid of spatial frequencies for the N x N FFT screen.
    2. Evaluate the chosen PSD on that grid.
    3. Multiply sqrt(PSD) by a complex Gaussian random field.
    4. Inverse FFT to obtain the phase screen in radians.
    5. Add a subharmonic correction covering frequencies below the
       fundamental FFT frequency to recover low-frequency power.

    Parameters
    ----------
    N : int
        Screen size (N x N pixels).
    pixel_scale : float
        Physical size of one pixel (m).
    r0 : float
        Fried parameter (m).
    L0 : float
        Outer scale (m).
    model : str
        'kolmogorov' or 'von_karman'.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    phase : np.ndarray, shape (N, N), radians
    """
    rng = np.random.default_rng(seed)

    # --- Main FFT screen ------------------------------------------------
    df = 1.0 / (N * pixel_scale)  # fundamental spatial frequency
    fx = np.fft.fftfreq(N, d=pixel_scale)
    fy = np.fft.fftfreq(N, d=pixel_scale)
    FX, FY = np.meshgrid(fx, fy)
    freq = np.sqrt(FX ** 2 + FY ** 2)

    psd = _psd_func(model, freq, r0, L0)
    psd[0, 0] = 0.0  # remove piston

    # Complex Gaussian random field with unit variance per component
    cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))

    # Scale by sqrt(PSD) and by 1/(N*pixel_scale) for proper FFT normalization
    fourier_phase = cn * np.sqrt(psd) * df

    phase = np.fft.ifft2(fourier_phase * N * N).real

    # --- Subharmonic correction ------------------------------------------
    # Add low-frequency content from sub-harmonics below the fundamental
    # frequency, following Johansson & Gavel (1994). We use 3 levels of
    # subharmonics on a small 3x3 grid of frequencies for each level.
    n_subharmonic_levels = 3
    x = (np.arange(N) - N / 2.0) * pixel_scale
    X, Y = np.meshgrid(x, x)

    for level in range(1, n_subharmonic_levels + 1):
        scale = 3.0 ** (-level)
        df_sub = df * scale
        fx_sub = (np.arange(-1, 2)) * df_sub
        fy_sub = (np.arange(-1, 2)) * df_sub
        for i in fx_sub:
            for j in fy_sub:
                if i == 0 and j == 0:
                    continue
                f_mag = np.sqrt(i ** 2 + j ** 2)
                psd_val = _psd_func(model, np.array([f_mag]), r0, L0)[0]
                amp = np.sqrt(psd_val * df_sub ** 2)
                rand_phase = rng.uniform(0, 2 * np.pi)
                a = rng.standard_normal()
                b = rng.standard_normal()
                phase += (
                    amp
                    * (a * np.cos(2 * np.pi * (i * X + j * Y) + rand_phase)
                       + b * np.sin(2 * np.pi * (i * X + j * Y) + rand_phase))
                )

    return phase


class TurbulenceLayer:
    """
    A single atmospheric turbulence layer with frozen-flow evolution.

    Parameters
    ----------
    N : int
        Phase screen size (N x N).
    pixel_scale : float
        Physical pixel size (m).
    r0 : float
        Layer-effective Fried parameter (m).
    L0 : float
        Outer scale (m).
    altitude : float
        Layer altitude (m).
    wind_speed : float
        Wind speed (m/s).
    wind_direction : float
        Wind direction (degrees, 0 = +x axis).
    seed : int, optional
        RNG seed.
    """

    def __init__(
        self,
        N: int,
        pixel_scale: float,
        r0: float,
        L0: float,
        altitude: float,
        wind_speed: float,
        wind_direction: float,
        seed: Optional[int] = None,
    ):
        self.N = N
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.altitude = altitude
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self._residual_shift = np.array([0.0, 0.0])
        self._seed = seed
        self._N = N
        self._pixel_scale = pixel_scale
        self._r0 = r0
        self._L0 = L0

        self.phase = generate_phase_screen(
            N, pixel_scale, r0, L0, model="von_karman", seed=seed
        )

    def evolve(self, dt: float) -> None:
        """
        Evolve the phase screen by wind_speed * dt along wind_direction
        using the frozen-flow (Taylor) hypothesis. The screen is shifted
        (with wrap-around) by the corresponding number of pixels.
        """
        theta = np.deg2rad(self.wind_direction)
        dx = self.wind_speed * dt * np.cos(theta) / self.pixel_scale
        dy = self.wind_speed * dt * np.sin(theta) / self.pixel_scale

        # Accumulate fractional shifts so sub-pixel motion is preserved
        self._residual_shift += np.array([dy, dx])
        shift_pixels = self._residual_shift

        self.phase = ndimage.shift(
            self.phase, shift=shift_pixels, mode="wrap", order=1
        )
        self._residual_shift = np.array([0.0, 0.0])


class MultiLayerAtmosphere:
    """
    Collection of TurbulenceLayer objects forming a multi-layer
    atmosphere with Cn2-weighted integration.

    Parameters
    ----------
    layers_config : Sequence[dict]
        Each dict must contain keys: 'r0', 'L0', 'altitude',
        'wind_speed', 'wind_direction', 'cn2_weight', and optionally
        'seed'.
    N : int
        Phase screen grid size.
    pixel_scale : float
        Physical pixel size (m).
    wavelength_m : float
        Wavelength used to convert phase (radians) to wavefront error
        (meters), default 550e-9.
    """

    def __init__(
        self,
        layers_config: Sequence[dict],
        N: int,
        pixel_scale: float,
        wavelength_m: float = 550e-9,
    ):
        self.N = N
        self.pixel_scale = pixel_scale
        self.wavelength_m = wavelength_m
        self.cn2_weights = np.array([cfg["cn2_weight"] for cfg in layers_config])

        self.layers: list[TurbulenceLayer] = []
        for cfg in layers_config:
            layer = TurbulenceLayer(
                N=N,
                pixel_scale=pixel_scale,
                r0=cfg["r0"],
                L0=cfg["L0"],
                altitude=cfg["altitude"],
                wind_speed=cfg["wind_speed"],
                wind_direction=cfg["wind_direction"],
                seed=cfg.get("seed", None),
            )
            self.layers.append(layer)

    def get_integrated_phase(self) -> np.ndarray:
        """
        Return the Cn2-weighted sum of all layer phase screens, scaled
        from radians to a wavefront error in meters via
        (wavelength / (2*pi)).

        Returns
        -------
        wavefront_m : np.ndarray, shape (N, N), meters
        """
        total_phase = np.zeros((self.N, self.N), dtype=float)
        for w, layer in zip(self.cn2_weights, self.layers):
            total_phase += w * layer.phase

        wavefront_m = total_phase * (self.wavelength_m / (2.0 * np.pi))
        return wavefront_m

    def get_integrated_phase_radians(self) -> np.ndarray:
        """Return the Cn2-weighted sum of layer phases in radians."""
        total_phase = np.zeros((self.N, self.N), dtype=float)
        for w, layer in zip(self.cn2_weights, self.layers):
            total_phase += w * layer.phase
        return total_phase

    def evolve(self, dt: float) -> None:
        """Evolve every layer by one timestep dt (seconds)."""
        for layer in self.layers:
            layer.evolve(dt)

    def reset(self) -> None:
        """Reset phase screens to initial state (re-seed from original seed)."""
        for layer in self.layers:
            layer._residual_shift = np.array([0.0, 0.0])
            layer.phase = generate_phase_screen(
                layer._N, layer._pixel_scale, layer._r0, layer._L0,
                model="von_karman", seed=layer._seed,
            )


@dataclass
class AtmosphereParams:
    """Convenience container for atmosphere configuration."""
    r0_m: float
    L0_m: float
    altitudes_m: Sequence[float]
    cn2_weights: Sequence[float]
    wind_speeds_ms: Sequence[float]
    wind_directions_deg: Sequence[float]


def build_atmosphere_from_config(config: dict, N: int, pixel_scale: float, seed: Optional[int] = None) -> MultiLayerAtmosphere:
    """
    Helper to construct a MultiLayerAtmosphere directly from a parsed
    config.yaml dictionary.
    """
    turb = config["turbulence"]
    sim_cfg = config["sim"]
    n_layers = turb["n_layers"]

    layers_config = []
    for i in range(n_layers):
        layer_r0 = turb["r0_m"] / (turb["layer_cn2_weights"][i] ** (3.0 / 5.0))
        layers_config.append(
            {
                "r0": layer_r0,
                "L0": turb["L0_m"],
                "altitude": turb["layer_altitudes_m"][i],
                "wind_speed": turb["layer_wind_speeds_ms"][i],
                "wind_direction": turb["layer_wind_directions_deg"][i],
                "cn2_weight": turb["layer_cn2_weights"][i],
                "seed": None if seed is None else seed + i,
            }
        )

    return MultiLayerAtmosphere(
        layers_config, N=N, pixel_scale=pixel_scale, wavelength_m=sim_cfg["wavelength_m"]
    )