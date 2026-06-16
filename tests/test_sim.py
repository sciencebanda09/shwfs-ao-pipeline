"""
tests/test_sim.py
==================
Unit tests for sim/turbulence.py, sim/shwfs.py, sim/noise.py.
"""

import numpy as np
import pytest

from sim.turbulence import (
    kolmogorov_psd,
    von_karman_psd,
    generate_phase_screen,
    MultiLayerAtmosphere,
)
from sim.shwfs import SHWFSSensor
from sim.noise import add_photon_noise, centroid_cog


def test_kolmogorov_psd_scaling():
    """PSD should scale as f^(-11/3) at fixed r0."""
    r0 = 0.15
    f1 = np.array([1.0])
    f2 = np.array([2.0])

    psd1 = kolmogorov_psd(f1, r0)[0]
    psd2 = kolmogorov_psd(f2, r0)[0]

    ratio = psd1 / psd2
    expected_ratio = (1.0 / 2.0) ** (-11.0 / 3.0)

    assert ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_von_karman_psd_outer_scale():
    """Von Karman PSD should flatten (approach a finite plateau) below f=1/L0."""
    r0 = 0.15
    L0 = 25.0

    f_low = np.array([1e-6])
    f_at_outer = np.array([1.0 / L0])

    psd_low = von_karman_psd(f_low, r0, L0)[0]
    psd_outer = von_karman_psd(f_at_outer, r0, L0)[0]

    # The PSD at very low frequency should be close to the plateau value
    # (f^2 term negligible compared to (1/L0)^2), not diverging.
    plateau = 0.023 * r0 ** (-5.0 / 3.0) * (1.0 / L0) ** (-11.0 / 3.0)
    assert psd_low == pytest.approx(plateau, rel=1e-2)

    # And should be finite (no f^(-11/3) divergence at f->0)
    assert np.isfinite(psd_low)
    assert psd_low < 1e10


def test_phase_screen_shape():
    """Generated phase screen should have shape (N, N)."""
    N = 256
    screen = generate_phase_screen(N, pixel_scale=0.5 / 128, r0=0.15, L0=25.0, seed=1)
    assert screen.shape == (N, N)


def test_phase_screen_statistics():
    """Structure function D(r) should roughly follow 6.88*(r/r0)^(5/3) for small r."""
    N = 128
    pixel_scale = 0.5 / N
    r0 = 0.15
    screen = generate_phase_screen(N, pixel_scale, r0, L0=25.0, seed=2)

    # Compute structure function for a small separation along x
    r_pixels = 2
    r_m = r_pixels * pixel_scale

    diffs = screen[:, r_pixels:] - screen[:, :-r_pixels]
    D_measured = np.mean(diffs ** 2)

    D_theory = 6.88 * (r_m / r0) ** (5.0 / 3.0)

    # Order-of-magnitude check (phase screens are stochastic; allow
    # a generous factor given finite-grid and subharmonic effects)
    assert D_measured > 0
    assert D_measured < 50 * D_theory + 1e-3


def test_atmosphere_evolution():
    """MultiLayerAtmosphere should evolve without error over 10 frames."""
    N = 64
    pixel_scale = 0.5 / N
    layers_config = [
        {"r0": 0.2, "L0": 25.0, "altitude": 0, "wind_speed": 8.0, "wind_direction": 0, "cn2_weight": 0.6, "seed": 1},
        {"r0": 0.3, "L0": 25.0, "altitude": 5000, "wind_speed": 15.0, "wind_direction": 90, "cn2_weight": 0.4, "seed": 2},
    ]
    atmo = MultiLayerAtmosphere(layers_config, N=N, pixel_scale=pixel_scale)

    for _ in range(10):
        atmo.evolve(0.001)
        phase = atmo.get_integrated_phase_radians()
        assert phase.shape == (N, N)
        assert np.all(np.isfinite(phase))


def test_shwfs_reference_spots():
    """Reference spots should be on a regular grid (constant per subaperture)."""
    sensor = SHWFSSensor(
        n_subapertures=10, pixels_per_subaperture=8, focal_length=0.02, pitch=0.05, wavelength=550e-9
    )
    ref = sensor.generate_reference_spots()
    assert ref.shape == (10, 10, 2)

    # All reference positions should be identical (center of subaperture)
    expected = ref[0, 0]
    assert np.allclose(ref, expected)


def test_shwfs_propagation():
    """Flat wavefront should produce zero slopes everywhere."""
    sensor = SHWFSSensor(
        n_subapertures=10, pixels_per_subaperture=8, focal_length=0.02, pitch=0.05, wavelength=550e-9
    )
    N = 80
    flat_phase = np.zeros((N, N))

    slopes_x, slopes_y = sensor.propagate(flat_phase)

    assert np.allclose(slopes_x, 0.0, atol=1e-10)
    assert np.allclose(slopes_y, 0.0, atol=1e-10)


def test_noise_photon():
    """Poisson noise should preserve the mean intensity (within tolerance)."""
    np.random.seed(0)
    spot = np.ones((8, 8)) / 64.0  # normalized
    flux = 10000

    noisy = add_photon_noise(spot, flux)

    expected_total = flux
    assert noisy.sum() == pytest.approx(expected_total, rel=0.05)


def test_simulate_spot_image_linearity():
    """
    Bug 1 regression: centroid shift must be linearly proportional to tilt
    (R^2 > 0.99) and cross-axis leakage must be negligible.
    """
    from sim.shwfs import _k_shift
    sensor = SHWFSSensor(
        n_subapertures=8, pixels_per_subaperture=16,
        focal_length=1e-3, pitch=0.5e-3, wavelength=500e-9,
    )
    n = sensor.pix_per_sub
    ref_spot = sensor.simulate_spot_image(0.0, 0.0)
    ref_cx, ref_cy = centroid_cog(ref_spot)

    tilts = np.linspace(-0.05, 0.05, 21)
    shifts_x, shifts_y = [], []
    for t in tilts:
        cx, cy = centroid_cog(sensor.simulate_spot_image(t, 0.0))
        shifts_x.append(cx - ref_cx)
        shifts_y.append(cy - ref_cy)

    shifts_x = np.array(shifts_x)
    shifts_y = np.array(shifts_y)

    p = np.polyfit(tilts, shifts_x, 1)
    resid = shifts_x - np.polyval(p, tilts)
    ss_tot = np.sum((shifts_x - shifts_x.mean()) ** 2)
    R2 = 1.0 - resid.var() / (shifts_x.var() + 1e-30)

    # Linearity
    assert R2 > 0.99, f"Centroid shift not linear in tilt: R^2={R2:.4f}"

    # Slope close to K_SHIFT = n/(2*pi)
    K = _k_shift(n)
    assert abs(p[0] - K) < 0.2 * K, f"Slope {p[0]:.4f} far from K_SHIFT={K:.4f}"

    # Cross-axis leakage small
    assert np.abs(shifts_y).max() < 0.05, (
        f"Cross-axis leakage too large: {np.abs(shifts_y).max():.4f} px"
    )


    """A Gaussian spot at a known position should be recovered within 0.1 pixel."""
    n = 16
    x0, y0 = 8.3, 7.6
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x)

    sigma = 1.5
    spot = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    spot += 0.001  # small background

    cx, cy = centroid_cog(spot)

    assert abs(cx - x0) < 0.1
    assert abs(cy - y0) < 0.1
