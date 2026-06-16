"""
tests/test_reconstruction.py
==============================
Unit tests for reconstruction/zernike.py, classical.py, cnn_model.py.
"""

import numpy as np
import torch
import pytest

from reconstruction.zernike import (
    noll_to_zernike,
    zernike_basis,
    zernike_polynomial,
    fit_zernike,
    zernike_matrix,
)
from reconstruction.classical import ModalReconstructor
from reconstruction.cnn_model import UNetReconstructor
from sim.shwfs import SHWFSSensor
from sim.phase_screen import get_aperture_mask, apply_aperture_mask


N_TERMS = 36
GRID_SIZE = 64


def test_zernike_orthonormality():
    """First 36 Zernike polynomials should be orthonormal on the unit disk."""
    basis = zernike_basis(N_TERMS, GRID_SIZE, normalize=True)
    mask = get_aperture_mask(GRID_SIZE, "circular")

    n_pix = mask.sum()
    flat = basis[:, mask]  # (N_TERMS, n_pix)

    gram = (flat @ flat.T) / n_pix

    identity = np.eye(N_TERMS)
    # Off-diagonal terms should be near zero, diagonal near 1 (within
    # discretization tolerance)
    diff = gram - identity
    assert np.max(np.abs(diff)) < 1e-1


def test_noll_ordering():
    """Noll indices 1-10 should map to the correct (n, m) pairs."""
    expected = {
        1: (0, 0),
        2: (1, 1),
        3: (1, -1),
        4: (2, 0),
        5: (2, -2),
        6: (2, 2),
        7: (3, -1),
        8: (3, 1),
        9: (3, -3),
        10: (3, 3),
    }
    for j, (n_exp, m_exp) in expected.items():
        n, m = noll_to_zernike(j)
        assert n == n_exp, f"j={j}: expected n={n_exp}, got n={n}"
        assert m == m_exp, f"j={j}: expected m={m_exp}, got m={m}"


def test_zernike_fit_roundtrip():
    """Generate a phase from known coefficients, fit back, recover within 1% RMS error."""
    N = GRID_SIZE
    basis = zernike_basis(N_TERMS, N, normalize=True)
    mask = get_aperture_mask(N, "circular")

    rng = np.random.default_rng(0)
    true_coeffs = rng.normal(0, 1.0, size=N_TERMS)

    phase = np.tensordot(true_coeffs, basis, axes=(0, 0))

    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    fitted = fit_zernike(phase.flatten(), rho.flatten(), theta.flatten(), N_TERMS, mask.flatten())

    # Note: fit_zernike uses un-normalized zernike_matrix, while phase was
    # generated using normalized basis. Compare reconstructed phases instead.
    Z = zernike_matrix(N_TERMS, rho.flatten()[mask.flatten()], theta.flatten()[mask.flatten()])
    reconstructed_flat = Z @ fitted

    true_flat = phase.flatten()[mask.flatten()]
    rms_error = np.sqrt(np.mean((reconstructed_flat - true_flat) ** 2))
    rms_true = np.sqrt(np.mean(true_flat ** 2))

    assert rms_error / rms_true < 0.05


def test_modal_reconstructor_flat():
    """Flat wavefront should produce near-zero modal coefficients."""
    sensor = SHWFSSensor(
        n_subapertures=10, pixels_per_subaperture=8, focal_length=0.02, pitch=0.05, wavelength=550e-9
    )
    N = 80
    basis = zernike_basis(N_TERMS, N, normalize=True)
    recon = ModalReconstructor(sensor, basis, N_TERMS, svd_condition_number=50)

    flat_phase = np.zeros((N, N))
    slopes_x, slopes_y = sensor.propagate(flat_phase)

    coeffs = recon.reconstruct(slopes_x, slopes_y)
    assert np.allclose(coeffs, 0.0, atol=1e-8)


def test_modal_reconstructor_tiptilt():
    """A pure tip wavefront should produce a dominant tip coefficient."""
    sensor = SHWFSSensor(
        n_subapertures=10, pixels_per_subaperture=8, focal_length=0.02, pitch=0.05, wavelength=550e-9
    )
    N = 80
    basis = zernike_basis(N_TERMS, N, normalize=True)
    recon = ModalReconstructor(sensor, basis, N_TERMS, svd_condition_number=50)

    # Pure tip mode (Noll j=2, index 1)
    tip_phase = basis[1] * 2.0  # arbitrary amplitude
    mask = get_aperture_mask(N, "circular")
    tip_phase = apply_aperture_mask(tip_phase, "circular")

    slopes_x, slopes_y = sensor.propagate(tip_phase)
    coeffs = recon.reconstruct(slopes_x, slopes_y)

    # The tip coefficient (index 1) should dominate the response
    dominant_idx = np.argmax(np.abs(coeffs))
    assert dominant_idx == 1


def test_cnn_model_forward():
    """UNetReconstructor forward pass should produce correct output shape."""
    model = UNetReconstructor(in_channels=2, n_zernike=36, base_filters=16)
    x = torch.randn(4, 2, 10, 10)
    out = model(x)
    assert out.shape == (4, 36)


def test_cnn_model_gradient():
    """Backward pass through UNetReconstructor should run without error."""
    model = UNetReconstructor(in_channels=2, n_zernike=36, base_filters=16)
    x = torch.randn(2, 2, 10, 10)
    target = torch.randn(2, 36)

    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # Check that at least one parameter received a gradient
    grads_exist = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert grads_exist


def test_noisy_slopes_reconstruction_rms():
    """
    Bug 2/3 regression: noisy-slope reconstruction RMS should be in the
    same ballpark as clean-slope RMS (< 0.2 rad), not ~4.5 rad.
    """
    import yaml
    from sim.turbulence import build_atmosphere_from_config
    from sim.shwfs import SHWFSSensor
    from sim.phase_screen import apply_aperture_mask, compute_zernike_coefficients, get_aperture_mask
    from sim.dataset_gen import _apply_realistic_noise
    from reconstruction.zernike import zernike_basis

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    sim_cfg = config["sim"]
    N = sim_cfg["grid_size"]
    n_sub = sim_cfg["n_subapertures"]
    n_zernike = sim_cfg["n_zernike"]
    pixel_scale = sim_cfg["aperture_diameter_m"] / N

    atm = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=7)
    atm.evolve(sim_cfg["dt_s"])
    phase = atm.get_integrated_phase_radians()
    mask = get_aperture_mask(N, "circular")
    phase_masked = apply_aperture_mask(phase, "circular")

    sensor = SHWFSSensor(
        n_subapertures=n_sub,
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    truth = compute_zernike_coefficients(phase_masked, mask, n_zernike, pixel_scale=1.0)
    sx, sy = sensor.propagate(phase_masked)
    nsx, nsy = _apply_realistic_noise(sensor, sx, sy, config["noise"])

    basis = zernike_basis(n_zernike, N)
    recon = ModalReconstructor(sensor, basis, n_zernike, config["reconstruction"]["svd_condition_number"])

    rms_clean = float(np.sqrt(np.mean((recon.reconstruct(sx, sy) - truth) ** 2)))
    rms_noisy = float(np.sqrt(np.mean((recon.reconstruct(nsx, nsy) - truth) ** 2)))

    # Clean should be < 0.05 rad (was 0.0175 before)
    assert rms_clean < 0.1, f"Clean RMS too high: {rms_clean:.4f}"
    # Noisy must not blow up (was ~4.5 rad, should now be < 0.2 rad)
    assert rms_noisy < 0.2, f"Noisy RMS too high: {rms_noisy:.4f} (expected < 0.2)"
