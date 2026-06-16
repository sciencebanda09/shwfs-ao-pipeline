"""
tests/test_bayesian.py
========================
Unit tests for reconstruction/bayesian.py (MMSE reconstructor).
"""

import numpy as np
import torch
import pytest

from reconstruction.bayesian import (
    KolmogorovCovariance,
    MMSEReconstructor,
    LearnedNoiseCov,
)
from reconstruction.classical import ModalReconstructor
from reconstruction.zernike import zernike_basis
from sim.shwfs import SHWFSSensor
from sim.noise import apply_sensor_noise


N_ZERNIKE = 36
GRID_SIZE = 64
D_APERTURE = 0.5


def _build_sensor_and_recon():
    sensor = SHWFSSensor(
        n_subapertures=10, pixels_per_subaperture=8, focal_length=0.02, pitch=0.05, wavelength=550e-9
    )
    basis = zernike_basis(N_ZERNIKE, GRID_SIZE, normalize=True)
    modal_recon = ModalReconstructor(sensor, basis, N_ZERNIKE, svd_condition_number=50)
    return sensor, basis, modal_recon


def test_mmse_beats_svd_low_snr():
    """At low flux (high noise), MMSE RMS WFE should be lower than SVD RMS WFE."""
    sensor, basis, modal_recon = _build_sensor_and_recon()
    r0 = 0.15

    mmse_recon = MMSEReconstructor(
        modal_recon.modal_matrix, r0=r0, D=D_APERTURE, wavelength=550e-9,
        noise_cov=np.eye(modal_recon.modal_matrix.shape[0]) * 1e-2,  # high noise
    )

    rng = np.random.default_rng(0)
    n_frames = 50
    valid = sensor.get_valid_subaperture_mask()

    rms_svd_list = []
    rms_mmse_list = []

    for k in range(n_frames):
        true_coeffs = np.zeros(N_ZERNIKE)
        for idx in range(1, N_ZERNIKE):
            j = idx + 1
            var_j = KolmogorovCovariance.noll_variance(j, D_APERTURE, r0)
            if var_j > 0:
                true_coeffs[idx] = rng.normal(0, np.sqrt(var_j))

        clean_slopes = modal_recon.modal_matrix @ true_coeffs
        n_slopes = clean_slopes.shape[0]
        noisy_slopes = clean_slopes + rng.normal(0, 0.1, size=n_slopes)  # high noise

        n_valid = int(valid.sum())
        sx = noisy_slopes[:n_valid]
        sy = noisy_slopes[n_valid:]

        pred_svd = modal_recon.reconstruct(
            np.zeros_like(valid, dtype=float) * 0,  # placeholder, unused
            np.zeros_like(valid, dtype=float) * 0,
        ) if False else None  # avoid unused-call; reconstruct directly below

        # SVD reconstruction directly from the flattened slope vector
        pred_svd = modal_recon.reconstruction_matrix @ noisy_slopes
        pred_mmse = mmse_recon.W @ noisy_slopes

        rms_svd_list.append(np.sqrt(np.mean((pred_svd - true_coeffs) ** 2)))
        rms_mmse_list.append(np.sqrt(np.mean((pred_mmse - true_coeffs) ** 2)))

    assert np.mean(rms_mmse_list) < np.mean(rms_svd_list)


def test_kolmogorov_covariance_diagonal():
    """C_phi diagonal entries should match Noll Table 1 within 5%."""
    r0 = 0.15
    C_phi = KolmogorovCovariance.build_phase_covariance(N_ZERNIKE, D_APERTURE, r0, 550e-9)

    for idx in range(1, N_ZERNIKE):
        j = idx + 1
        expected = KolmogorovCovariance.noll_variance(j, D_APERTURE, r0)
        if expected == 0:
            continue
        assert C_phi[idx, idx] == pytest.approx(expected, rel=0.05)


def test_mmse_reconstructor_flat():
    """A flat (zero) wavefront should produce near-zero MMSE coefficients."""
    sensor, basis, modal_recon = _build_sensor_and_recon()
    mmse_recon = MMSEReconstructor(modal_recon.modal_matrix, r0=0.15, D=D_APERTURE, wavelength=550e-9)

    n_slopes = modal_recon.modal_matrix.shape[0]
    zero_slopes = np.zeros(n_slopes)

    coeffs = mmse_recon.W @ zero_slopes
    assert np.allclose(coeffs, 0.0, atol=1e-12)


def test_noise_cov_update():
    """update_noise_cov should change the W matrix."""
    sensor, basis, modal_recon = _build_sensor_and_recon()
    mmse_recon = MMSEReconstructor(modal_recon.modal_matrix, r0=0.15, D=D_APERTURE, wavelength=550e-9)

    W_before = mmse_recon.W.copy()

    n_slopes = modal_recon.modal_matrix.shape[0]
    new_cov = np.eye(n_slopes) * 10.0
    mmse_recon.update_noise_cov(new_cov)

    W_after = mmse_recon.W

    assert not np.allclose(W_before, W_after)


def test_learned_noise_cov_output_shape():
    """LearnedNoiseCov forward pass should return (B, n_slopes, n_slopes) matrix."""
    n_sub = 10
    n_slopes = 2 * n_sub * n_sub

    model = LearnedNoiseCov(n_sub=n_sub, hidden_size=64)
    x = torch.randn(3, n_slopes)

    C_n = model(x)
    assert C_n.shape == (3, n_slopes, n_slopes)

    # Check symmetry (PSD via L @ L.T is symmetric)
    for b in range(3):
        assert torch.allclose(C_n[b], C_n[b].T, atol=1e-5)
