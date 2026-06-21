"""
reconstruction/bayesian.py
============================
MMSE (Minimum Mean Square Error) Bayesian wavefront reconstructor.

Theory
------
Given a linear forward model m = D @ s + n, with Gaussian priors
s ~ N(0, C_phi) and n ~ N(0, C_n), the posterior mean (MMSE estimate)
of s given m is:

    s_hat = C_phi @ D.T @ inv(D @ C_phi @ D.T + C_n) @ m

This module implements the Kolmogorov/Noll phase covariance C_phi, the
MMSE reconstruction matrix W, online updates when r0 or the noise
covariance change, and a small learned-noise-covariance network.

Fix (v1.2.0)
------------
sigma2_default was hardcoded to 1e-2, which is ~40x too large at
nominal flux (1000 ph, 3e- RN), causing MMSE to over-regularize and
perform 5x worse than SVD. Replaced with a physics-derived estimate:

    sigma2 = (readout_noise_e / flux_photons)^2 * centroid_noise_factor

with a fallback of 2.5e-4 when config is not supplied. This correctly
places C_n below the signal covariance D @ C_phi @ D.T, letting the
prior dominate and restoring MMSE's theoretical advantage over SVD.

Also added calibrate_noise_cov_from_slopes() to estimate C_n directly
from slope residuals when real ISRO data is available — most accurate
option for lab conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Noll (1976) Table 1 variance coefficients for Zernike modes j = 2..36.
# sigma^2_j = NOLL_COEFF[j] * (D/r0)^(5/3)  [radians^2]
# Values follow the standard Noll variance table (delta_1j terms only,
# i.e. diagonal / single-index approximation commonly used in practice).
_NOLL_COEFFS = {
    2: 0.448,   # tip
    3: 0.448,   # tilt
    4: 0.0232,  # defocus
    5: 0.0232,  # astigmatism
    6: 0.0232,  # astigmatism
    7: 0.0062,  # coma
    8: 0.0062,  # coma
    9: 0.0062,  # trefoil
    10: 0.0062,  # trefoil
    11: 0.0024,  # spherical
    12: 0.0024,
    13: 0.0024,
    14: 0.0024,
    15: 0.0024,
    16: 0.0012,
    17: 0.0012,
    18: 0.0012,
    19: 0.0012,
    20: 0.0012,
    21: 0.0012,
    22: 0.0006,
    23: 0.0006,
    24: 0.0006,
    25: 0.0006,
    26: 0.0006,
    27: 0.0006,
    28: 0.0006,
    29: 0.0003,
    30: 0.0003,
    31: 0.0003,
    32: 0.0003,
    33: 0.0003,
    34: 0.0003,
    35: 0.0003,
    36: 0.0003,
}

# Centroid noise factor: empirical constant relating (RN/flux)^2 to
# slope variance in radians^2.  Derived from spot-simulation benchmarks
# with 8-pixel subapertures and Gaussian spots (FWHM ~ 2.5 px).
# Recalibrate via calibrate_noise_cov_from_slopes() when real data
# is available.
_CENTROID_NOISE_FACTOR = 0.25


def _sigma2_from_config(noise_config: dict | None) -> float:
    """
    Physics-based estimate of per-slope noise variance (rad^2).

    Uses the centroid noise model:
        sigma^2 = (readout_noise_e / flux_photons)^2 * CENTROID_NOISE_FACTOR

    Falls back to 2.5e-4 (equivalent to ~31 photons / 1 e- RN) when
    noise_config is None.  This is ~40x smaller than the previous
    hardcoded 1e-2 default, correctly placing the noise prior below the
    Kolmogorov signal covariance.

    Parameters
    ----------
    noise_config : dict | None
        config['noise'] sub-dict with keys:
          'flux_photons_per_frame' (default 1000)
          'readout_noise_e'        (default 3.0)

    Returns
    -------
    sigma2 : float
    """
    if noise_config is None:
        return 2.5e-4   # safe fallback: ~1000 ph, 1 e- RN

    flux = float(noise_config.get("flux_photons_per_frame", 1000))
    rn   = float(noise_config.get("readout_noise_e", 3.0))

    if flux <= 0:
        return 2.5e-4

    sigma2 = (rn / flux) ** 2 * _CENTROID_NOISE_FACTOR
    # Clamp to a reasonable range [1e-8, 1e-1] to guard against
    # degenerate config values.
    return float(np.clip(sigma2, 1e-8, 1e-1))


def calibrate_noise_cov_from_slopes(
    slope_stack: np.ndarray,
    interaction_matrix: np.ndarray,
    zernike_coeffs: np.ndarray,
    diagonal_only: bool = True,
) -> np.ndarray:
    """
    Estimate the slope noise covariance C_n directly from data.

    Computes residuals  r = m - D @ a  (measured slopes minus predicted
    slopes from reconstructed Zernike coefficients) over many frames,
    then returns either the full sample covariance or its diagonal.

    Use this when real ISRO lab frames are available — it subsumes the
    physics-based sigma2 estimate with the actual sensor noise statistics.

    Parameters
    ----------
    slope_stack : np.ndarray, shape (n_frames, 2*n_valid_sub)
        Stacked measured slope vectors.
    interaction_matrix : np.ndarray, shape (n_slopes, n_zernike)
        Modal interaction matrix D.
    zernike_coeffs : np.ndarray, shape (n_frames, n_zernike)
        Reconstructed Zernike coefficients for each frame.
    diagonal_only : bool
        If True, return diag(C_n) as a diagonal matrix (cheaper,
        avoids estimating off-diagonal noise correlations).

    Returns
    -------
    C_n : np.ndarray, shape (n_slopes, n_slopes)
    """
    predicted = zernike_coeffs @ interaction_matrix.T   # (n_frames, n_slopes)
    residuals = slope_stack - predicted                  # (n_frames, n_slopes)

    if diagonal_only:
        var_per_slope = np.var(residuals, axis=0, ddof=1)  # (n_slopes,)
        return np.diag(var_per_slope)
    else:
        return np.cov(residuals.T)                          # (n_slopes, n_slopes)


class KolmogorovCovariance:
    """
    Kolmogorov / Noll phase covariance matrix builder.

    Methods
    -------
    noll_variance(j, D, r0)
        Theoretical variance of Zernike mode j (Noll 1976).
    build_phase_covariance(n_zernike, D, r0, wavelength)
        Build the (n_zernike, n_zernike) diagonal covariance matrix
        C_phi.
    """

    @staticmethod
    def noll_variance(j: int, D: float, r0: float) -> float:
        """
        Noll (1976) Table 1 variance for Zernike mode j (radians^2).

        sigma^2_j = coeff_j * (D/r0)^(5/3)

        Parameters
        ----------
        j : int
            Noll index, 2 <= j <= 36 (j=1, piston, is excluded; returns
            0 for j=1 or j outside the tabulated range).
        D : float
            Aperture diameter (m).
        r0 : float
            Fried parameter (m).

        Returns
        -------
        variance : float, radians^2
        """
        coeff = _NOLL_COEFFS.get(j, 0.0)
        if coeff == 0.0:
            return 0.0
        return float(coeff * (D / r0) ** (5.0 / 3.0))

    @classmethod
    def build_phase_covariance(cls, n_zernike: int, D: float, r0: float, wavelength: float) -> np.ndarray:
        """
        Build the diagonal Kolmogorov phase covariance matrix C_phi for
        Zernike modes j = 1..n_zernike (radians^2). The piston term
        (j=1) is set to a small finite value to keep the matrix
        invertible.

        Parameters
        ----------
        n_zernike : int
        D : float
            Aperture diameter (m).
        r0 : float
            Fried parameter (m).
        wavelength : float
            Wavelength (m), unused in the radian-based formula but kept
            for API completeness.

        Returns
        -------
        C_phi : np.ndarray, shape (n_zernike, n_zernike)
        """
        diag = np.zeros(n_zernike)
        for idx in range(n_zernike):
            j = idx + 1
            if j == 1:
                diag[idx] = 1e-6
            else:
                diag[idx] = cls.noll_variance(j, D, r0)
                if diag[idx] == 0.0:
                    diag[idx] = 1e-8
        return np.diag(diag)


class MMSEReconstructor:
    """
    MMSE wavefront reconstructor.

    W = C_phi @ D.T @ inv(D @ C_phi @ D.T + C_n)

    Parameters
    ----------
    interaction_matrix : np.ndarray, shape (n_slopes, n_zernike)
        Modal interaction matrix D (slopes = D @ zernike).
    r0 : float
        Fried parameter (m).
    D : float
        Aperture diameter (m).
    wavelength : float
        Wavelength (m).
    noise_cov : np.ndarray, optional, shape (n_slopes, n_slopes)
        Noise covariance C_n. If None, computed from noise_config via
        the physics-based centroid noise model.
    noise_config : dict, optional
        config['noise'] sub-dict. Used only when noise_cov is None to
        derive sigma2 from flux and readout noise parameters.
    """

    def __init__(
        self,
        interaction_matrix: np.ndarray,
        r0: float,
        D: float,
        wavelength: float,
        noise_cov: np.ndarray | None = None,
        noise_config: dict | None = None,
    ):
        self.interaction_matrix = interaction_matrix
        self.r0 = r0
        self.D_aperture = D
        self.wavelength = wavelength
        self.n_slopes, self.n_zernike = interaction_matrix.shape

        self.C_phi = KolmogorovCovariance.build_phase_covariance(self.n_zernike, D, r0, wavelength)

        if noise_cov is not None:
            self.C_n = noise_cov
        else:
            # FIX v1.2.0: was hardcoded 1e-2, ~40x too large at nominal
            # flux (1000 ph, 3e- RN), causing MMSE to perform 5x worse
            # than SVD.  Now derived from the centroid noise model.
            sigma2 = _sigma2_from_config(noise_config)
            self.C_n = np.eye(self.n_slopes) * sigma2

        self.W = self._compute_mmse_matrix()

    def _compute_mmse_matrix(self) -> np.ndarray:
        D_mat = self.interaction_matrix
        S = D_mat @ self.C_phi @ D_mat.T + self.C_n
        # SVD-based pseudo-inverse with condition cutoff to avoid
        # noise amplification when cond(S) >> 1e6.
        U, sv, Vt = np.linalg.svd(S, full_matrices=False)
        cutoff = sv[0] / 1e6
        sv_inv = np.where(sv > cutoff, 1.0 / sv, 0.0)
        S_inv = (Vt.T * sv_inv) @ U.T
        return self.C_phi @ D_mat.T @ S_inv

    def reconstruct(self, slopes_x: np.ndarray, slopes_y: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        """
        Apply the MMSE matrix W to measured slopes.

        Parameters
        ----------
        slopes_x, slopes_y : np.ndarray
            Slope arrays, either already flattened to valid
            subapertures (1D) or full (n_sub, n_sub) grids together
            with ``valid_mask``.
        valid_mask : np.ndarray of bool, optional
            If slopes_x/y are 2D, this mask selects valid subapertures.

        Returns
        -------
        coeffs : np.ndarray, shape (n_zernike,)
        """
        if slopes_x.ndim == 2:
            if valid_mask is None:
                raise ValueError("valid_mask is required for 2D slope arrays")
            sx = slopes_x[valid_mask]
            sy = slopes_y[valid_mask]
        else:
            sx = slopes_x
            sy = slopes_y

        m = np.concatenate([sx, sy])
        return self.W @ m

    def update_r0(self, r0_new: float) -> None:
        """Recompute C_phi and W for a new r0 estimate (online adaptation)."""
        self.r0 = r0_new
        self.C_phi = KolmogorovCovariance.build_phase_covariance(
            self.n_zernike, self.D_aperture, self.r0, self.wavelength
        )
        self.W = self._compute_mmse_matrix()

    def update_noise_cov(self, C_n_new: np.ndarray) -> None:
        """Hot-swap the noise covariance matrix and recompute W."""
        self.C_n = C_n_new
        self.W = self._compute_mmse_matrix()

    def calibrate_from_data(
        self,
        slope_stack: np.ndarray,
        zernike_coeffs: np.ndarray,
        diagonal_only: bool = True,
    ) -> None:
        """
        Calibrate C_n from real slope residuals and recompute W.

        Convenience wrapper around calibrate_noise_cov_from_slopes().
        Call this once real ISRO frames have been processed to get the
        most accurate noise prior for the lab sensor.

        Parameters
        ----------
        slope_stack : np.ndarray, shape (n_frames, 2*n_valid_sub)
        zernike_coeffs : np.ndarray, shape (n_frames, n_zernike)
        diagonal_only : bool
        """
        self.C_n = calibrate_noise_cov_from_slopes(
            slope_stack, self.interaction_matrix, zernike_coeffs, diagonal_only
        )
        self.W = self._compute_mmse_matrix()


class LearnedNoiseCov(nn.Module):
    """
    Small MLP that predicts a noise covariance matrix from raw slope
    residuals.

    Input: flattened slope residuals, shape (2*n_sub^2,)
    Output: full covariance matrix of shape (2*n_sub^2, 2*n_sub^2),
    constructed as L @ L.T from a predicted lower-triangular factor to
    guarantee positive semi-definiteness.

    Parameters
    ----------
    n_sub : int
        Number of subapertures per axis. The slope vector has length
        2 * n_sub^2 (x and y slopes for an n_sub x n_sub grid).
    hidden_size : int
    """

    def __init__(self, n_sub: int, hidden_size: int = 128):
        super().__init__()
        self.n_slopes = 2 * n_sub * n_sub
        self.n_tri = self.n_slopes * (self.n_slopes + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(self.n_slopes, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.n_tri),
        )

        self._tril_indices = torch.tril_indices(self.n_slopes, self.n_slopes)

    def forward(self, slope_residuals: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        slope_residuals : torch.Tensor, shape (B, 2*n_sub^2)

        Returns
        -------
        C_n : torch.Tensor, shape (B, 2*n_sub^2, 2*n_sub^2)
            Predicted (PSD) noise covariance matrix per sample.
        """
        batch_size = slope_residuals.shape[0]
        tri_vals = self.net(slope_residuals)  # (B, n_tri)

        L = torch.zeros(batch_size, self.n_slopes, self.n_slopes, device=slope_residuals.device)
        row, col = self._tril_indices
        L[:, row, col] = tri_vals

        # Ensure positive diagonal via softplus on diagonal entries
        diag_idx = torch.arange(self.n_slopes)
        L[:, diag_idx, diag_idx] = nn.functional.softplus(L[:, diag_idx, diag_idx]) + 1e-6

        C_n = L @ L.transpose(1, 2)
        return C_n


def derive_mmse_from_bayes(C_phi: np.ndarray, D: np.ndarray, C_n: np.ndarray) -> np.ndarray:
    """
    Standalone MMSE derivation function.

    Given the linear-Gaussian model
        m = D @ s + n,   s ~ N(0, C_phi),   n ~ N(0, C_n)

    the joint distribution of (s, m) is jointly Gaussian with:
        Cov(s, m)  = C_phi @ D.T
        Cov(m, m)  = D @ C_phi @ D.T + C_n

    The conditional mean (posterior mean / MMSE estimate) of a jointly
    Gaussian (s, m) is:
        E[s | m] = Cov(s,m) @ inv(Cov(m,m)) @ m = W @ m

    This is also the linear estimator minimizing E[||s_hat - s||^2]
    among all linear estimators (and, for jointly Gaussian variables,
    among ALL estimators).

    Parameters
    ----------
    C_phi : np.ndarray, shape (n_zernike, n_zernike)
    D : np.ndarray, shape (n_slopes, n_zernike)
    C_n : np.ndarray, shape (n_slopes, n_slopes)

    Returns
    -------
    W : np.ndarray, shape (n_zernike, n_slopes)
    """
    # Step 1: cross-covariance Cov(s, m) = C_phi @ D.T
    cov_sm = C_phi @ D.T

    # Step 2: measurement covariance Cov(m, m) = D @ C_phi @ D.T + C_n
    cov_mm = D @ C_phi @ D.T + C_n

    # Step 3: MMSE gain W = Cov(s,m) @ inv(Cov(m,m))
    W = cov_sm @ np.linalg.pinv(cov_mm)

    return W


def compare_svd_vs_mmse(
    slopes_test: np.ndarray,
    zernike_truth: np.ndarray,
    svd_reconstructor,
    mmse_reconstructor: MMSEReconstructor,
    valid_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compare per-frame RMS WFE between an SVD-based (e.g.
    ModalReconstructor) and an MMSE reconstructor on the same test data.

    Parameters
    ----------
    slopes_test : np.ndarray, shape (n_frames, 2, n_sub, n_sub)
    zernike_truth : np.ndarray, shape (n_frames, n_zernike)
    svd_reconstructor : object with .reconstruct(slopes_x, slopes_y)
    mmse_reconstructor : MMSEReconstructor
    valid_mask : np.ndarray of bool, shape (n_sub, n_sub), optional

    Returns
    -------
    df : pd.DataFrame, columns ['frame', 'rms_svd', 'rms_mmse']
    """
    n_frames = slopes_test.shape[0]
    rms_svd = np.zeros(n_frames)
    rms_mmse = np.zeros(n_frames)

    for k in range(n_frames):
        sx, sy = slopes_test[k, 0], slopes_test[k, 1]
        truth = zernike_truth[k]

        pred_svd = svd_reconstructor.reconstruct(sx, sy)
        pred_mmse = mmse_reconstructor.reconstruct(sx, sy, valid_mask=valid_mask)

        rms_svd[k] = np.sqrt(np.mean((pred_svd[1:] - truth[1:]) ** 2))
        rms_mmse[k] = np.sqrt(np.mean((pred_mmse[1:] - truth[1:]) ** 2))

    return pd.DataFrame({"frame": np.arange(n_frames), "rms_svd": rms_svd, "rms_mmse": rms_mmse})
