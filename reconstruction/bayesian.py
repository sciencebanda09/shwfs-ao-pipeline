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
        Noise covariance C_n. If None, uses identity scaled by a
        default readout-noise variance.
    """

    def __init__(self, interaction_matrix: np.ndarray, r0: float, D: float, wavelength: float, noise_cov: np.ndarray | None = None):
        self.interaction_matrix = interaction_matrix
        self.r0 = r0
        self.D_aperture = D
        self.wavelength = wavelength
        self.n_slopes, self.n_zernike = interaction_matrix.shape

        self.C_phi = KolmogorovCovariance.build_phase_covariance(self.n_zernike, D, r0, wavelength)

        if noise_cov is None:
            sigma2_default = 1e-2  # FIX: was 1e-4, too small vs slope variance
            self.C_n = np.eye(self.n_slopes) * sigma2_default
        else:
            self.C_n = noise_cov

        self.W = self._compute_mmse_matrix()

    def _compute_mmse_matrix(self) -> np.ndarray:
        D_mat = self.interaction_matrix
        S = D_mat @ self.C_phi @ D_mat.T + self.C_n
        # FIX: use SVD-based pseudo-inverse with condition cutoff (was plain pinv).
        # Plain pinv on a badly scaled S amplifies noise when cond(S) >> 1e6.
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

        rms_svd[k] = np.sqrt(np.mean((pred_svd - truth) ** 2))
        rms_mmse[k] = np.sqrt(np.mean((pred_mmse - truth) ** 2))

    return pd.DataFrame({"frame": np.arange(n_frames), "rms_svd": rms_svd, "rms_mmse": rms_mmse})
