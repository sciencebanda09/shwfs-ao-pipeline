"""
reconstruction/zernike.py
==========================
Zernike polynomial basis generation, Noll indexing, and least-squares
phase fitting utilities used throughout the AO pipeline.
"""

from __future__ import annotations

import numpy as np
from math import factorial


def noll_to_zernike(j: int) -> tuple[int, int]:
    """
    Convert a 1-indexed Noll index ``j`` to the (n, m) radial / azimuthal
    Zernike orders using the standard Noll (1976) ordering table.

    Parameters
    ----------
    j : int
        Noll index, j >= 1 (j=1 is piston).

    Returns
    -------
    (n, m) : tuple[int, int]
        Radial order n >= 0 and signed azimuthal frequency m.
        m > 0 -> cosine term, m < 0 -> sine term, m == 0 -> rotationally
        symmetric term.
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")

    # Find radial order n: the smallest n such that j <= (n+1)(n+2)/2
    n = 0
    while j > (n + 1) * (n + 2) // 2:
        n += 1

    # Position within row n (1-indexed)
    p = j - n * (n + 1) // 2

    # Possible |m| values for this n, in increasing order: m takes
    # values 0 or 1, ..., up to n, stepping by 2.
    if n % 2 == 0:
        m_values = list(range(0, n + 1, 2))  # 0, 2, 4, ..., n
    else:
        m_values = list(range(1, n + 1, 2))  # 1, 3, 5, ..., n

    if m_values[0] == 0:
        # n even: order is m=0, then |m|=2 (pair), |m|=4 (pair), ...
        if p == 1:
            return n, 0
        idx = p - 2
        pair = idx // 2
        m_abs = m_values[1 + pair]
    else:
        # n odd: |m|=1 (pair), |m|=3 (pair), ...
        idx = p - 1
        pair = idx // 2
        m_abs = m_values[pair]

    # Standard Noll sign convention: even j -> m = +|m| (cosine term),
    # odd j -> m = -|m| (sine term).
    m = m_abs if (j % 2 == 0) else -m_abs
    return n, m


def _radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Evaluate the radial Zernike polynomial R_n^|m|(rho)."""
    m = abs(m)
    R = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        c = (
            (-1) ** k
            * factorial(n - k)
            / (
                factorial(k)
                * factorial((n + m) // 2 - k)
                * factorial((n - m) // 2 - k)
            )
        )
        R += c * rho ** (n - 2 * k)
    return R


def zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Evaluate a single Zernike polynomial Z_n^m at polar coordinates
    (rho, theta), rho in [0, 1].

    Uses the standard normalization such that each mode has unit RMS
    over the unit disk (ANSI / Noll normalization).
    """
    R = _radial_polynomial(n, m, rho)

    if m == 0:
        norm = np.sqrt(n + 1)
        Z = norm * R
    elif m > 0:
        norm = np.sqrt(2 * (n + 1))
        Z = norm * R * np.cos(m * theta)
    else:
        norm = np.sqrt(2 * (n + 1))
        Z = norm * R * np.sin(abs(m) * theta)

    return Z


def zernike_basis(n_terms: int, N: int, normalize: bool = True) -> np.ndarray:
    """
    Generate an (n_terms, N, N) array containing the first ``n_terms``
    Zernike polynomials (Noll indices 1..n_terms) evaluated on an N x N
    grid covering the unit disk, masked to a circular aperture.

    Parameters
    ----------
    n_terms : int
        Number of Zernike modes (Noll indices 1..n_terms).
    N : int
        Grid size (N x N).
    normalize : bool
        If True, rescale each mode so it has unit RMS over the aperture
        (accounts for discretization effects on top of the analytic
        normalization).

    Returns
    -------
    basis : np.ndarray, shape (n_terms, N, N)
    """
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    mask = rho <= 1.0

    basis = np.zeros((n_terms, N, N), dtype=float)
    for idx in range(n_terms):
        j = idx + 1
        n, m = noll_to_zernike(j)
        Z = zernike_polynomial(n, m, rho, theta)
        Z = Z * mask
        if normalize:
            rms = np.sqrt(np.mean(Z[mask] ** 2)) if mask.any() else 1.0
            if rms > 1e-12:
                Z = Z / rms
        basis[idx] = Z

    return basis


def zernike_matrix(n_terms: int, rho_flat: np.ndarray, theta_flat: np.ndarray) -> np.ndarray:
    """
    Build the Zernike design matrix Z of shape (n_pixels, n_terms) for a
    flattened set of polar coordinates (rho_flat, theta_flat).

    Z[:, idx] = Z_{Noll index idx+1}(rho_flat, theta_flat)
    """
    n_pixels = rho_flat.shape[0]
    Z = np.zeros((n_pixels, n_terms), dtype=float)
    for idx in range(n_terms):
        j = idx + 1
        n, m = noll_to_zernike(j)
        Z[:, idx] = zernike_polynomial(n, m, rho_flat, theta_flat)
    return Z


def fit_zernike(
    phase_flat: np.ndarray,
    rho_flat: np.ndarray,
    theta_flat: np.ndarray,
    n_terms: int,
    mask_flat: np.ndarray,
) -> np.ndarray:
    """
    Least-squares fit of Zernike coefficients to a flattened phase map.

    coeffs = pinv(Z[mask]) @ phase_flat[mask]

    Parameters
    ----------
    phase_flat : np.ndarray, shape (n_pixels,)
    rho_flat, theta_flat : np.ndarray, shape (n_pixels,)
    n_terms : int
    mask_flat : np.ndarray of bool, shape (n_pixels,)

    Returns
    -------
    coeffs : np.ndarray, shape (n_terms,)
    """
    Z = zernike_matrix(n_terms, rho_flat[mask_flat], theta_flat[mask_flat])
    coeffs = np.linalg.pinv(Z) @ phase_flat[mask_flat]
    return coeffs
