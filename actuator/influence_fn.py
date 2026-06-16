"""
actuator/influence_fn.py
=========================
Deformable mirror actuator influence functions (Gaussian model) and
the resulting influence/command matrices.
"""

from __future__ import annotations

import numpy as np

from sim.phase_screen import get_aperture_mask


def gaussian_influence_function(
    x: np.ndarray, y: np.ndarray, x0: float, y0: float, coupling: float, pitch: float
) -> np.ndarray:
    """
    Gaussian actuator influence function:

    IF(x,y) = exp(ln(coupling) * ((x-x0)^2 + (y-y0)^2) / pitch^2)

    Since 0 < coupling < 1, ln(coupling) < 0, so the function decays
    smoothly from 1.0 at the actuator center to ``coupling`` at one
    actuator pitch away.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate grids (normalized aperture units).
    x0, y0 : float
        Actuator center position.
    coupling : float
        Influence-function coupling at one actuator pitch (typically
        ~0.3 -> IF(pitch) = 0.3).
    pitch : float
        Actuator pitch (same units as x, y).

    Returns
    -------
    influence : np.ndarray, same shape as x
    """
    ln_c = np.log(coupling)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return np.exp(ln_c * r2 / pitch ** 2)


def build_influence_matrix(
    actuator_positions: tuple[np.ndarray, np.ndarray],
    N: int,
    pixel_scale: float,
    coupling: float,
    pitch: float,
) -> np.ndarray:
    """
    Construct the full influence matrix M of shape (N*N, n_actuators),
    where column k is the flattened Gaussian influence function of
    actuator k, masked to the circular aperture.

    Parameters
    ----------
    actuator_positions : tuple[np.ndarray, np.ndarray]
        (x, y) actuator positions in normalized [-1, 1] units.
    N : int
        Grid size.
    pixel_scale : float
        Unused directly (coordinates are normalized), kept for API
        consistency.
    coupling : float
        Influence function coupling parameter.
    pitch : float
        Actuator pitch in normalized units.

    Returns
    -------
    M : np.ndarray, shape (N*N, n_actuators)
    """
    x_pos, y_pos = actuator_positions
    n_act = x_pos.shape[0]

    coords = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(coords, coords)
    mask = get_aperture_mask(N, "circular")

    M = np.zeros((N * N, n_act))
    for k in range(n_act):
        infl = gaussian_influence_function(xx, yy, x_pos[k], y_pos[k], coupling, pitch)
        infl = infl * mask
        M[:, k] = infl.flatten()

    return M


def compute_command_matrix(influence_matrix: np.ndarray, svd_cutoff: float = 50.0) -> np.ndarray:
    """
    Compute the command matrix (pseudo-inverse of the influence matrix)
    via SVD with a condition-number cutoff.

    Parameters
    ----------
    influence_matrix : np.ndarray, shape (n_pixels, n_actuators)
    svd_cutoff : float
        Condition-number cutoff for singular value inversion.

    Returns
    -------
    command_matrix : np.ndarray, shape (n_actuators, n_pixels)
    """
    U, S, Vt = np.linalg.svd(influence_matrix, full_matrices=False)
    if S.size == 0 or S[0] == 0:
        return np.linalg.pinv(influence_matrix)
    cutoff = S[0] / svd_cutoff
    S_inv = np.where(S > cutoff, 1.0 / S, 0.0)
    return (Vt.T * S_inv) @ U.T
