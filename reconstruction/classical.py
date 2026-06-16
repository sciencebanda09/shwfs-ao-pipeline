"""
reconstruction/classical.py
============================
Classical wavefront reconstruction: zonal (actuator-space) and modal
(Zernike-space) reconstructors via SVD pseudo-inverse.
"""

from __future__ import annotations

import numpy as np
import time

from reconstruction.zernike import zernike_basis, noll_to_zernike, zernike_polynomial


def _pinv_svd(M: np.ndarray, condition_number: float) -> np.ndarray:
    """SVD pseudo-inverse with a condition-number cutoff on singular values."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    if S.size == 0 or S[0] == 0:
        return np.linalg.pinv(M)
    cutoff = S[0] / condition_number
    S_inv = np.where(S > cutoff, 1.0 / S, 0.0)
    return (Vt.T * S_inv) @ U.T


class ZonalReconstructor:
    """
    Zonal wavefront reconstructor mapping SH-WFS slopes directly to DM
    actuator commands via an interaction matrix built from Gaussian
    influence functions (Fried geometry).

    Parameters
    ----------
    sensor_geometry : SHWFSSensor
        Provides subaperture positions and valid mask.
    svd_condition_number : float
        Condition-number cutoff for the pseudo-inverse.
    actuator_positions : tuple[np.ndarray, np.ndarray], optional
        (x, y) actuator positions in normalized aperture units. If not
        given, a default 97-actuator hexagonal grid is generated.
    coupling : float
        Influence-function coupling parameter (default 0.3).
    """

    def __init__(
        self,
        sensor_geometry,
        svd_condition_number: float = 50.0,
        actuator_positions: tuple[np.ndarray, np.ndarray] | None = None,
        coupling: float = 0.3,
    ):
        from actuator.geometry import hexagonal_actuator_positions

        self.sensor = sensor_geometry
        self.svd_condition_number = svd_condition_number
        self.coupling = coupling

        if actuator_positions is None:
            self.act_x, self.act_y = hexagonal_actuator_positions(11, coupling)
        else:
            self.act_x, self.act_y = actuator_positions

        self.n_actuators = self.act_x.shape[0]
        self.interaction_matrix = self.build_interaction_matrix()
        self.command_matrix = _pinv_svd(self.interaction_matrix, self.svd_condition_number)

    def build_interaction_matrix(self) -> np.ndarray:
        """
        Build the interaction matrix D of shape (2*n_valid_sub,
        n_actuators), where each column is the slope response (x and y)
        of all valid subapertures to a unit poke of one actuator's
        Gaussian influence function.
        """
        sub_x, sub_y = self.sensor.get_subaperture_positions()
        n_valid = sub_x.shape[0]
        D = np.zeros((2 * n_valid, self.n_actuators))

        # Pitch in normalized units between adjacent actuators
        pitch = 2.0 / 11.0

        eps = 1e-4
        for k in range(self.n_actuators):
            x0, y0 = self.act_x[k], self.act_y[k]
            # Analytic gradient of Gaussian influence function
            # IF = exp(ln_c * r2 / pitch^2)
            r2 = (sub_x - x0) ** 2 + (sub_y - y0) ** 2
            ln_c = np.log(self.coupling)
            base = np.exp(ln_c * r2 / pitch ** 2)
            dphi_dx = base * (2.0 * ln_c * (sub_x - x0) / pitch ** 2)
            dphi_dy = base * (2.0 * ln_c * (sub_y - y0) / pitch ** 2)
            D[:n_valid, k] = dphi_dx
            D[n_valid:, k] = dphi_dy

        return D

    def reconstruct(self, slopes_x: np.ndarray, slopes_y: np.ndarray) -> np.ndarray:
        """
        Reconstruct actuator commands from slope measurements.

        Parameters
        ----------
        slopes_x, slopes_y : np.ndarray, shape (n_sub, n_sub)

        Returns
        -------
        commands : np.ndarray, shape (n_actuators,)
        """
        valid = self.sensor.get_valid_subaperture_mask()
        sx = slopes_x[valid]
        sy = slopes_y[valid]
        s = np.concatenate([sx, sy])
        return self.command_matrix @ s


class ModalReconstructor:
    """
    Modal (Zernike-space) wavefront reconstructor.

    Builds an interaction matrix mapping Zernike mode coefficients to
    SH-WFS slope responses (via finite-difference gradients of each
    mode), then inverts it via SVD pseudo-inverse to recover modal
    coefficients from measured slopes.

    Parameters
    ----------
    sensor : SHWFSSensor
    zernike_basis : np.ndarray, shape (n_modes, N, N)
        Precomputed Zernike basis on an N x N grid matching the
        simulation grid size.
    n_modes : int
    svd_condition_number : float
    """

    def __init__(self, sensor, zernike_basis: np.ndarray, n_modes: int, svd_condition_number: float = 50.0):
        self.sensor = sensor
        self.zernike_basis = zernike_basis
        self.n_modes = n_modes
        self.svd_condition_number = svd_condition_number

        self.modal_matrix = self.build_modal_matrix()
        self.reconstruction_matrix = _pinv_svd(self.modal_matrix, self.svd_condition_number)
        self._last_reconstruct_ms: float = 0.0

    def build_modal_matrix(self) -> np.ndarray:
        """
        Compute the slope response (x and y) of each Zernike mode at
        every valid subaperture, assembling a (2*n_valid_sub, n_modes)
        matrix.
        """
        n_sub = self.sensor.n_sub
        valid = self.sensor.get_valid_subaperture_mask()
        n_valid = int(valid.sum())

        N = self.zernike_basis.shape[1]
        tile = N // n_sub

        M = np.zeros((2 * n_valid, self.n_modes))

        for m in range(self.n_modes):
            phase = self.zernike_basis[m]
            grad_y, grad_x = np.gradient(phase)

            sx_list, sy_list = [], []
            for i in range(n_sub):
                for j in range(n_sub):
                    if not valid[i, j]:
                        continue
                    y0, y1 = i * tile, (i + 1) * tile
                    x0, x1 = j * tile, (j + 1) * tile
                    sx_list.append(np.mean(grad_x[y0:y1, x0:x1]))
                    sy_list.append(np.mean(grad_y[y0:y1, x0:x1]))

            M[:n_valid, m] = sx_list
            M[n_valid:, m] = sy_list

        return M

    def reconstruct(self, slopes_x: np.ndarray, slopes_y: np.ndarray) -> np.ndarray:
        """
        Reconstruct Zernike modal coefficients from slope measurements.

        Parameters
        ----------
        slopes_x, slopes_y : np.ndarray, shape (n_sub, n_sub)

        Returns
        -------
        coeffs : np.ndarray, shape (n_modes,)
        """
        valid = self.sensor.get_valid_subaperture_mask()
        sx = slopes_x[valid]
        sy = slopes_y[valid]
        s = np.concatenate([sx, sy])
        t0 = time.perf_counter()
        result = self.reconstruction_matrix @ s
        self._last_reconstruct_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def get_reconstructed_phase(self, coeffs: np.ndarray, zernike_basis: np.ndarray) -> np.ndarray:
        """
        Reconstruct a phase map from modal coefficients:
        phase = sum_j coeffs[j] * basis[j]

        Parameters
        ----------
        coeffs : np.ndarray, shape (n_modes,)
        zernike_basis : np.ndarray, shape (n_modes, N, N)

        Returns
        -------
        phase : np.ndarray, shape (N, N)
        """
        return np.tensordot(coeffs, zernike_basis, axes=(0, 0))
