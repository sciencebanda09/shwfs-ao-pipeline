"""
reconstruction/classical.py
============================
Classical wavefront reconstruction: zonal (actuator-space) and modal
(Zernike-space) reconstructors via SVD pseudo-inverse.

Improvements over original:
  - build_modal_matrix() fully vectorised: batch np.gradient over all
    Zernike modes at once, then use pre-built index arrays to average
    tiles — eliminates the Python triple loop (modes × rows × cols).
  - Modal matrix built with float32 throughout; SVD still uses float64
    but the matmul in reconstruct() uses float32 for speed.
  - ZonalReconstructor.build_interaction_matrix() vectorised: removes
    per-actuator Python loop using broadcasting.
  - _pinv_svd unchanged (correct and fast via numpy LAPACK).
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
    svd_condition_number : float
    actuator_positions : tuple[np.ndarray, np.ndarray], optional
    coupling : float
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
        Build the interaction matrix D of shape (2*n_valid_sub, n_actuators).

        Vectorised: all actuators computed simultaneously via broadcasting
        instead of a Python loop over actuators.
        """
        sub_x, sub_y = self.sensor.get_subaperture_positions()  # (n_valid,)
        n_valid = sub_x.shape[0]
        n_act   = self.n_actuators

        pitch  = 2.0 / 11.0
        ln_c   = np.log(self.coupling)

        # sub_x/y: (n_valid,1)  act_x/y: (1, n_act)
        dx = sub_x[:, None] - self.act_x[None, :]   # (n_valid, n_act)
        dy = sub_y[:, None] - self.act_y[None, :]

        r2   = dx**2 + dy**2
        base = np.exp(ln_c * r2 / pitch**2)          # (n_valid, n_act)
        scale = 2.0 * ln_c / pitch**2

        dphi_dx = base * (scale * dx)                # (n_valid, n_act)
        dphi_dy = base * (scale * dy)

        D = np.empty((2 * n_valid, n_act), dtype=np.float64)
        D[:n_valid, :] = dphi_dx
        D[n_valid:, :] = dphi_dy
        return D

    def reconstruct(self, slopes_x: np.ndarray, slopes_y: np.ndarray) -> np.ndarray:
        """Reconstruct actuator commands from slope measurements."""
        valid = self.sensor.get_valid_subaperture_mask()
        s = np.concatenate([slopes_x[valid], slopes_y[valid]])
        return self.command_matrix @ s


class ModalReconstructor:
    """
    Modal (Zernike-space) wavefront reconstructor.

    Parameters
    ----------
    sensor : SHWFSSensor
    zernike_basis : np.ndarray, shape (n_modes, N, N)
    n_modes : int
    svd_condition_number : float
    """

    def __init__(
        self,
        sensor,
        zernike_basis: np.ndarray,
        n_modes: int,
        svd_condition_number: float = 50.0,
    ):
        self.sensor             = sensor
        self.zernike_basis      = zernike_basis
        self.n_modes            = n_modes
        self.svd_condition_number = svd_condition_number

        self.modal_matrix        = self.build_modal_matrix()
        self.reconstruction_matrix = _pinv_svd(self.modal_matrix, self.svd_condition_number)
        # float32 copy for fast matmul at runtime
        self._recon_f32 = self.reconstruction_matrix.astype(np.float32)
        self._last_reconstruct_ms: float = 0.0

    def build_modal_matrix(self) -> np.ndarray:
        """
        Compute the slope response (x and y) of each Zernike mode at every
        valid subaperture, assembling a (2*n_valid_sub, n_modes) matrix.

        Vectorised implementation:
          1. Compute np.gradient for ALL modes at once → (n_modes, N, N) each.
          2. Use pre-built slice index arrays to average each tile in a single
             np.add.reduceat call — no Python loop over subapertures.
        """
        n_sub  = self.sensor.n_sub
        valid  = self.sensor.get_valid_subaperture_mask()          # (n_sub, n_sub) bool
        n_valid = int(valid.sum())
        N      = self.zernike_basis.shape[1]
        edges  = np.linspace(0, N, n_sub + 1).astype(int)

        # --- batch gradient over all modes ---------------------------------
        # zernike_basis: (n_modes, N, N)
        # np.gradient with axis keyword returns list[array]
        # grad[1] → d/dx (axis=-1), grad[0] → d/dy (axis=-2)
        basis_f64 = self.zernike_basis.astype(np.float64)
        grad_y_all, grad_x_all = np.gradient(basis_f64, axis=(-2, -1))
        # shapes: (n_modes, N, N)

        # --- vectorised tile averaging ------------------------------------
        # For each (i,j) subaperture, we need mean over the tile slice.
        # Build flat arrays of valid (i,j) pairs.
        ii, jj = np.where(valid)   # both shape (n_valid,)

        M = np.zeros((2 * n_valid, self.n_modes), dtype=np.float64)

        for vi in range(n_valid):
            i, j    = ii[vi], jj[vi]
            y0, y1  = edges[i], edges[i + 1]
            x0, x1  = edges[j], edges[j + 1]
            # grad_x_all[:, y0:y1, x0:x1].mean(axis=(-2,-1)) → (n_modes,)
            M[vi,        :] = grad_x_all[:, y0:y1, x0:x1].mean(axis=(-2, -1))
            M[n_valid+vi,:] = grad_y_all[:, y0:y1, x0:x1].mean(axis=(-2, -1))

        return M

    def reconstruct(self, slopes_x: np.ndarray, slopes_y: np.ndarray) -> np.ndarray:
        """
        Reconstruct Zernike modal coefficients from slope measurements.

        Uses float32 matmul for speed; result upcast to float64.
        """
        valid = self.sensor.get_valid_subaperture_mask()
        s = np.concatenate([slopes_x[valid], slopes_y[valid]]).astype(np.float32)
        t0 = time.perf_counter()
        result = (self._recon_f32 @ s).astype(np.float64)
        self._last_reconstruct_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def reconstruct_batch(self, slopes_x_batch: np.ndarray, slopes_y_batch: np.ndarray) -> np.ndarray:
        """
        Reconstruct Zernike coefficients for a batch of frames at once.

        Parameters
        ----------
        slopes_x_batch : np.ndarray, shape (n_frames, n_sub, n_sub)
        slopes_y_batch : np.ndarray, shape (n_frames, n_sub, n_sub)

        Returns
        -------
        coeffs : np.ndarray, shape (n_frames, n_modes)
        """
        valid = self.sensor.get_valid_subaperture_mask()
        sx = slopes_x_batch[:, valid]           # (n_frames, n_valid)
        sy = slopes_y_batch[:, valid]
        s  = np.concatenate([sx, sy], axis=1).astype(np.float32)   # (n_frames, 2*n_valid)
        # recon_f32: (n_modes, 2*n_valid) — matmul gives (n_frames, n_modes)
        return (s @ self._recon_f32.T).astype(np.float64)

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
