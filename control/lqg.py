"""
control/lqg.py
===============
Linear Quadratic Gaussian (LQG) AO controller: Kalman filter state
estimation + LQR optimal control, AR(1) turbulence state modeling, and
L1/L2 actuator stroke minimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
from sklearn.linear_model import Lasso

from profiling.temporal_psd import compute_tau0_per_mode


class KalmanFilter:
    """
    Discrete-time Kalman filter.

    State model:   x_{t+1} = A x_t + w_t,   w_t ~ N(0, Q)
    Observation:   y_t = C x_t + v_t,        v_t ~ N(0, R)

    Parameters
    ----------
    A : np.ndarray, shape (n_state, n_state)
    C : np.ndarray, shape (n_obs, n_state)
    Q : np.ndarray, shape (n_state, n_state)
    R : np.ndarray, shape (n_obs, n_obs)
    P0 : np.ndarray, shape (n_state, n_state)
        Initial state covariance.
    """

    def __init__(self, A: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, P0: np.ndarray):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R

        n_state = A.shape[0]
        self.x_post = np.zeros(n_state)
        self.P_post = P0.copy()

        self.x_prior = self.x_post.copy()
        self.P_prior = self.P_post.copy()

    def predict(self) -> np.ndarray:
        """
        Prediction step:
            x_prior = A @ x_post
            P_prior = A @ P_post @ A.T + Q

        Returns
        -------
        x_prior : np.ndarray, shape (n_state,)
        """
        self.x_prior = self.A @ self.x_post
        self.P_prior = self.A @ self.P_post @ self.A.T + self.Q
        return self.x_prior

    def update(self, y: np.ndarray) -> np.ndarray:
        """
        Update (correction) step:
            K = P_prior @ C.T @ inv(C @ P_prior @ C.T + R)
            x_post = x_prior + K @ (y - C @ x_prior)
            P_post = (I - K @ C) @ P_prior

        Parameters
        ----------
        y : np.ndarray, shape (n_obs,)
            Measurement vector.

        Returns
        -------
        x_post : np.ndarray, shape (n_state,)
        """
        S = self.C @ self.P_prior @ self.C.T + self.R
        K = self.P_prior @ self.C.T @ np.linalg.pinv(S)

        innovation = y - self.C @ self.x_prior
        self.x_post = self.x_prior + K @ innovation

        I = np.eye(self.A.shape[0])
        self.P_post = (I - K @ self.C) @ self.P_prior

        return self.x_post

    def steady_state_gain(self) -> np.ndarray:
        """
        Solve the discrete algebraic Riccati equation (DARE) for the
        steady-state error covariance and return the corresponding
        steady-state Kalman gain.

        Returns
        -------
        K_ss : np.ndarray, shape (n_state, n_obs)
        """
        # DARE for the filter: P = A P A.T - A P C.T inv(C P C.T + R) C P A.T + Q
        # solve_discrete_are(A, B, Q, R) solves:
        #   A.T X A - X - (A.T X B)(R + B.T X B)^-1 (B.T X A) + Q = 0
        # For the filter Riccati equation we use the dual form with
        # A -> A.T, B -> C.T.
        P_ss = solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
        S = self.C @ P_ss @ self.C.T + self.R
        K_ss = P_ss @ self.C.T @ np.linalg.pinv(S)
        return K_ss

    def reset(self) -> None:
        """Reset the filter state to zero."""
        n_state = self.A.shape[0]
        self.x_post = np.zeros(n_state)
        self.x_prior = np.zeros(n_state)


class LQRController:
    """
    Discrete-time Linear Quadratic Regulator.

    Cost: J = sum_t (x_t.T @ Q @ x_t + u_t.T @ R @ u_t)
    Dynamics: x_{t+1} = A x_t + B u_t

    Parameters
    ----------
    A : np.ndarray, shape (n_state, n_state)
    B : np.ndarray, shape (n_state, n_actuators)
    Q : np.ndarray, shape (n_state, n_state)
    R : np.ndarray, shape (n_actuators, n_actuators)
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.L = self.compute_gain()

    def compute_gain(self) -> np.ndarray:
        """
        Solve the discrete algebraic Riccati equation and return the
        LQR gain matrix L such that u = -L @ x.

        Returns
        -------
        L : np.ndarray, shape (n_actuators, n_state)
        """
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        S = self.R + self.B.T @ P @ self.B
        L = np.linalg.pinv(S) @ (self.B.T @ P @ self.A)
        return L

    def control(self, x_estimated: np.ndarray) -> np.ndarray:
        """
        Compute the optimal control input u = -L @ x_estimated.

        Parameters
        ----------
        x_estimated : np.ndarray, shape (n_state,)

        Returns
        -------
        u : np.ndarray, shape (n_actuators,)
        """
        return -self.L @ x_estimated


class LQGController:
    """
    Full LQG controller combining a Kalman filter state estimator with
    an LQR controller, operating in Zernike-coefficient state space.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    interaction_matrix : np.ndarray, shape (n_slopes, n_zernike)
        Modal interaction matrix C (slopes = C @ zernike).
    actuator_influence : np.ndarray, shape (n_zernike, n_actuators)
        Projection from actuator commands to Zernike-space effect
        (B matrix).
    dt : float
        Control loop timestep (s).
    """

    def __init__(self, config: dict, interaction_matrix: np.ndarray, actuator_influence: np.ndarray, dt: float):
        self.config = config
        self.C = interaction_matrix
        self.B = actuator_influence
        self.dt = dt

        n_zernike = self.C.shape[1]
        n_slopes = self.C.shape[0]
        n_actuators = self.B.shape[1]

        lqg_cfg = config["lqg"]

        # Default AR(1) model and process noise; will be replaced by
        # fit_state_model if training data is provided.
        self.A = np.eye(n_zernike) * 0.95
        self.Q = np.eye(n_zernike) * lqg_cfg["process_noise_q"]
        self.R_meas = np.eye(n_slopes) * lqg_cfg["measurement_noise_r"]
        P0 = np.eye(n_zernike)

        self.kf = KalmanFilter(self.A, self.C, self.Q, self.R_meas, P0)

        Q_lqr = np.eye(n_zernike) * lqg_cfg["lqr_state_weight"]
        R_lqr = np.eye(n_actuators) * lqg_cfg["lqr_control_weight"]
        self.lqr = LQRController(self.A, self.B, Q_lqr, R_lqr)

    def fit_state_model(self, zernike_sequence: np.ndarray) -> None:
        """
        Fit the AR(1) state-transition matrix A (Zernike-to-Zernike)
        from a training sequence via least squares, and estimate the
        process noise covariance Q from the fit residuals. Updates the
        internal Kalman filter and LQR controller with the new A.

        Parameters
        ----------
        zernike_sequence : np.ndarray, shape (n_frames, n_zernike)
        """
        X = zernike_sequence[:-1]  # (n-1, n_zernike)
        Y = zernike_sequence[1:]   # (n-1, n_zernike)

        # Per-mode AR(1): y_j = a_j * x_j  ->  a_j = sum(x_j*y_j)/sum(x_j^2)
        n_modes = zernike_sequence.shape[1]
        a_diag = np.zeros(n_modes)
        residual_var = np.zeros(n_modes)

        for j in range(n_modes):
            xj = X[:, j]
            yj = Y[:, j]
            denom = np.sum(xj ** 2)
            if denom > 1e-12:
                a_j = np.sum(xj * yj) / denom
            else:
                a_j = 0.0
            a_j = np.clip(a_j, -0.999, 0.999)
            a_diag[j] = a_j
            residual_var[j] = np.var(yj - a_j * xj)

        self.A = np.diag(a_diag)
        self.Q = np.diag(np.maximum(residual_var, 1e-10))

        n_slopes = self.C.shape[0]
        n_actuators = self.B.shape[1]
        lqg_cfg = self.config["lqg"]

        P0 = np.eye(n_modes)
        self.kf = KalmanFilter(self.A, self.C, self.Q, self.R_meas, P0)

        Q_lqr = np.eye(n_modes) * lqg_cfg["lqr_state_weight"]
        R_lqr = np.eye(n_actuators) * lqg_cfg["lqr_control_weight"]
        self.lqr = LQRController(self.A, self.B, Q_lqr, R_lqr)

    def step(self, slopes: np.ndarray) -> np.ndarray:
        """
        One control cycle: Kalman predict -> Kalman update(slopes) ->
        LQR control(x_est).

        Parameters
        ----------
        slopes : np.ndarray, shape (n_slopes,)
            Flattened (valid-subaperture) slope measurement vector.

        Returns
        -------
        commands : np.ndarray, shape (n_actuators,)
        """
        self.kf.predict()
        x_est = self.kf.update(slopes)
        commands = self.lqr.control(x_est)
        return commands

    def reset(self) -> None:
        """Reset the Kalman filter state to zero."""
        self.kf.reset()


class ActuatorStrokeMinimizer:
    """
    Actuator stroke minimization via L1 (LASSO) vs standard L2
    least-squares solutions.
    """

    def solve_l1(self, residual_phase: np.ndarray, influence_matrix: np.ndarray, beta: float = 0.1) -> np.ndarray:
        """
        Solve commands = argmin_c ||M @ c - phi||^2 + beta * ||c||_1
        using sklearn's Lasso.

        Parameters
        ----------
        residual_phase : np.ndarray, shape (n_pixels,)
            Flattened residual phase to be corrected.
        influence_matrix : np.ndarray, shape (n_pixels, n_actuators)
        beta : float
            L1 regularization strength.

        Returns
        -------
        commands : np.ndarray, shape (n_actuators,)
            Sparse actuator commands.
        """
        lasso = Lasso(alpha=beta, fit_intercept=False, max_iter=5000)
        lasso.fit(influence_matrix, residual_phase)
        return lasso.coef_

    def solve_l2(self, residual_phase: np.ndarray, influence_matrix: np.ndarray) -> np.ndarray:
        """
        Standard L2 least-squares solution via pseudo-inverse.

        Parameters
        ----------
        residual_phase : np.ndarray, shape (n_pixels,)
        influence_matrix : np.ndarray, shape (n_pixels, n_actuators)

        Returns
        -------
        commands : np.ndarray, shape (n_actuators,)
        """
        return np.linalg.pinv(influence_matrix) @ residual_phase

    def compare_l2_vs_l1(self, phase_map: np.ndarray, mask: np.ndarray, influence_matrix: np.ndarray, beta: float = 0.1) -> dict:
        """
        Run both L2 and L1 solutions for a given residual phase map and
        compare actuator sparsity.

        Parameters
        ----------
        phase_map : np.ndarray, shape (N, N)
        mask : np.ndarray of bool, shape (N, N)
        influence_matrix : np.ndarray, shape (N*N, n_actuators)
        beta : float

        Returns
        -------
        result : dict with keys
            'commands_l2', 'commands_l1', 'sparsity_l2', 'sparsity_l1'
        """
        flat_phase = (phase_map * mask).flatten()

        commands_l2 = self.solve_l2(flat_phase, influence_matrix)
        commands_l1 = self.solve_l1(flat_phase, influence_matrix, beta=beta)

        tol = 1e-6 * np.max(np.abs(commands_l2)) if np.max(np.abs(commands_l2)) > 0 else 1e-9
        sparsity_l2 = float(np.mean(np.abs(commands_l2) < tol))
        sparsity_l1 = float(np.mean(np.abs(commands_l1) < tol))

        return {
            "commands_l2": commands_l2,
            "commands_l1": commands_l1,
            "sparsity_l2": sparsity_l2,
            "sparsity_l1": sparsity_l1,
        }


def fit_ar1_from_noll(n_zernike: int, dt: float, r0: float, wind_speed: float, D: float) -> np.ndarray:
    """
    Compute analytical AR(1) coefficients a_j = exp(-dt / tau_0(j)) from
    a frozen-flow model, using mode-dependent coherence times.

    Since per-mode tau_0 requires actual time-series data (see
    compute_tau0_per_mode in profiling/temporal_psd.py), this function
    provides an analytical approximation: tau_0(j) ~ r0 / (v * sqrt(n_j)),
    where n_j is the radial order of mode j (higher-order modes
    decorrelate faster), and v = wind_speed.

    Parameters
    ----------
    n_zernike : int
    dt : float
    r0 : float
        Fried parameter (m).
    wind_speed : float
        Wind speed (m/s).
    D : float
        Aperture diameter (m), unused directly but kept for API
        completeness.

    Returns
    -------
    a : np.ndarray, shape (n_zernike,)
        AR(1) coefficients per Zernike mode (Noll order 1..n_zernike).
    """
    from reconstruction.zernike import noll_to_zernike

    a = np.zeros(n_zernike)
    v = max(wind_speed, 1e-3)

    for idx in range(n_zernike):
        j = idx + 1
        n, _ = noll_to_zernike(j)
        n_eff = max(n, 1)
        tau0 = r0 / (v * np.sqrt(n_eff))
        tau0 = max(tau0, 1e-4)
        a[idx] = np.exp(-dt / tau0)

    return a


def compare_controllers(atmosphere, sensor, dm, reconstructor, config: dict, n_frames: int) -> pd.DataFrame:
    """
    Run three controllers (integrator, LQG, LQG+prediction) on the same
    turbulence sequence and return their Strehl ratio and RMS WFE per
    frame.

    Parameters
    ----------
    atmosphere : MultiLayerAtmosphere
    sensor : SHWFSSensor
    dm : DMController
    reconstructor : ModalReconstructor (or compatible)
    config : dict
    n_frames : int

    Returns
    -------
    df : pd.DataFrame, columns
        ['frame', 'controller', 'strehl', 'rms_wfe_nm']
    """
    from sim.phase_screen import apply_aperture_mask, get_aperture_mask, compute_rms_wavefront_error, compute_strehl_ratio
    from reconstruction.zernike import zernike_basis as zb_func

    sim_cfg = config["sim"]
    N = sim_cfg["grid_size"]
    n_zernike = sim_cfg["n_zernike"]
    dt = sim_cfg["dt_s"]
    wavelength = sim_cfg["wavelength_m"]

    mask = get_aperture_mask(N, "circular")
    basis = zb_func(n_zernike, N)

    n_actuators = dm.n_actuators
    # Build real influence matrix from Gaussian coupling model (Fried geometry)
    from reconstruction.classical import DMInfluenceReconstructor
    from actuator.geometry import hexagonal_actuator_positions
    _act_x, _act_y = hexagonal_actuator_positions(11, coupling=0.3)
    _infl_rec = DMInfluenceReconstructor(sensor, coupling=0.3, actuator_positions=(_act_x, _act_y))
    # D shape: (2*n_sub, n_actuators) � project to Zernike space via modal matrix
    # modal_matrix: (n_zernike, 2*n_sub); B = modal_matrix @ pinv(D) -> (n_zernike, n_actuators)
    import numpy.linalg as _nla
    actuator_influence = reconstructor.modal_matrix @ _nla.pinv(_infl_rec.interaction_matrix)

    assert actuator_influence.shape == (n_zernike, n_actuators), f"influence shape mismatch: {actuator_influence.shape}"
    assert actuator_influence.shape == (n_zernike, n_actuators), f"influence shape mismatch: {actuator_influence.shape}"
    lqg = LQGController(config, reconstructor.modal_matrix, actuator_influence, dt)
    lqg_pred = LQGController(config, reconstructor.modal_matrix, actuator_influence, dt)

    # FIX: fit AR(1) state model from a warm-up sequence BEFORE the eval loop.
    # Without this, A stays at 0.95*I which is wrong for multi-layer turbulence.
    _warmup_frames = min(50, n_frames)
    _warmup_zernike = np.zeros((_warmup_frames, n_zernike))
    for _wk in range(_warmup_frames):
        atmosphere.evolve(dt)
        _ph = atmosphere.get_integrated_phase_radians()
        from sim.phase_screen import apply_aperture_mask
        _ph_masked = apply_aperture_mask(_ph, "circular")
        _sx, _sy = sensor.propagate(_ph_masked)
        _z = reconstructor.reconstruct(_sx, _sy)
        _warmup_zernike[_wk] = _z
    lqg.fit_state_model(_warmup_zernike)
    lqg_pred.fit_state_model(_warmup_zernike)
    # Reset atmosphere and DM after warm-up so eval starts fresh
    atmosphere.reset()
    dm.reset_integrator()
    lqg.reset()
    lqg_pred.reset()

    dm.reset_integrator()

    results = []

    for k in range(n_frames):
        atmosphere.evolve(dt)
        phase_rad = atmosphere.get_integrated_phase_radians()
        phase_masked = apply_aperture_mask(phase_rad, "circular")
        phase_m = phase_masked * (wavelength / (2 * np.pi))

        sx, sy = sensor.propagate(phase_masked)
        valid = sensor.get_valid_subaperture_mask()
        slopes_vec = np.concatenate([sx[valid], sy[valid]])

        # --- Integrator ---
        _, residual_int = dm.closed_loop_step(phase_m, get_aperture_mask(N, "circular"), gain=0.5)
        rms_int = compute_rms_wavefront_error(phase_m, phase_m - residual_int + residual_int, mask)
        rms_int_rad = float(np.sqrt(np.mean(residual_int[mask] ** 2)) / (wavelength / (2 * np.pi)))
        strehl_int = compute_strehl_ratio(rms_int_rad)
        results.append({"frame": k, "controller": "integrator", "strehl": strehl_int,
                         "rms_wfe_nm": rms_int_rad * (wavelength / (2 * np.pi)) * 1e9})

        # --- LQG ---
        commands_lqg = lqg.step(slopes_vec)
        zernike_est = lqg.kf.x_post
        phase_corr = np.tensordot(zernike_est, basis, axes=(0, 0)) * (wavelength / (2 * np.pi))
        residual_lqg = (phase_m - phase_corr) * mask
        rms_lqg_rad = float(np.sqrt(np.mean(residual_lqg[mask] ** 2)) / (wavelength / (2 * np.pi)))
        strehl_lqg = compute_strehl_ratio(rms_lqg_rad)
        results.append({"frame": k, "controller": "lqg", "strehl": strehl_lqg,
                         "rms_wfe_nm": rms_lqg_rad * (wavelength / (2 * np.pi)) * 1e9})

        # --- LQG + prediction (predict next state via A before applying) ---
        lqg_pred.kf.predict()
        x_est_pred = lqg_pred.kf.x_prior
        lqg_pred.kf.update(slopes_vec)
        x_pred_next = lqg_pred.A @ lqg_pred.kf.x_post
        phase_corr_pred = np.tensordot(x_pred_next, basis, axes=(0, 0)) * (wavelength / (2 * np.pi))
        residual_pred = (phase_m - phase_corr_pred) * mask
        rms_pred_rad = float(np.sqrt(np.mean(residual_pred[mask] ** 2)) / (wavelength / (2 * np.pi)))
        strehl_pred = compute_strehl_ratio(rms_pred_rad)
        results.append({"frame": k, "controller": "lqg_predictive", "strehl": strehl_pred,
                         "rms_wfe_nm": rms_pred_rad * (wavelength / (2 * np.pi)) * 1e9})

    return pd.DataFrame(results)



