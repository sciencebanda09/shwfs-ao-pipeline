"""
tests/test_lqg.py
==================
Unit tests for control/lqg.py.
"""

import numpy as np
import pytest

from control.lqg import KalmanFilter, LQRController, ActuatorStrokeMinimizer, fit_ar1_from_noll


N_STATE = 5
N_OBS = 5
N_ACT = 8


def _build_kalman():
    A = np.eye(N_STATE) * 0.9
    C = np.eye(N_OBS, N_STATE)
    Q = np.eye(N_STATE) * 1e-3
    R = np.eye(N_OBS) * 1e-2
    P0 = np.eye(N_STATE)
    return KalmanFilter(A, C, Q, R, P0)


def test_kalman_steady_state():
    """Kalman gain should converge to steady-state after ~50 steps."""
    kf = _build_kalman()
    K_ss = kf.steady_state_gain()

    rng = np.random.default_rng(0)
    x_true = np.zeros(N_STATE)

    P_history = []
    for _ in range(50):
        x_true = kf.A @ x_true + rng.normal(0, np.sqrt(1e-3), size=N_STATE)
        y = kf.C @ x_true + rng.normal(0, np.sqrt(1e-2), size=N_OBS)

        kf.predict()
        kf.update(y)
        P_history.append(kf.P_post.copy())

    # P_post should stabilize (small change between last two steps)
    diff = np.abs(P_history[-1] - P_history[-2]).max()
    assert diff < 1e-2

    assert K_ss.shape == (N_STATE, N_OBS)


def test_lqg_beats_integrator():
    """On a 200-frame simulation, mean LQG Strehl should exceed integrator Strehl.

    This is tested using a simplified AR(1) wavefront model and a
    direct Kalman+LQR vs simple-integrator comparison on scalar state
    tracking error, as a proxy for the full closed-loop benchmark
    (which is covered end-to-end in pipeline.run_evaluation).
    """
    n_frames = 200
    dt = 0.001
    a_true = 0.98

    rng = np.random.default_rng(3)

    A = np.array([[a_true]])
    C = np.array([[1.0]])
    Q = np.array([[1e-4]])
    R = np.array([[1e-3]])
    P0 = np.array([[1.0]])

    kf = KalmanFilter(A, C, Q, R, P0)
    B = np.array([[1.0]])
    Q_lqr = np.array([[1.0]])
    R_lqr = np.array([[0.1]])
    lqr = LQRController(A, B, Q_lqr, R_lqr)

    x_true = 0.0
    lqg_errors = []
    integrator_errors = []

    integrator_state = 0.0
    gain = 0.5

    for _ in range(n_frames):
        x_true = a_true * x_true + rng.normal(0, np.sqrt(1e-4))
        y = x_true + rng.normal(0, np.sqrt(1e-3))

        # LQG
        kf.predict()
        x_est = kf.update(np.array([y]))
        u_lqg = lqr.control(x_est)
        residual_lqg = x_true - u_lqg[0]
        lqg_errors.append(residual_lqg ** 2)

        # Simple integrator
        integrator_state += gain * y
        residual_int = x_true - integrator_state
        integrator_errors.append(residual_int ** 2)

    rms_lqg = np.sqrt(np.mean(lqg_errors))
    rms_int = np.sqrt(np.mean(integrator_errors))

    strehl_lqg = np.exp(-rms_lqg ** 2)
    strehl_int = np.exp(-rms_int ** 2)

    assert strehl_lqg >= strehl_int


def test_lqr_gain_shape():
    """LQR gain matrix should have shape (n_actuators, n_zernike)."""
    A = np.eye(N_STATE) * 0.9
    B = np.random.default_rng(0).normal(0, 0.1, size=(N_STATE, N_ACT))
    Q = np.eye(N_STATE)
    R = np.eye(N_ACT) * 0.1

    lqr = LQRController(A, B, Q, R)
    assert lqr.L.shape == (N_ACT, N_STATE)


def test_l1_sparsity():
    """L1 solution should have >=10% fewer nonzero commands than L2 on the same wavefront."""
    rng = np.random.default_rng(0)
    n_pixels = 200
    n_actuators = 30

    influence_matrix = rng.normal(0, 1, size=(n_pixels, n_actuators))
    true_commands = np.zeros(n_actuators)
    true_commands[:5] = rng.normal(0, 1, size=5)  # sparse ground truth

    phase = influence_matrix @ true_commands + rng.normal(0, 0.01, size=n_pixels)

    minimizer = ActuatorStrokeMinimizer()
    commands_l2 = minimizer.solve_l2(phase, influence_matrix)
    commands_l1 = minimizer.solve_l1(phase, influence_matrix, beta=0.5)

    tol = 1e-3
    n_nonzero_l2 = np.sum(np.abs(commands_l2) > tol)
    n_nonzero_l1 = np.sum(np.abs(commands_l1) > tol)

    assert n_nonzero_l1 <= n_nonzero_l2 * 0.9


def test_ar1_fit():
    """AR(1) coefficients fit from data should predict better than persistence baseline."""
    n_zernike = 10
    dt = 0.001
    r0 = 0.15
    wind_speed = 10.0
    D = 0.5

    a = fit_ar1_from_noll(n_zernike, dt, r0, wind_speed, D)

    assert a.shape == (n_zernike,)
    assert np.all((a > 0) & (a < 1))

    # Simulate a process with this AR(1) coefficient and check that
    # x_{t+1} = a*x_t predicts better (lower error) than persistence
    # (x_{t+1} = x_t) when a < 1.
    rng = np.random.default_rng(5)
    j = 0  # first mode
    n_frames = 1000
    x = np.zeros(n_frames)
    for t in range(1, n_frames):
        x[t] = a[j] * x[t - 1] + rng.normal(0, 1)

    ar1_pred = a[j] * x[:-1]
    persistence_pred = x[:-1]
    target = x[1:]

    ar1_error = np.sqrt(np.mean((ar1_pred - target) ** 2))
    persistence_error = np.sqrt(np.mean((persistence_pred - target) ** 2))

    assert ar1_error <= persistence_error
