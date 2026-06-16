"""
tests/test_actuator.py
========================
Unit tests for actuator/geometry.py, influence_fn.py, dm_command.py.
"""

import numpy as np
import pytest

from actuator.geometry import hexagonal_actuator_positions
from actuator.influence_fn import gaussian_influence_function, build_influence_matrix
from actuator.dm_command import DMController
from sim.phase_screen import get_aperture_mask


def test_hexagonal_positions():
    """97-actuator hexagonal grid should lie within the unit circle."""
    x, y = hexagonal_actuator_positions(11, coupling=0.3)

    r2 = x ** 2 + y ** 2
    assert np.all(r2 <= 1.0 + 1e-6)
    assert x.shape[0] > 50  # roughly pi/4 * 121 ~ 95


def test_influence_function_normalization():
    """Influence function should equal 1.0 at the actuator center."""
    x0, y0 = 0.0, 0.0
    coupling = 0.3
    pitch = 2.0 / 11.0

    val_center = gaussian_influence_function(np.array([x0]), np.array([y0]), x0, y0, coupling, pitch)
    assert val_center[0] == pytest.approx(1.0)


def test_influence_matrix_shape():
    """Influence matrix should have shape (N*N, n_actuators)."""
    N = 32
    x, y = hexagonal_actuator_positions(11, coupling=0.3)
    pitch = 2.0 / 11.0

    M = build_influence_matrix((x, y), N, pixel_scale=1.0, coupling=0.3, pitch=pitch)

    assert M.shape == (N * N, x.shape[0])


def test_dm_flat_command():
    """Zero wavefront error should produce near-zero actuator commands."""
    config = {
        "actuator": {"n_actuators": 97, "geometry": "hexagonal", "coupling": 0.3, "stroke_limit_um": 5.0},
        "sim": {"grid_size": 32, "wavelength_m": 550e-9},
    }
    dm = DMController(config)
    N = config["sim"]["grid_size"]
    mask = get_aperture_mask(N, "circular")

    flat_phase = np.zeros((N, N))
    commands = dm.wavefront_to_commands(flat_phase, mask)

    assert np.allclose(commands, 0.0, atol=1e-12)


def test_stroke_limit_clipping():
    """Commands should be clipped to the configured stroke limit."""
    config = {
        "actuator": {"n_actuators": 97, "geometry": "hexagonal", "coupling": 0.3, "stroke_limit_um": 1.0},
        "sim": {"grid_size": 32, "wavelength_m": 550e-9},
    }
    dm = DMController(config)
    N = config["sim"]["grid_size"]
    mask = get_aperture_mask(N, "circular")

    # Large wavefront error to force clipping
    large_phase = np.ones((N, N)) * 1.0  # meters (unrealistically large, for test)
    commands = dm.wavefront_to_commands(large_phase, mask)

    stroke_limit_m = config["actuator"]["stroke_limit_um"] * 1e-6
    assert np.all(np.abs(commands) <= stroke_limit_m + 1e-15)
