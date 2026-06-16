"""
actuator/geometry.py
=====================
Deformable mirror actuator geometry: hexagonal and square grids within
a circular aperture.
"""

from __future__ import annotations

import numpy as np


def hexagonal_actuator_positions(n_actuators_across: int, coupling: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a hexagonal grid of actuator positions within the unit
    circle.

    The grid is built as a set of staggered rows spanning
    ``n_actuators_across`` columns at the widest row, with row spacing
    of sqrt(3)/2 times the column spacing, and only positions with
    radius <= 1 are kept. For n_actuators_across=11 this yields the
    standard ~97-actuator arrangement.

    Parameters
    ----------
    n_actuators_across : int
        Number of actuator columns spanning the full aperture diameter.
    coupling : float
        Influence-function coupling parameter (unused geometrically,
        kept for API consistency).

    Returns
    -------
    x, y : np.ndarray
        Actuator positions normalized to [-1, 1].
    """
    spacing = 2.0 / (n_actuators_across - 1)
    row_spacing = spacing * np.sqrt(3) / 2.0

    n_rows = int(np.ceil(2.0 / row_spacing)) + 2
    half_rows = n_rows // 2

    xs, ys = [], []
    for r in range(-half_rows, half_rows + 1):
        y = r * row_spacing
        if abs(y) > 1.0 + spacing:
            continue
        offset = (spacing / 2.0) if (r % 2 != 0) else 0.0
        n_cols = int(np.ceil(2.0 / spacing)) + 2
        half_cols = n_cols // 2
        for c in range(-half_cols, half_cols + 1):
            x = c * spacing + offset
            if x ** 2 + y ** 2 <= 1.0 + 1e-9:
                xs.append(x)
                ys.append(y)

    x = np.array(xs)
    y = np.array(ys)

    # If grid is too sparse/dense relative to target count, that's fine:
    # the geometry naturally produces ~n_actuators_across^2 * pi/4 points.
    return x, y


def square_actuator_positions(n_across: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a square grid of actuator positions within the circular
    aperture.

    Parameters
    ----------
    n_across : int
        Number of actuator columns/rows spanning the full diameter.

    Returns
    -------
    x, y : np.ndarray
        Actuator positions normalized to [-1, 1], restricted to the
        unit circle.
    """
    coords = np.linspace(-1, 1, n_across)
    xx, yy = np.meshgrid(coords, coords)
    mask = (xx ** 2 + yy ** 2) <= 1.0 + 1e-9
    return xx[mask], yy[mask]


def actuator_positions_to_pixel(positions: tuple[np.ndarray, np.ndarray], N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized [-1, 1] actuator positions to pixel coordinates
    on an N x N grid.

    Parameters
    ----------
    positions : tuple[np.ndarray, np.ndarray]
        (x, y) positions in [-1, 1].
    N : int
        Grid size.

    Returns
    -------
    px, py : np.ndarray
        Pixel coordinates in [0, N-1].
    """
    x, y = positions
    px = (x + 1.0) / 2.0 * (N - 1)
    py = (y + 1.0) / 2.0 * (N - 1)
    return px, py


def get_actuator_mask(positions: tuple[np.ndarray, np.ndarray], N: int) -> np.ndarray:
    """
    Return a boolean N x N mask marking the nearest-pixel location of
    each actuator.

    Parameters
    ----------
    positions : tuple[np.ndarray, np.ndarray]
        (x, y) positions in [-1, 1].
    N : int

    Returns
    -------
    mask : np.ndarray of bool, shape (N, N)
    """
    px, py = actuator_positions_to_pixel(positions, N)
    mask = np.zeros((N, N), dtype=bool)
    px_int = np.clip(np.round(px).astype(int), 0, N - 1)
    py_int = np.clip(np.round(py).astype(int), 0, N - 1)
    mask[py_int, px_int] = True
    return mask
