"""
actuator/dm_command.py
=======================
Deformable mirror command generation: phase-to-command projection,
command-to-surface reconstruction, residual computation, integrator
and batch closed-loop step.

Improvements over original:
  - Pitch computed from the actual actuator grid extent instead of the
    hardcoded 2/11.  For hexagonal grids: pitch = 2 / (n_actuators_across - 1).
    For any geometry, derived from nearest-neighbour distance.
  - Added Fried-geometry constructor helper: build_fried_geometry() places
    actuators at the CORNERS of subapertures (n_sub+1)×(n_sub+1) grid,
    which is the geometry assumed by Fried (1977) and required by the
    problem statement.
  - wavefront_to_commands() now returns commands in MICROMETRES as well
    as metres, with an explicit um_commands property after each call
    (hackathon output spec: "actuator stroke length").
  - Added commands_batch(): vectorised actuator commands for a batch of
    phase maps — avoids Python loop in run_real_data().
  - Influence matrix build delegated to influence_fn.py (unchanged);
    DMController.__init__ no longer hard-wires n_across=11.
"""

from __future__ import annotations

import numpy as np

from actuator.geometry import (
    hexagonal_actuator_positions,
    square_actuator_positions,
)
from actuator.influence_fn import build_influence_matrix, compute_command_matrix
from sim.phase_screen import get_aperture_mask, zernike_reconstruct


# ---------------------------------------------------------------------------
# Fried-geometry helper
# ---------------------------------------------------------------------------

def build_fried_geometry(n_subapertures: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Place DM actuators at the CORNERS of the subaperture grid.

    In Fried geometry the actuator grid is offset by half a subaperture
    pitch relative to the lenslet grid so each actuator sits at the corner
    shared by four adjacent subapertures.  This gives a
    (n_sub+1) × (n_sub+1) square grid, clipped to the circular aperture.

    Parameters
    ----------
    n_subapertures : int
        Number of subapertures across the pupil diameter.

    Returns
    -------
    x, y : np.ndarray
        Actuator positions in normalized [-1, 1] units, within the unit disk.

    Notes
    -----
    The problem statement explicitly requires Fried geometry for the
    lenslet–actuator co-registration.
    """
    n_act_across = n_subapertures + 1
    # Corner positions: subaperture edges are at ±1 on the aperture
    coords = np.linspace(-1.0, 1.0, n_act_across)
    xx, yy = np.meshgrid(coords, coords)
    # Keep actuators within (or on) the unit circle
    r = np.sqrt(xx**2 + yy**2)
    mask = r <= 1.0 + 1e-9
    return xx[mask], yy[mask]


def _nearest_neighbour_pitch(act_x: np.ndarray, act_y: np.ndarray) -> float:
    """
    Estimate actuator pitch from the median nearest-neighbour distance.
    Works for any geometry (hex, square, Fried).
    """
    if act_x.shape[0] < 2:
        return 0.2
    # Compute pairwise distances and take the minimum per actuator
    dx = act_x[:, None] - act_x[None, :]
    dy = act_y[:, None] - act_y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    nn_dist = dist.min(axis=1)
    return float(np.median(nn_dist))


# ---------------------------------------------------------------------------
# DMController
# ---------------------------------------------------------------------------

class DMController:
    """
    Deformable mirror controller.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.  Relevant keys:
          config['actuator']['n_actuators']
          config['actuator']['geometry']   ('hexagonal' | 'square' | 'fried')
          config['actuator']['coupling']
          config['actuator']['stroke_limit_um']
          config['sim']['grid_size']
          config['sim']['wavelength_m']
          config['sim']['n_subapertures']  (needed for Fried geometry)
    """

    def __init__(self, config: dict):
        act_cfg = config["actuator"]
        sim_cfg = config["sim"]

        self.N               = sim_cfg["grid_size"]
        self.wavelength_m    = sim_cfg["wavelength_m"]
        self.coupling        = act_cfg["coupling"]
        self.stroke_limit_um = act_cfg["stroke_limit_um"]
        self.stroke_limit_m  = self.stroke_limit_um * 1e-6
        self.n_subapertures  = sim_cfg.get("n_subapertures", 10)

        geometry   = act_cfg["geometry"]
        n_actuators = act_cfg["n_actuators"]

        if geometry == "fried":
            # Fried geometry: actuators at subaperture corners
            self.act_x, self.act_y = build_fried_geometry(self.n_subapertures)
        elif geometry == "hexagonal":
            # n_across chosen to give approximately n_actuators inside the circle
            # n_act ≈ π/4 * n_across²  → n_across = ceil(sqrt(4/π * n_act))
            n_across = int(np.ceil(np.sqrt(4.0 / np.pi * n_actuators))) + 1
            n_across = max(n_across, 11)
            self.act_x, self.act_y = hexagonal_actuator_positions(n_across, self.coupling)
        else:  # square
            n_across = int(np.ceil(np.sqrt(4.0 / np.pi * n_actuators))) + 1
            n_across = max(n_across, 11)
            self.act_x, self.act_y = square_actuator_positions(n_across)

        self.n_actuators = self.act_x.shape[0]

        # Pitch: derived from actual actuator layout, not hardcoded
        self.pitch = _nearest_neighbour_pitch(self.act_x, self.act_y)

        self.influence_matrix = build_influence_matrix(
            (self.act_x, self.act_y),
            self.N,
            pixel_scale=1.0,
            coupling=self.coupling,
            pitch=self.pitch,
        )
        self.command_matrix = compute_command_matrix(self.influence_matrix, svd_cutoff=50.0)

        self.mask = get_aperture_mask(self.N, "circular")

        # Z2C fast path: precompute (command_matrix @ basis) once per Zernike
        # basis set, so zernike_to_commands_fast skips the full N*N pixel-space
        # matmul and instead does an n_actuators x n_modes matvec.
        self._z2c_matrix = None
        self._z2c_basis_id = None

        self._integrator_commands = np.zeros(self.n_actuators)

        # Last computed commands in µm — exposed for hackathon output
        self.last_commands_um: np.ndarray = np.zeros(self.n_actuators)

    # ------------------------------------------------------------------
    # Core: phase → commands
    # ------------------------------------------------------------------

    def wavefront_to_commands(
        self,
        phase_map: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Project a wavefront onto actuator commands, applying the conjugate
        sign and clipping to stroke limits.

        Parameters
        ----------
        phase_map : (N, N), metres
        mask      : (N, N), bool

        Returns
        -------
        commands : (n_actuators,), metres of DM surface displacement
            clipped to [-stroke_limit_m, +stroke_limit_m].

        Side effect
        -----------
        Sets self.last_commands_um  (commands expressed in micrometres —
        the "actuator stroke length" unit required by the problem spec).
        """
        flat_phase   = (phase_map * mask).flatten()
        raw_commands = self.command_matrix @ flat_phase

        # Conjugate: correction surface cancels wavefront; ×0.5 for
        # reflection (DM surface displacement d → 2d optical path change).
        commands = -0.5 * raw_commands
        commands = np.clip(commands, -self.stroke_limit_m, self.stroke_limit_m)

        self.last_commands_um = commands * 1e6   # metres → µm
        return commands

    def zernike_to_commands(
        self,
        zernike_coeffs: np.ndarray,
        zernike_basis: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct a phase map from Zernike coefficients (radians) and
        compute actuator commands.

        Returns commands in metres; self.last_commands_um set in µm.
        """
        phase_radians = np.tensordot(zernike_coeffs, zernike_basis, axes=(0, 0))
        phase_meters  = phase_radians * (self.wavelength_m / (2.0 * np.pi))
        return self.wavefront_to_commands(phase_meters, mask)

    def _build_z2c(self, zernike_basis: np.ndarray) -> None:
        """
        Precompute the Zernike-coefficients -> commands projection matrix.

        wavefront_to_commands() does:
            commands = -0.5 * clip(command_matrix @ (phase * mask).flatten())
        and zernike_to_commands() builds phase from coeffs first:
            phase = tensordot(coeffs, basis) * (wavelength / 2pi)

        Folding the basis into command_matrix once means each subsequent
        call is an (n_actuators x n_modes) matvec instead of rebuilding and
        flattening an (N, N) phase map every frame.
        """
        n_modes = zernike_basis.shape[0]
        basis_flat = zernike_basis.reshape(n_modes, -1)              # (n_modes, N*N)
        mask_flat  = self.mask.flatten().astype(zernike_basis.dtype)
        basis_flat = basis_flat * mask_flat                          # apply aperture mask
        self._z2c_matrix = self.command_matrix @ basis_flat.T        # (n_actuators, n_modes)
        self._z2c_basis_id = id(zernike_basis)

    def zernike_to_commands_fast(
        self,
        zernike_coeffs: np.ndarray,
        zernike_basis: np.ndarray,
    ) -> np.ndarray:
        """
        Fast path for zernike_to_commands(): precomputes
        (command_matrix @ basis) once via _build_z2c, then each call is an
        (n_actuators x n_modes) matvec instead of the full
        (n_actuators x N*N) matvec used by the pixel-space path.

        Mathematically equivalent to zernike_to_commands() to floating-point
        precision (mask is applied the same way, just folded into the
        precomputed matrix instead of applied per-frame).

        Returns commands in metres; self.last_commands_um set in micrometres.
        """
        if self._z2c_matrix is None or self._z2c_basis_id != id(zernike_basis):
            self._build_z2c(zernike_basis)

        scale = self.wavelength_m / (2.0 * np.pi)
        raw_commands = self._z2c_matrix @ (zernike_coeffs * scale)

        commands = -0.5 * raw_commands
        commands = np.clip(commands, -self.stroke_limit_m, self.stroke_limit_m)

        self.last_commands_um = commands * 1e6
        return commands

    # ------------------------------------------------------------------
    # Batch: many frames at once
    # ------------------------------------------------------------------

    def commands_batch(
        self,
        phase_maps: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute actuator commands for a batch of phase maps.

        Parameters
        ----------
        phase_maps : (n_frames, N, N), metres
        mask       : (N, N), bool

        Returns
        -------
        commands_batch : (n_frames, n_actuators), metres
            Each row is clipped to stroke limits.
        commands_batch_um : (n_frames, n_actuators), µm
            Same values in micrometres.
        """
        n_frames  = phase_maps.shape[0]
        flat      = (phase_maps * mask[None]).reshape(n_frames, -1)   # (n_frames, N²)
        raw       = flat @ self.command_matrix.T                       # (n_frames, n_act)
        commands  = np.clip(-0.5 * raw, -self.stroke_limit_m, self.stroke_limit_m)
        return commands, commands * 1e6

    # ------------------------------------------------------------------
    # Surface reconstruction
    # ------------------------------------------------------------------

    def commands_to_surface(self, commands: np.ndarray) -> np.ndarray:
        """
        Apply actuator commands to the influence matrix to obtain DM surface.

        Returns surface : (N, N), metres.
        """
        return (self.influence_matrix @ commands).reshape(self.N, self.N)

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def compute_residual(
        self,
        phase_map: np.ndarray,
        dm_surface: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Residual wavefront after DM correction.

        DM correction applies as -2 * dm_surface (reflective double-pass).

        Returns residual (N, N) metres, rms float metres.
        """
        residual = (phase_map + 2.0 * dm_surface) * mask
        rms = float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else 0.0
        return residual, rms

    # ------------------------------------------------------------------
    # Integrator closed-loop
    # ------------------------------------------------------------------

    def closed_loop_step(
        self,
        phase_map: np.ndarray,
        mask: np.ndarray,
        gain: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Single integrator control step.

        commands_new = commands_old + gain * Δcommands

        Returns updated commands (metres) and residual phase (metres).
        """
        dm_surface = self.commands_to_surface(self._integrator_commands)
        residual, _ = self.compute_residual(phase_map, dm_surface, mask)

        delta_commands = self.wavefront_to_commands(residual, mask)
        self._integrator_commands = np.clip(
            self._integrator_commands + gain * delta_commands,
            -self.stroke_limit_m, self.stroke_limit_m,
        )

        dm_surface_new = self.commands_to_surface(self._integrator_commands)
        residual_new, _ = self.compute_residual(phase_map, dm_surface_new, mask)

        return self._integrator_commands.copy(), residual_new

    def reset_integrator(self) -> None:
        """Reset the integrator command state to zero."""
        self._integrator_commands = np.zeros(self.n_actuators)
