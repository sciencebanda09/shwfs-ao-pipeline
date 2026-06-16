"""
actuator/dm_command.py
=======================
Deformable mirror command generation: phase-to-command projection,
command-to-surface reconstruction, residual computation, and a simple
integrator closed-loop step.
"""

from __future__ import annotations

import numpy as np

from actuator.geometry import hexagonal_actuator_positions, square_actuator_positions
from actuator.influence_fn import build_influence_matrix, compute_command_matrix
from sim.phase_screen import get_aperture_mask, zernike_reconstruct


class DMController:
    """
    Deformable mirror controller.

    Parameters
    ----------
    config : dict
        Parsed config.yaml. Relevant keys:
          - config['actuator']['n_actuators']
          - config['actuator']['geometry'] ('hexagonal' | 'square')
          - config['actuator']['coupling']
          - config['actuator']['stroke_limit_um']
          - config['sim']['grid_size']
          - config['sim']['wavelength_m']
    """

    def __init__(self, config: dict):
        act_cfg = config["actuator"]
        sim_cfg = config["sim"]

        self.N = sim_cfg["grid_size"]
        self.wavelength_m = sim_cfg["wavelength_m"]
        self.coupling = act_cfg["coupling"]
        self.stroke_limit_um = act_cfg["stroke_limit_um"]

        n_actuators = act_cfg["n_actuators"]
        geometry = act_cfg["geometry"]

        if geometry == "hexagonal":
            n_across = int(round(np.sqrt(n_actuators / (np.pi / 4)))) + 1
            self.act_x, self.act_y = hexagonal_actuator_positions(max(n_across, 11), self.coupling)
        else:
            n_across = int(round(np.sqrt(n_actuators / (np.pi / 4)))) + 1
            self.act_x, self.act_y = square_actuator_positions(max(n_across, 11))

        self.n_actuators = self.act_x.shape[0]
        self.pitch = 2.0 / 11.0

        self.influence_matrix = build_influence_matrix(
            (self.act_x, self.act_y), self.N, pixel_scale=1.0, coupling=self.coupling, pitch=self.pitch
        )
        self.command_matrix = compute_command_matrix(self.influence_matrix, svd_cutoff=50.0)

        self.mask = get_aperture_mask(self.N, "circular")

        # Convert wavefront (meters) <-> DM stroke (meters): DM surface
        # displacement = wavefront / 2 (reflection doubles optical path),
        # so commands represent surface displacement.
        self.stroke_limit_m = self.stroke_limit_um * 1e-6

        self._integrator_commands = np.zeros(self.n_actuators)

    def wavefront_to_commands(self, phase_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Project a wavefront (radians or meters, see note) onto actuator
        commands, applying the correction sign (negate) and clipping to
        stroke limits.

        Parameters
        ----------
        phase_map : np.ndarray, shape (N, N)
            Wavefront error in meters.
        mask : np.ndarray of bool, shape (N, N)
            Valid aperture mask.

        Returns
        -------
        commands : np.ndarray, shape (n_actuators,)
            Actuator commands (meters of surface displacement),
            clipped to [-stroke_limit_m, +stroke_limit_m].
        """
        flat_phase = (phase_map * mask).flatten()
        raw_commands = self.command_matrix @ flat_phase

        # Correction surface must cancel the wavefront -> negate.
        # DM surface displacement of d produces 2*d optical path change,
        # so the correction command is -phase/2.
        commands = -0.5 * raw_commands
        commands = np.clip(commands, -self.stroke_limit_m, self.stroke_limit_m)
        return commands

    def zernike_to_commands(self, zernike_coeffs: np.ndarray, zernike_basis: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct a phase map from Zernike coefficients (radians),
        convert to meters, then compute actuator commands.

        Parameters
        ----------
        zernike_coeffs : np.ndarray, shape (n_modes,), radians
        zernike_basis : np.ndarray, shape (n_modes, N, N)
        mask : np.ndarray of bool, shape (N, N)

        Returns
        -------
        commands : np.ndarray, shape (n_actuators,)
        """
        phase_radians = np.tensordot(zernike_coeffs, zernike_basis, axes=(0, 0))
        phase_meters = phase_radians * (self.wavelength_m / (2.0 * np.pi))
        return self.wavefront_to_commands(phase_meters, mask)

    def commands_to_surface(self, commands: np.ndarray) -> np.ndarray:
        """
        Apply actuator commands to the influence matrix to obtain the
        resulting DM surface shape.

        Parameters
        ----------
        commands : np.ndarray, shape (n_actuators,)

        Returns
        -------
        surface : np.ndarray, shape (N, N), meters
        """
        flat_surface = self.influence_matrix @ commands
        return flat_surface.reshape(self.N, self.N)

    def compute_residual(self, phase_map: np.ndarray, dm_surface: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute the residual wavefront after DM correction and its RMS.

        The DM correction applies as ``-2 * dm_surface`` of optical
        path (reflective double-pass).

        Parameters
        ----------
        phase_map : np.ndarray, shape (N, N), meters
        dm_surface : np.ndarray, shape (N, N), meters
        mask : np.ndarray of bool, shape (N, N)

        Returns
        -------
        residual : np.ndarray, shape (N, N), meters
        rms : float
            RMS residual over the valid aperture, meters.
        """
        residual = (phase_map + 2.0 * dm_surface) * mask
        rms = float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else 0.0
        return residual, rms

    def closed_loop_step(self, phase_map: np.ndarray, mask: np.ndarray, gain: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Single integrator control step.

        commands_new = commands_old + gain * delta_commands

        where ``delta_commands`` are the commands computed from the
        current residual wavefront.

        Parameters
        ----------
        phase_map : np.ndarray, shape (N, N), meters
        mask : np.ndarray of bool, shape (N, N)
        gain : float
            Integrator loop gain (0 < gain <= 1).

        Returns
        -------
        commands : np.ndarray, shape (n_actuators,)
            Updated integrator command vector.
        residual : np.ndarray, shape (N, N), meters
            Residual wavefront after applying the updated commands.
        """
        dm_surface = self.commands_to_surface(self._integrator_commands)
        residual, _ = self.compute_residual(phase_map, dm_surface, mask)

        delta_commands = self.wavefront_to_commands(residual, mask)
        self._integrator_commands = self._integrator_commands + gain * delta_commands
        self._integrator_commands = np.clip(
            self._integrator_commands, -self.stroke_limit_m, self.stroke_limit_m
        )

        dm_surface_new = self.commands_to_surface(self._integrator_commands)
        residual_new, _ = self.compute_residual(phase_map, dm_surface_new, mask)

        return self._integrator_commands.copy(), residual_new

    def reset_integrator(self) -> None:
        """Reset the integrator command state to zero."""
        self._integrator_commands = np.zeros(self.n_actuators)
