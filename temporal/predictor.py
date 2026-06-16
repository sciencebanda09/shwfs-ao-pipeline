"""
temporal/predictor.py
=======================
Wavefront prediction (servo-lag compensation) and a full closed-loop
AO simulator comparing open-loop, closed-loop, and predictive AO.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from sim.phase_screen import apply_aperture_mask, get_aperture_mask, compute_strehl_ratio


class WavefrontPredictor:
    """
    Wraps a trained LSTM/Transformer temporal model with a ring buffer
    of recent Zernike frames for one-step-ahead prediction.

    Parameters
    ----------
    model : nn.Module
        Trained temporal model (ZernikeTimeSeries or TemporalTransformer).
    seq_len : int
        Required input sequence length.
    device : torch.device
    """

    def __init__(self, model, seq_len: int, device: torch.device):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.buffer: deque = deque(maxlen=seq_len)

    def update(self, new_zernike_frame: np.ndarray) -> None:
        """
        Append a new Zernike coefficient frame to the ring buffer.

        Parameters
        ----------
        new_zernike_frame : np.ndarray, shape (n_zernike,)
        """
        self.buffer.append(np.asarray(new_zernike_frame, dtype=np.float32))

    def predict(self) -> np.ndarray:
        """
        Run the model forward pass on the current buffer if full;
        otherwise return the most recent frame (cold-start fallback).

        Returns
        -------
        predicted : np.ndarray, shape (n_zernike,)
        """
        if len(self.buffer) < self.seq_len:
            if len(self.buffer) == 0:
                raise RuntimeError("Predictor buffer is empty; call update() first")
            return self.buffer[-1].copy()

        seq = np.stack(list(self.buffer), axis=0)  # (seq_len, n_zernike)
        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)

        return pred.squeeze(0).cpu().numpy()

    def reset(self) -> None:
        """Clear the ring buffer."""
        self.buffer.clear()


class ClosedLoopSimulator:
    """
    Full AO closed-loop simulator comparing open-loop, closed-loop
    (no prediction), and closed-loop with predictive compensation.

    Parameters
    ----------
    atmosphere : MultiLayerAtmosphere
    sensor : SHWFSSensor
    dm_controller : DMController
    reconstructor : object with .reconstruct(slopes_x, slopes_y) ->
        Zernike coefficients
    predictor : WavefrontPredictor
    config : dict
        Parsed config.yaml.
    """

    def __init__(self, atmosphere, sensor, dm_controller, reconstructor, predictor: WavefrontPredictor, config: dict):
        self.atmosphere = atmosphere
        self.sensor = sensor
        self.dm = dm_controller
        self.reconstructor = reconstructor
        self.predictor = predictor
        self.config = config

        sim_cfg = config["sim"]
        self.N = sim_cfg["grid_size"]
        self.dt = sim_cfg["dt_s"]
        self.wavelength = sim_cfg["wavelength_m"]
        self.n_zernike = sim_cfg["n_zernike"]

        from reconstruction.zernike import zernike_basis
        self.basis = zernike_basis(self.n_zernike, self.N)
        self.mask = get_aperture_mask(self.N, "circular")

    def run(self, n_frames: int, use_prediction: bool = True) -> dict:
        """
        Run the main simulation loop.

        For each frame:
          1. atmosphere.evolve(dt)
          2. sensor.propagate(phase) -> slopes
          3. reconstructor.reconstruct(slopes) -> Zernike (measured)
          4. predictor.update + predictor.predict -> predicted Zernike
          5. dm_controller.zernike_to_commands -> DM surface
          6. residual = phase - dm_surface (with/without prediction)
          7. log metrics

        Parameters
        ----------
        n_frames : int
        use_prediction : bool
            Whether to additionally compute the predictive-AO branch.

        Returns
        -------
        results : dict with keys
            'rms_open_loop', 'rms_closed_loop_no_pred',
            'rms_closed_loop_with_pred', 'strehl_no_pred',
            'strehl_with_pred'
            Each value is an np.ndarray of length n_frames (radians for
            rms_*, dimensionless for strehl_*).
        """
        self.dm.reset_integrator()
        self.predictor.reset()

        rms_open = np.zeros(n_frames)
        rms_closed_no_pred = np.zeros(n_frames)
        rms_closed_with_pred = np.zeros(n_frames)
        strehl_no_pred = np.zeros(n_frames)
        strehl_with_pred = np.zeros(n_frames)
        all_zernike = np.zeros((n_frames, self.n_zernike))
        last_residual_phase = None
        last_actuator_commands = None

        wavelength_factor = self.wavelength / (2.0 * np.pi)

        for k in range(n_frames):
            self.atmosphere.evolve(self.dt)
            phase_rad = self.atmosphere.get_integrated_phase_radians()
            phase_masked = apply_aperture_mask(phase_rad, "circular")

            sx, sy = self.sensor.propagate(phase_masked)
            measured_zernike = self.reconstructor.reconstruct(sx, sy)

            # --- Open loop: RMS of the raw wavefront ---
            rms_open[k] = float(np.sqrt(np.mean(phase_masked[self.mask] ** 2)))

            # --- Closed loop, no prediction ---
            commands_np = self.dm.zernike_to_commands(
                measured_zernike, self.basis, self.mask
            )
            dm_surface = self.dm.commands_to_surface(commands_np)
            residual_np, rms_np = self.dm.compute_residual(
                phase_masked * wavelength_factor, dm_surface, self.mask
            )
            rms_closed_no_pred[k] = rms_np / wavelength_factor

            # --- Closed loop with prediction ---
            self.predictor.update(measured_zernike)
            if use_prediction:
                predicted_zernike = self.predictor.predict()
            else:
                predicted_zernike = measured_zernike

            commands_pred = self.dm.zernike_to_commands(
                predicted_zernike, self.basis, self.mask
            )
            dm_surface_pred = self.dm.commands_to_surface(commands_pred)
            residual_pred, rms_pred = self.dm.compute_residual(
                phase_masked * wavelength_factor, dm_surface_pred, self.mask
            )
            rms_closed_with_pred[k] = rms_pred / wavelength_factor
            all_zernike[k] = measured_zernike
            last_residual_phase = residual_pred
            last_actuator_commands = commands_pred

            strehl_no_pred[k] = compute_strehl_ratio(rms_closed_no_pred[k])
            strehl_with_pred[k] = compute_strehl_ratio(rms_closed_with_pred[k])

        return {
            "rms_open_loop": rms_open,
            "rms_closed_loop_no_pred": rms_closed_no_pred,
            "rms_closed_loop_with_pred": rms_closed_with_pred,
            "strehl_no_pred": strehl_no_pred,
            "strehl_with_pred": strehl_with_pred,
            "all_zernike": all_zernike,
            "last_residual_phase": last_residual_phase,
            "actuator_commands": last_actuator_commands,
        }



