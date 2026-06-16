"""
viz/dashboard.py
=================
Plotly/Dash-based interactive dashboard for the AO pipeline results.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_dashboard(results_dict: dict, config: dict) -> go.Figure:
    """
    Build a multi-panel Plotly figure summarizing AO pipeline results.

    Panels
    ------
    1. Strehl ratio time series (open loop vs closed loop vs predictive)
    2. RMS WFE vs noise level (classical vs CNN)
    3. Zernike coefficient power spectrum (mean variance per mode)
    4. r0 estimation accuracy scatter (estimated vs true)
    5. DM actuator command map (heatmap)
    6. Residual phase map

    Parameters
    ----------
    results_dict : dict
        Expected (optional) keys:
          - 'strehl_open', 'strehl_closed', 'strehl_pred', 'time'
          - 'noise_levels', 'rms_classical', 'rms_cnn'
          - 'zernike_variance' (n_zernike,)
          - 'r0_est', 'r0_true' (arrays)
          - 'actuator_commands' (n_actuators,) or 2D map
          - 'residual_phase' (N, N)
    config : dict
        Parsed config.yaml.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Strehl ratio time series",
            "RMS WFE vs noise level",
            "Zernike power spectrum",
            "r0 estimation accuracy",
            "DM actuator commands",
            "Residual phase map",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "heatmap"}],
        ],
    )

    # Panel 1: Strehl time series
    t = results_dict.get("time", np.arange(len(results_dict.get("strehl_open", []))))
    if "strehl_open" in results_dict:
        fig.add_trace(go.Scatter(x=t, y=results_dict["strehl_open"], name="open loop", line=dict(color="gray")), row=1, col=1)
    if "strehl_closed" in results_dict:
        fig.add_trace(go.Scatter(x=t, y=results_dict["strehl_closed"], name="closed loop", line=dict(color="blue")), row=1, col=1)
    if "strehl_pred" in results_dict:
        fig.add_trace(go.Scatter(x=t, y=results_dict["strehl_pred"], name="predictive AO", line=dict(color="green")), row=1, col=1)

    # Panel 2: RMS WFE vs noise level
    if "noise_levels" in results_dict:
        fig.add_trace(go.Scatter(x=results_dict["noise_levels"], y=results_dict.get("rms_classical", []), name="classical", mode="lines+markers"), row=1, col=2)
        fig.add_trace(go.Scatter(x=results_dict["noise_levels"], y=results_dict.get("rms_cnn", []), name="CNN", mode="lines+markers"), row=1, col=2)
        fig.update_xaxes(type="log", row=1, col=2)
        fig.update_yaxes(type="log", row=1, col=2)

    # Panel 3: Zernike power spectrum
    if "zernike_variance" in results_dict:
        var = results_dict["zernike_variance"]
        fig.add_trace(go.Bar(x=np.arange(1, len(var) + 1), y=var, name="variance"), row=1, col=3)

    # Panel 4: r0 estimation accuracy scatter
    if "r0_est" in results_dict and "r0_true" in results_dict:
        r0_est = results_dict["r0_est"]
        r0_true = results_dict["r0_true"]
        fig.add_trace(go.Scatter(x=r0_true, y=r0_est, mode="markers", name="r0 estimates"), row=2, col=1)
        lo, hi = float(np.min(r0_true)), float(np.max(r0_true))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line=dict(dash="dash", color="black")), row=2, col=1)

    # Panel 5: DM actuator commands
    if "actuator_commands" in results_dict:
        commands = np.asarray(results_dict["actuator_commands"])
        if commands.ndim == 1:
            fig.add_trace(go.Bar(x=np.arange(len(commands)), y=commands, name="commands"), row=2, col=2)
        else:
            fig.add_trace(go.Heatmap(z=commands, colorscale="RdBu"), row=2, col=2)

    # Panel 6: Residual phase map
    if "residual_phase" in results_dict:
        fig.add_trace(go.Heatmap(z=results_dict["residual_phase"], colorscale="RdBu"), row=2, col=3)

    fig.update_layout(height=800, width=1200, title_text="BAH2026 AO Pipeline Dashboard", showlegend=True)

    return fig


def run_dashboard(results_dict: dict, config: dict, port: int = 8050):
    """
    Launch a Dash app displaying the dashboard, with a dropdown to
    select a turbulence scenario and sliders for r0 and wind speed that
    re-run the closed-loop simulation on the fly.

    Parameters
    ----------
    results_dict : dict
    config : dict
    port : int

    Returns
    -------
    app : dash.Dash
        The Dash application instance (call app.run(...) to serve).
    """
    from dash import Dash, dcc, html, Input, Output

    app = Dash(__name__)

    base_fig = create_dashboard(results_dict, config)

    app.layout = html.Div(
        [
            html.H1("BAH2026 AO Pipeline Dashboard"),
            html.Div(
                [
                    html.Label("Turbulence scenario"),
                    dcc.Dropdown(
                        id="scenario-dropdown",
                        options=[
                            {"label": "Baseline", "value": "baseline"},
                            {"label": "Strong turbulence", "value": "strong"},
                            {"label": "Weak turbulence", "value": "weak"},
                        ],
                        value="baseline",
                    ),
                    html.Label("r0 (m)"),
                    dcc.Slider(id="r0-slider", min=0.05, max=0.30, step=0.01, value=config["turbulence"]["r0_m"]),
                    html.Label("Wind speed (m/s)"),
                    dcc.Slider(id="wind-slider", min=1, max=30, step=1, value=config["turbulence"]["layer_wind_speeds_ms"][0]),
                ],
                style={"width": "50%", "margin": "auto"},
            ),
            dcc.Graph(id="main-dashboard", figure=base_fig),
        ]
    )

    @app.callback(
        Output("main-dashboard", "figure"),
        Input("scenario-dropdown", "value"),
        Input("r0-slider", "value"),
        Input("wind-slider", "value"),
    )
    def _update_dashboard(scenario, r0, wind_speed):
        # Re-run a short closed-loop simulation with the updated
        # turbulence parameters and rebuild the dashboard figure.
        import copy
        from sim.turbulence import build_atmosphere_from_config
        from sim.shwfs import SHWFSSensor
        from reconstruction.zernike import zernike_basis
        from reconstruction.classical import ModalReconstructor
        from temporal.predictor import WavefrontPredictor, ClosedLoopSimulator
        from actuator.dm_command import DMController
        from temporal.train_temporal import load_trained_temporal_model
        import torch
        from pathlib import Path as _Path

        cfg = copy.deepcopy(config)
        cfg["turbulence"]["r0_m"] = r0
        n_layers = cfg["turbulence"]["n_layers"]
        cfg["turbulence"]["layer_wind_speeds_ms"] = [wind_speed] * n_layers

        if scenario == "strong":
            cfg["turbulence"]["r0_m"] = min(r0, 0.08)
        elif scenario == "weak":
            cfg["turbulence"]["r0_m"] = max(r0, 0.25)

        sim_cfg = cfg["sim"]
        N = sim_cfg["grid_size"]
        pixel_scale = sim_cfg["aperture_diameter_m"] / N

        atmosphere = build_atmosphere_from_config(cfg, N=N, pixel_scale=pixel_scale, seed=123)
        sensor = SHWFSSensor(
            n_subapertures=sim_cfg["n_subapertures"],
            pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
            focal_length=sim_cfg["focal_length_m"],
            pitch=sim_cfg["mla_pitch_m"],
            wavelength=sim_cfg["wavelength_m"],
        )
        basis = zernike_basis(sim_cfg["n_zernike"], N)
        modal_recon = ModalReconstructor(sensor, basis, sim_cfg["n_zernike"], cfg["reconstruction"]["svd_condition_number"])
        dm = DMController(cfg)

        device = torch.device("cpu")
        lstm_path = _Path(cfg["paths"]["models_dir"]) / "temporal_model.pt"
        use_prediction = lstm_path.exists()

        if use_prediction:
            model = load_trained_temporal_model(str(lstm_path), cfg, device)
            predictor = WavefrontPredictor(model, cfg["temporal"]["sequence_length"], device)
        else:
            import warnings
            warnings.warn(
                f"temporal model checkpoint not found at {lstm_path}; "
                "dashboard predictive-AO branch falling back to closed-loop."
            )
            predictor = None

        closed_loop = ClosedLoopSimulator(atmosphere, sensor, dm, modal_recon, predictor, cfg)
        sim_results = closed_loop.run(n_frames=50, use_prediction=use_prediction)

        new_results = dict(results_dict)
        new_results["strehl_open"] = np.exp(-(sim_results["rms_open_loop"] ** 2))
        new_results["strehl_closed"] = sim_results["strehl_no_pred"]
        new_results["strehl_pred"] = (
            sim_results["strehl_with_pred"] if use_prediction else sim_results["strehl_no_pred"]
        )
        new_results["time"] = np.arange(50) * sim_cfg["dt_s"]

        return create_dashboard(new_results, cfg)

    return app


def export_dashboard_html(fig: go.Figure, path: str) -> str:
    """
    Save a standalone HTML file for a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    path : str

    Returns
    -------
    path : str
    """
    fig.write_html(path, include_plotlyjs="cdn")
    return path
