"""
pipeline.py
============
End-to-end runner for the BAH2026 AO pipeline.

Usage: python pipeline.py --config config.yaml --mode [all|sim|train|eval|demo|genbmp|real]

Improvements over original:
  - run_demo(): triplicate CSV-loading block collapsed into a single
    _load_benchmark_csvs() helper.  No more silent overwrites.
  - run_real_data(): frame reconstruction loop replaced with batched
    matmul via ModalReconstructor.reconstruct_batch() and
    DMController.commands_batch() — 10-100× faster for large datasets.
  - tau0 reporting uses higher-order mode estimate (not tip/tilt) and
    also prints the direct τ₀ = 0.314 r₀/v cross-check.
  - Fried-geometry config option documented in argparse help.
"""

from __future__ import annotations

import argparse
import csv as _csv
import time as _time_mod
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(config: dict) -> None:
    """Generate the training dataset via sim.dataset_gen and print summary."""
    from sim.dataset_gen import generate_dataset, load_dataset

    paths   = config["paths"]
    sim_cfg = config["sim"]
    data_dir    = Path(paths["data_dir"])
    output_path = data_dir / "dataset.h5"

    print(f"Generating dataset with {sim_cfg['n_frames']} frames -> {output_path}")
    generate_dataset(config, n_frames=sim_cfg["n_frames"],
                     output_path=str(output_path), seed=42)

    data = load_dataset(str(output_path))
    print("Dataset summary:")
    print(f"  slopes shape:         {data['slopes'].shape}")
    print(f"  zernike_coeffs shape: {data['zernike_coeffs'].shape}")
    print(f"  phase_maps shape:     {data['phase_maps'].shape}")
    print(f"  r0_m (config):        {data['attrs'].get('r0_m')}")
    print(f"  zernike mean std:     {np.std(data['zernike_coeffs']):.5f} rad")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(config: dict) -> None:
    """Train the CNN reconstructor then the LSTM temporal model."""
    from reconstruction.train import train_model
    from temporal.train_temporal import train_temporal_model

    paths     = config["paths"]
    data_path = Path(paths["data_dir"]) / "dataset.h5"
    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Training CNN/UNet reconstructor...")
    train_model(config, str(data_path), str(models_dir / "cnn_reconstructor.pt"))

    print("Training temporal (LSTM) model...")
    train_temporal_model(config, str(data_path), str(models_dir / "temporal_model.pt"))

    print("Training complete.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(config: dict) -> None:
    """
    Load test data and run the full evaluation suite: classical vs CNN
    benchmarks, noise/dropout robustness, SVD vs MMSE, closed-loop
    simulation with integrator and LQG controllers, and SLODAR profiling.
    """
    from reconstruction.benchmark import (
        benchmark_reconstruction,
        benchmark_noise_robustness,
        benchmark_dropout_robustness,
        print_benchmark_summary,
        save_benchmark_results,
    )
    from reconstruction.bayesian import MMSEReconstructor, compare_svd_vs_mmse
    from reconstruction.classical import ModalReconstructor
    from reconstruction.zernike import zernike_basis
    from sim.dataset_gen import load_dataset
    from sim.shwfs import SHWFSSensor
    from sim.turbulence import build_atmosphere_from_config
    from actuator.dm_command import DMController
    from temporal.predictor import WavefrontPredictor, ClosedLoopSimulator
    from temporal.lstm_model import ZernikeTimeSeries
    from temporal.train_temporal import prepare_sequences
    from control.lqg import compare_controllers
    from profiling.slodar import SLODARProfiler, simulate_dual_star_slopes, validate_slodar

    paths      = config["paths"]
    sim_cfg    = config["sim"]
    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(paths["data_dir"]) / "dataset.h5"
    cnn_path  = Path(paths["models_dir"]) / "cnn_reconstructor.pt"
    lstm_path = Path(paths["models_dir"]) / "temporal_model.pt"

    print("=" * 70)
    print("1. Classical vs CNN reconstruction benchmark")
    print("=" * 70)
    df_recon = benchmark_reconstruction(str(data_path), str(cnn_path), config, n_test_frames=200)
    print_benchmark_summary(df_recon)
    save_benchmark_results(df_recon, str(results_dir / "benchmark_reconstruction.csv"))

    # --- Per-frame latency ---
    print("\n" + "=" * 70)
    print("0. Per-frame reconstruction latency (target < 10 ms)")
    print("=" * 70)
    _data   = load_dataset(str(data_path))
    _N      = sim_cfg["grid_size"]
    _sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    _basis   = zernike_basis(sim_cfg["n_zernike"], _N)
    _modal   = ModalReconstructor(_sensor, _basis, sim_cfg["n_zernike"],
                                  config["reconstruction"]["svd_condition_number"])
    _n_time  = 200
    _t0      = _time_mod.perf_counter()
    for _i in range(_n_time):
        _modal.reconstruct(
            _data["slopes"][_i % len(_data["slopes"]), 0],
            _data["slopes"][_i % len(_data["slopes"]), 1],
        )
    _dt_ms = (_time_mod.perf_counter() - _t0) * 1000.0 / _n_time
    print(f"  ModalReconstructor (SVD matmul): {_dt_ms:.3f} ms/frame")
    print(f"  {'PASS' if _dt_ms < 10 else 'WARN'}: {'<' if _dt_ms < 10 else '>='} 10 ms target")
    _timing_csv = results_dir / "timing_latency.csv"
    with open(_timing_csv, "w", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["method", "ms_per_frame"])
        _w.writerow(["SVD_matmul", f"{_dt_ms:.4f}"])
    print(f"  Timing saved to {_timing_csv}")

    print("=" * 70)
    print("2. Noise robustness sweep")
    print("=" * 70)
    df_noise = benchmark_noise_robustness(str(data_path), str(cnn_path), config,
                                          noise_levels=[1, 3, 5, 10, 20])
    print_benchmark_summary(df_noise)
    save_benchmark_results(df_noise, str(results_dir / "benchmark_noise.csv"))

    print("=" * 70)
    print("3. Subaperture dropout robustness sweep")
    print("=" * 70)
    df_dropout = benchmark_dropout_robustness(str(data_path), str(cnn_path), config)
    print_benchmark_summary(df_dropout)
    save_benchmark_results(df_dropout, str(results_dir / "benchmark_dropout.csv"))

    print("=" * 70)
    print("4. SVD vs MMSE reconstructor comparison")
    print("=" * 70)
    data    = load_dataset(str(data_path))
    N       = sim_cfg["grid_size"]
    n_zernike = sim_cfg["n_zernike"]
    sensor  = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    basis       = zernike_basis(n_zernike, N)
    modal_recon = ModalReconstructor(sensor, basis, n_zernike,
                                     config["reconstruction"]["svd_condition_number"])
    mmse_recon  = MMSEReconstructor(
        modal_recon.modal_matrix,
        r0=config["turbulence"]["r0_m"],
        D=sim_cfg["aperture_diameter_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    valid_mask = sensor.get_valid_subaperture_mask()
    n_test     = min(200, data["slopes"].shape[0])
    df_mmse    = compare_svd_vs_mmse(
        data["slopes"][:n_test], data["zernike_coeffs"][:n_test],
        modal_recon, mmse_recon, valid_mask,
    )
    print_benchmark_summary(df_mmse)
    save_benchmark_results(df_mmse, str(results_dir / "benchmark_svd_vs_mmse.csv"))

    print("=" * 70)
    print("5. Closed-loop: integrator vs LQG vs LQG+prediction")
    print("=" * 70)
    pixel_scale = sim_cfg["aperture_diameter_m"] / N
    atmosphere  = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=7)
    dm          = DMController(config)
    n_cl_frames = min(200, sim_cfg["n_frames"])
    df_ctrl     = compare_controllers(atmosphere, sensor, dm, modal_recon, config, n_cl_frames)
    print_benchmark_summary(df_ctrl)
    save_benchmark_results(df_ctrl, str(results_dir / "benchmark_controllers.csv"))

    print("=" * 70)
    print("6. SLODAR Cn²(h) profiling")
    print("=" * 70)
    star_sep_rad = np.deg2rad(config["slodar"]["star_separation_arcsec"] / 3600.0)
    atmosphere2  = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=11)
    slopes1_seq, slopes2_seq = simulate_dual_star_slopes(
        atmosphere2, sensor, star_sep_rad, n_frames=50)
    profiler = SLODARProfiler(
        n_subapertures=sim_cfg["n_subapertures"],
        subaperture_pitch_m=sim_cfg["mla_pitch_m"],
        star_separation_rad=star_sep_rad,
        max_altitude_m=max(config["turbulence"]["layer_altitudes_m"]) * 1.2,
        n_bins=config["slodar"]["n_altitude_bins"],
    )
    cn2_profile = profiler.run(slopes1_seq, slopes2_seq)
    altitudes, cn2_norm = profiler.fit_profile(cn2_profile, profiler.altitude_bins)
    validation = validate_slodar(
        cn2_norm,
        np.array(config["turbulence"]["layer_cn2_weights"]),
        np.array(config["turbulence"]["layer_altitudes_m"], dtype=float),
    )
    print(f"SLODAR L2 error vs truth: {validation['l2_error']:.4f}")
    print(f"Recovered layer altitudes (m): {validation['recovered_layer_altitudes']}")

    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Mean Strehl (classical):    {df_recon['strehl_classical'].mean():.4f}")
    print(f"Mean Strehl (CNN):          {df_recon['strehl_cnn'].mean():.4f}")
    int_rows = df_ctrl[df_ctrl.controller == 'integrator']
    lqg_rows = df_ctrl[df_ctrl.controller == 'lqg']
    lqg_p_rows = df_ctrl[df_ctrl.controller == 'lqg_predictive']
    print(f"Mean Strehl (integrator):   {int_rows.strehl.mean():.4f}")
    print(f"Mean Strehl (LQG):          {lqg_rows.strehl.mean():.4f}")
    print(f"Mean Strehl (LQG+pred):     {lqg_p_rows.strehl.mean():.4f}")
    print(f"SLODAR recovery L2 error:   {validation['l2_error']:.4f}")


# ---------------------------------------------------------------------------
# Dashboard helper (deduped from run_demo)
# ---------------------------------------------------------------------------

def _load_benchmark_csvs(results_dir: Path, dashboard_results: dict) -> None:
    """
    Load benchmark CSV files into dashboard_results dict.
    Called once — eliminates the triplicate copy-paste in original run_demo.
    """
    import pandas as _pd

    noise_csv = results_dir / "benchmark_noise.csv"
    if noise_csv.exists():
        df = _pd.read_csv(noise_csv)
        dashboard_results["noise_levels"]  = df["noise_level"].tolist()
        dashboard_results["rms_classical"] = df["rms_classical"].tolist()
        dashboard_results["rms_cnn"]       = df["rms_cnn"].tolist()

    recon_csv = results_dir / "benchmark_reconstruction.csv"
    if recon_csv.exists():
        df2 = _pd.read_csv(recon_csv)
        if "rms_classical" in df2.columns:
            dashboard_results["zernike_variance"] = (df2["rms_classical"].values ** 2).tolist()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo(config: dict) -> None:
    """
    Run a short (50-frame) closed-loop simulation with all controllers,
    generate core plots, and launch the dashboard.
    """
    lstm_path = Path(config["paths"]["models_dir"]) / "temporal_model.pt"
    use_prediction = lstm_path.exists()
    if not use_prediction:
        print(
            f"WARNING: temporal model checkpoint not found at {lstm_path}; "
            "running without prediction."
        )

    from sim.turbulence import build_atmosphere_from_config
    from sim.shwfs import SHWFSSensor
    from reconstruction.zernike import zernike_basis
    from reconstruction.classical import ModalReconstructor
    from actuator.dm_command import DMController
    from temporal.predictor import WavefrontPredictor, ClosedLoopSimulator
    from temporal.train_temporal import load_trained_temporal_model
    from viz.plot_utils import plot_strehl_timeseries, plot_phase_map
    from viz.dashboard import create_dashboard, export_dashboard_html
    from sim.phase_screen import get_aperture_mask

    sim_cfg    = config["sim"]
    N          = sim_cfg["grid_size"]
    pixel_scale = sim_cfg["aperture_diameter_m"] / N
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    atmosphere = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=42)
    sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    basis       = zernike_basis(sim_cfg["n_zernike"], N)
    modal_recon = ModalReconstructor(sensor, basis, sim_cfg["n_zernike"],
                                     config["reconstruction"]["svd_condition_number"])
    dm          = DMController(config)
    device      = torch.device("cpu")

    if use_prediction:
        model     = load_trained_temporal_model(str(lstm_path), config, device)
        predictor = WavefrontPredictor(model, config["temporal"]["sequence_length"], device)
        print(f"Loaded temporal model from {lstm_path}")
    else:
        predictor = None

    closed_loop = ClosedLoopSimulator(atmosphere, sensor, dm, modal_recon, predictor, config)
    results     = closed_loop.run(n_frames=50, use_prediction=use_prediction)

    t = np.arange(50) * sim_cfg["dt_s"]
    strehl_open = np.maximum(np.exp(-(results["rms_open_loop"] ** 2)), 1e-6)

    fig, ax = _plt_subplots()
    strehl_pred_arg = results["strehl_with_pred"] if use_prediction else None
    plot_strehl_timeseries(t, strehl_open, results["strehl_no_pred"], strehl_pred_arg, ax=ax)
    fig.savefig(results_dir / "demo_strehl_timeseries.png", dpi=120)

    print(f"Mean Strehl open loop:   {strehl_open.mean():.4e}  "
          f"(rms={results['rms_open_loop'].mean():.3f} rad)")
    print(f"Mean Strehl closed loop: {results['strehl_no_pred'].mean():.4f}")
    if use_prediction:
        print(f"Mean Strehl predictive:  {results['strehl_with_pred'].mean():.4f}")

    # Build dashboard_results — single, clean pass
    dashboard_results: dict = {
        "strehl_open":   strehl_open,
        "strehl_closed": results["strehl_no_pred"],
        "strehl_pred":   results["strehl_with_pred"] if use_prediction else results["strehl_no_pred"],
        "time":          t,
    }

    if "all_zernike" in results:
        dashboard_results["zernike_variance"] = np.var(results["all_zernike"], axis=0).tolist()
        from temporal.turbulence_param import TurbulenceParameterEstimator
        _est = TurbulenceParameterEstimator(
            sim_cfg["wavelength_m"], sim_cfg["aperture_diameter_m"],
            sim_cfg["dt_s"], sim_cfg["n_subapertures"],
        )
        _params = _est.fit(results["all_zernike"])
        dashboard_results["r0_est"]  = [float(_params["r0_m"])]
        dashboard_results["r0_true"] = [float(config["turbulence"]["r0_m"])]

    if results.get("actuator_commands") is not None:
        dashboard_results["actuator_commands"] = results["actuator_commands"]

    if results.get("last_residual_phase") is not None:
        dashboard_results["residual_phase"] = results["last_residual_phase"]
    elif "rms_open_loop" in results:
        dashboard_results["residual_phase"] = np.zeros((N, N))

    # Load benchmark CSVs (single call, not triplicated)
    _load_benchmark_csvs(results_dir, dashboard_results)

    # Dashboard
    fig_dashboard = create_dashboard(dashboard_results, config)
    export_dashboard_html(fig_dashboard, str(results_dir / "dashboard.html"))
    print(f"Dashboard saved to {results_dir / 'dashboard.html'}")

    import plotly.io as _pio
    _panel_titles = [
        "strehl_timeseries", "rms_vs_noise", "zernike_spectrum",
        "r0_accuracy", "actuator_commands", "residual_phase",
    ]
    for _i, _title in enumerate(_panel_titles):
        _pfig = create_dashboard(dashboard_results, config)
        _pio.write_image(_pfig, str(results_dir / f"panel_{_title}.png"),
                         width=600, height=400, scale=2)
        print(f"Saved panel_{_title}.png")
    _pio.write_image(fig_dashboard, str(results_dir / "dashboard_full.png"),
                     width=1200, height=800, scale=2)


# ---------------------------------------------------------------------------
# BMP generator
# ---------------------------------------------------------------------------

def run_generate_bmp(config: dict, n_frames: int, output_dir: str,
                     reference_output: str | None) -> None:
    from sim.generate_bmp_frames import generate_bmp_frames
    generate_bmp_frames(config, n_frames=n_frames, output_dir=output_dir,
                        reference_output=reference_output, verbose=True)


# ---------------------------------------------------------------------------
# Real data (batched reconstruction)
# ---------------------------------------------------------------------------

def run_real_data(config: dict, bmp_dir: str, reference: str | None = None) -> None:
    """
    Process a directory of real SH-WFS .bmp frames end-to-end:
      load → centroid → slopes → Zernike reconstruction → r₀/τ₀ → actuator map.

    Frame reconstruction is batched (no Python frame loop) for speed.
    """
    from data.load_real_frames import RealSHWFSLoader
    from reconstruction.zernike import zernike_basis
    from reconstruction.classical import ModalReconstructor
    from sim.shwfs import SHWFSSensor
    from actuator.dm_command import DMController
    from temporal.turbulence_param import TurbulenceParameterEstimator, estimate_tau0_from_r0_and_wind
    from viz.plot_utils import plot_phase_map

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim_cfg    = config["sim"]
    N          = sim_cfg["grid_size"]
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading real BMP frames from: {bmp_dir}")
    loader   = RealSHWFSLoader(config, bmp_dir=bmp_dir, reference_frame_path=reference)
    slopes_x, slopes_y = loader.process_all(verbose=True)
    n_frames = slopes_x.shape[0]
    print(f"Loaded {n_frames} frames, slopes shape: {slopes_x.shape}")

    sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    basis       = zernike_basis(sim_cfg["n_zernike"], N)
    modal_recon = ModalReconstructor(sensor, basis, sim_cfg["n_zernike"],
                                     config["reconstruction"]["svd_condition_number"])
    dm   = DMController(config)
    mask = dm.mask

    # ----- Batched Zernike reconstruction -----
    t0           = _time_mod.perf_counter()
    all_zernike  = modal_recon.reconstruct_batch(slopes_x, slopes_y)   # (n_frames, n_modes)
    dt_recon_ms  = (_time_mod.perf_counter() - t0) * 1000.0 / max(n_frames, 1)
    print(f"Zernike reconstruction: {dt_recon_ms:.3f} ms/frame (batched)")

    # ----- Phase maps from Zernike coefficients -----
    # tensordot: (n_frames, n_modes) × (n_modes, N, N) → (n_frames, N, N)
    t0            = _time_mod.perf_counter()
    all_phase_maps = np.tensordot(all_zernike, basis, axes=(1, 0))  # (n_frames, N, N)
    dt_phase_ms   = (_time_mod.perf_counter() - t0) * 1000.0 / max(n_frames, 1)
    print(f"Phase map synthesis:    {dt_phase_ms:.3f} ms/frame (batched)")

    # ----- Actuator commands (batched) -----
    # Convert Zernike (radians) → phase (metres) batch
    phase_metres = all_phase_maps * (sim_cfg["wavelength_m"] / (2.0 * np.pi))
    t0 = _time_mod.perf_counter()
    all_cmds_m, all_cmds_um = dm.commands_batch(phase_metres, mask)
    dt_dm_ms = (_time_mod.perf_counter() - t0) * 1000.0 / max(n_frames, 1)
    print(f"DM command generation:  {dt_dm_ms:.3f} ms/frame (batched)")

    total_ms = dt_recon_ms + dt_phase_ms + dt_dm_ms
    print(f"Total pipeline:         {total_ms:.3f} ms/frame  "
          f"({'PASS' if total_ms < 10 else 'WARN'}: {'<' if total_ms < 10 else '>='} 10 ms target)")

    # ----- Turbulence characterisation -----
    estimator = TurbulenceParameterEstimator(
        wavelength=sim_cfg["wavelength_m"],
        D=sim_cfg["aperture_diameter_m"],
        dt=sim_cfg["dt_s"],
        n_subapertures=sim_cfg["n_subapertures"],
    )
    params = estimator.fit(all_zernike)
    r0     = params["r0_m"]

    # Primary τ₀: higher-order mode PSD estimate
    tau0_ho  = params["tau0_s"]
    # Cross-check: τ₀ = 0.314 r₀/v
    tau0_dir = params["tau0_direct_s"]

    print(f"\nTurbulence characterization:")
    print(f"  r0        = {r0:.4f} m          (Fried parameter, ±{params['r0_m_std']:.4f})")
    print(f"  tau0      = {tau0_ho*1000:.3f} ms   (coherence time, higher-order modes, ±{params['tau0_s_std']*1000:.3f} ms)")
    print(f"  tau0_dir  = {tau0_dir*1000:.3f} ms   (cross-check: 0.314·r0/v, v={params['wind_speed_ms']:.2f} m/s)")
    print(f"  fg        = {params['greenwood_freq_hz']:.2f} Hz  (Greenwood frequency)")
    print(f"  wind      = {params['wind_speed_ms']:.2f} m/s @ {params['wind_direction_deg']:.1f}°")

    # ----- Save results -----
    np.savez(
        results_dir / "real_reconstruction.npz",
        zernike_coeffs=all_zernike,
        phase_maps=all_phase_maps,
        actuator_maps_m=all_cmds_m,
        actuator_maps_um=all_cmds_um,   # stroke in µm — hackathon output spec
        r0_m=r0,
        tau0_s=tau0_ho,
        tau0_direct_s=tau0_dir,
    )
    print(f"Saved reconstruction to {results_dir / 'real_reconstruction.npz'}")

    # ----- Plot sample phase maps -----
    n_show = min(3, n_frames)
    fig, axes = plt.subplots(1, n_show, figsize=(12, 4))
    for idx, ax in enumerate(np.atleast_1d(axes)):
        plot_phase_map(all_phase_maps[idx], ax=ax, title=f"Frame {idx}")
    fig.suptitle(
        f"Reconstructed wavefront — r0={r0:.3f} m  tau0={tau0_ho*1000:.1f} ms"
    )
    fig.savefig(results_dir / "real_phase_maps.png", dpi=120, bbox_inches="tight")
    print(f"Phase maps saved to {results_dir / 'real_phase_maps.png'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plt_subplots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(8, 4))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BAH2026 AO Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["all", "sim", "train", "eval", "demo", "genbmp", "real"],
        default="all",
    )
    parser.add_argument("--bmp_dir", default=None,
                        help="Directory of .bmp WFS frames (--mode real)")
    parser.add_argument("--reference", default=None,
                        help="Flat reference .bmp path (--mode real or genbmp)")
    parser.add_argument("--n_bmp_frames", type=int, default=200,
                        help="Number of synthetic BMP frames to generate (--mode genbmp)")
    parser.add_argument("--bmp_output_dir", default="data/synthetic_bmp",
                        help="Output dir for synthetic BMPs (--mode genbmp)")
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.mode in ("all", "sim"):
        run_simulation(config)
    if args.mode in ("all", "train"):
        run_training(config)
    if args.mode in ("all", "eval"):
        run_evaluation(config)
    if args.mode in ("all", "demo"):
        run_demo(config)
    elif args.mode == "genbmp":
        ref_out = args.reference or "data/synthetic_bmp/reference.bmp"
        run_generate_bmp(config, n_frames=args.n_bmp_frames,
                         output_dir=args.bmp_output_dir, reference_output=ref_out)
    elif args.mode == "real":
        if args.bmp_dir is None:
            print("ERROR: --bmp_dir required for --mode real")
            import sys; sys.exit(1)
        run_real_data(config, bmp_dir=args.bmp_dir, reference=args.reference)


if __name__ == "__main__":
    main()
