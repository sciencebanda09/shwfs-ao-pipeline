"""
pipeline.py
============
End-to-end runner for the BAH2026 AO pipeline.

Usage: python pipeline.py --config config.yaml --mode [all|sim|train|eval|demo]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_simulation(config: dict) -> None:
    """
    Generate the training dataset via sim.dataset_gen.generate_dataset
    and print summary statistics.
    """
    from sim.dataset_gen import generate_dataset, load_dataset

    paths = config["paths"]
    sim_cfg = config["sim"]
    data_dir = Path(paths["data_dir"])
    output_path = data_dir / "dataset.h5"

    print(f"Generating dataset with {sim_cfg['n_frames']} frames -> {output_path}")
    generate_dataset(config, n_frames=sim_cfg["n_frames"], output_path=str(output_path), seed=42)

    data = load_dataset(str(output_path))
    print("Dataset summary:")
    print(f"  slopes shape:         {data['slopes'].shape}")
    print(f"  zernike_coeffs shape: {data['zernike_coeffs'].shape}")
    print(f"  phase_maps shape:     {data['phase_maps'].shape}")
    print(f"  r0_m (config):        {data['attrs'].get('r0_m')}")
    print(f"  zernike mean std:     {np.std(data['zernike_coeffs']):.5f} rad")


def run_training(config: dict) -> None:
    """Train the CNN reconstructor then the LSTM temporal model."""
    from reconstruction.train import train_model
    from temporal.train_temporal import train_temporal_model

    paths = config["paths"]
    data_path = Path(paths["data_dir"]) / "dataset.h5"
    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Training CNN/UNet reconstructor...")
    cnn_save_path = models_dir / "cnn_reconstructor.pt"
    train_model(config, str(data_path), str(cnn_save_path))

    print("Training temporal (LSTM) model...")
    lstm_save_path = models_dir / "temporal_model.pt"
    train_temporal_model(config, str(data_path), str(lstm_save_path))

    print("Training complete.")


def run_evaluation(config: dict) -> None:
    """
    Load test data and run the full evaluation suite: classical vs CNN
    benchmarks, noise/dropout robustness, SVD vs MMSE comparison,
    closed-loop simulation with integrator and LQG controllers, and
    SLODAR profiling. Prints a full summary table with Strehl per
    method.
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

    paths = config["paths"]
    sim_cfg = config["sim"]
    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(paths["data_dir"]) / "dataset.h5"
    cnn_path = Path(paths["models_dir"]) / "cnn_reconstructor.pt"
    lstm_path = Path(paths["models_dir"]) / "temporal_model.pt"

    print("=" * 70)
    print("1. Classical vs CNN reconstruction benchmark")
    print("=" * 70)
    df_recon = benchmark_reconstruction(str(data_path), str(cnn_path), config, n_test_frames=200)
    print_benchmark_summary(df_recon)
    save_benchmark_results(df_recon, str(results_dir / "benchmark_reconstruction.csv"))

    # --- Timing: measure single-frame reconstruction latency ---
    print("\n======================================================================")
    print("0. Per-frame reconstruction latency (key for <10ms AO loop)")
    print("======================================================================")
    import time as _time_mod
    _data_tmp = load_dataset(str(data_path))
    _N_tmp = sim_cfg["grid_size"]
    _sensor_tmp = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    _basis_tmp = zernike_basis(sim_cfg["n_zernike"], _N_tmp)
    _modal_tmp = ModalReconstructor(_sensor_tmp, _basis_tmp, sim_cfg["n_zernike"],
                                    config["reconstruction"]["svd_condition_number"])
    _n_time = 200
    _t0 = _time_mod.perf_counter()
    for _i in range(_n_time):
        _modal_tmp.reconstruct(
            _data_tmp["slopes"][_i % len(_data_tmp["slopes"]), 0],
            _data_tmp["slopes"][_i % len(_data_tmp["slopes"]), 1],
        )
    _dt_ms = (_time_mod.perf_counter() - _t0) * 1000.0 / _n_time
    print(f"  ModalReconstructor (SVD matmul): {_dt_ms:.3f} ms/frame")
    print(f"  {'PASS' if _dt_ms < 10 else 'WARN'}: {'<' if _dt_ms < 10 else '>='} 10 ms target")
    import csv as _csv
    _timing_csv = results_dir / "timing_latency.csv"
    with open(_timing_csv, "w", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["method", "ms_per_frame"])
        _w.writerow(["SVD_matmul", f"{_dt_ms:.4f}"])
    print(f"  Timing saved to {_timing_csv}")

    print("=" * 70)
    print("2. Noise robustness sweep")
    print("=" * 70)
    df_noise = benchmark_noise_robustness(str(data_path), str(cnn_path), config, noise_levels=[1, 3, 5, 10, 20])
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
    data = load_dataset(str(data_path))
    N = sim_cfg["grid_size"]
    n_zernike = sim_cfg["n_zernike"]
    sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    basis = zernike_basis(n_zernike, N)
    modal_recon = ModalReconstructor(sensor, basis, n_zernike, config["reconstruction"]["svd_condition_number"])

    mmse_recon = MMSEReconstructor(
        modal_recon.modal_matrix,
        r0=config["turbulence"]["r0_m"],
        D=sim_cfg["aperture_diameter_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    valid_mask = sensor.get_valid_subaperture_mask()
    n_test = min(200, data["slopes"].shape[0])
    df_mmse = compare_svd_vs_mmse(
        data["slopes"][:n_test], data["zernike_coeffs"][:n_test], modal_recon, mmse_recon, valid_mask
    )
    print_benchmark_summary(df_mmse)
    save_benchmark_results(df_mmse, str(results_dir / "benchmark_svd_vs_mmse.csv"))

    print("=" * 70)
    print("5. Closed-loop simulation: integrator vs LQG vs LQG+prediction")
    print("=" * 70)
    pixel_scale = sim_cfg["aperture_diameter_m"] / N
    atmosphere = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=7)
    dm = DMController(config)

    n_cl_frames = min(200, sim_cfg["n_frames"])
    df_controllers = compare_controllers(atmosphere, sensor, dm, modal_recon, config, n_cl_frames)
    print_benchmark_summary(df_controllers)
    save_benchmark_results(df_controllers, str(results_dir / "benchmark_controllers.csv"))

    print("=" * 70)
    print("6. SLODAR Cn2(h) profiling")
    print("=" * 70)
    star_sep_rad = np.deg2rad(config["slodar"]["star_separation_arcsec"] / 3600.0)
    atmosphere2 = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=11)
    slopes1_seq, slopes2_seq = simulate_dual_star_slopes(atmosphere2, sensor, star_sep_rad, n_frames=50)

    profiler = SLODARProfiler(
        n_subapertures=sim_cfg["n_subapertures"],
        subaperture_pitch_m=sim_cfg["mla_pitch_m"],
        star_separation_rad=star_sep_rad,
        max_altitude_m=max(config["turbulence"]["layer_altitudes_m"]) * 1.2,
        n_bins=config["slodar"]["n_altitude_bins"],
    )
    cn2_profile = profiler.run(slopes1_seq, slopes2_seq)
    altitudes, cn2_profile_norm = profiler.fit_profile(cn2_profile, profiler.altitude_bins)
    validation = validate_slodar(
        cn2_profile_norm,
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
    print(f"Mean Strehl (integrator):   {df_controllers[df_controllers.controller == 'integrator'].strehl.mean():.4f}")
    print(f"Mean Strehl (LQG):          {df_controllers[df_controllers.controller == 'lqg'].strehl.mean():.4f}")
    print(f"Mean Strehl (LQG+pred):     {df_controllers[df_controllers.controller == 'lqg_predictive'].strehl.mean():.4f}")
    print(f"SLODAR recovery L2 error:   {validation['l2_error']:.4f}")


def run_demo(config: dict) -> None:
    """
    Run a short (50-frame) closed-loop simulation with all controllers,
    generate the core plots, and launch the dashboard.
    """
    # Check checkpoint early so the warning always prints (even if later imports fail)
    lstm_path = Path(config["paths"]["models_dir"]) / "temporal_model.pt"
    use_prediction = lstm_path.exists()
    if not use_prediction:
        print(
            f"WARNING: temporal model checkpoint not found at {lstm_path}; "
            "running predictive-AO branch with use_prediction=False "
            "(falling back to closed-loop without prediction)."
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

    sim_cfg = config["sim"]
    N = sim_cfg["grid_size"]
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
    basis = zernike_basis(sim_cfg["n_zernike"], N)
    modal_recon = ModalReconstructor(sensor, basis, sim_cfg["n_zernike"], config["reconstruction"]["svd_condition_number"])
    dm = DMController(config)

    device = torch.device("cpu")

    if use_prediction:
        model = load_trained_temporal_model(str(lstm_path), config, device)
        predictor = WavefrontPredictor(model, config["temporal"]["sequence_length"], device)
        print(f"Loaded temporal model from {lstm_path}")
    else:
        predictor = None

    closed_loop = ClosedLoopSimulator(atmosphere, sensor, dm, modal_recon, predictor, config)
    results = closed_loop.run(n_frames=50, use_prediction=use_prediction)

    t = np.arange(50) * sim_cfg["dt_s"]
    # Maréchal approximation: valid for sigma < 1 rad; underflows for large turbulence.
    # Clip to a display floor so it reads as "< 0.001" rather than "0.0000".
    strehl_open_raw = np.exp(-(results["rms_open_loop"] ** 2))
    strehl_open = np.maximum(strehl_open_raw, 1e-6)   # avoid printing 0.0000

    fig, ax = plt_subplots_helper()
    strehl_pred_arg = results["strehl_with_pred"] if use_prediction else None
    plot_strehl_timeseries(t, strehl_open, results["strehl_no_pred"], strehl_pred_arg, ax=ax)
    fig.savefig(results_dir / "demo_strehl_timeseries.png", dpi=120)

    print(f"Mean Strehl open loop:  {strehl_open.mean():.4e}  "
          f"(rms={results['rms_open_loop'].mean():.3f} rad — large turbulence, Marechal underflows)")
    print(f"Mean Strehl closed loop: {results['strehl_no_pred'].mean():.4f}")
    if use_prediction:
        print(f"Mean Strehl predictive:  {results['strehl_with_pred'].mean():.4f}")
    else:
        print("Mean Strehl predictive:  N/A (no trained checkpoint found)")

    dashboard_results = {
        "strehl_open": strehl_open,
        "strehl_closed": results["strehl_no_pred"],
        "strehl_pred": results["strehl_with_pred"] if use_prediction else results["strehl_no_pred"],
        "time": t,
    }
    # Panel 2: noise benchmarks
    import pandas as _pd
    _bf = Path(config["paths"]["results_dir"])
    _ncsv = _bf / "benchmark_noise.csv"
    if _ncsv.exists():
        _df = _pd.read_csv(_ncsv)
        dashboard_results["noise_levels"]  = _df["noise_level"].tolist()
        dashboard_results["rms_classical"] = _df["rms_classical"].tolist()
        dashboard_results["rms_cnn"]       = _df["rms_cnn"].tolist()
    # Panel 3: zernike variance from all_zernike
    if "all_zernike" in results:
        dashboard_results["zernike_variance"] = np.var(results["all_zernike"], axis=0).tolist()
    # Panel 4: r0 estimation from zernike history
    if "all_zernike" in results:
        from temporal.turbulence_param import TurbulenceParameterEstimator
        _est = TurbulenceParameterEstimator(config["sim"]["wavelength_m"], config["sim"]["aperture_diameter_m"], config["sim"]["dt_s"], config["sim"]["n_subapertures"])
        _params = _est.fit(results["all_zernike"])
        _r0_est = float(_params["r0_m"])
        _r0_true = float(config["turbulence"]["r0_m"])
        dashboard_results["r0_est"]  = [_r0_est]
        dashboard_results["r0_true"] = [_r0_true]
    # Panel 5: actuator commands
    if results.get("actuator_commands") is not None:
        dashboard_results["actuator_commands"] = results["actuator_commands"]
    # Panel 6: residual phase
    if results.get("last_residual_phase") is not None:
        dashboard_results["residual_phase"] = results["last_residual_phase"]
    import pandas as _pd
    _bf = Path(config["paths"]["results_dir"])
    _ncsv = _bf / "benchmark_noise.csv"
    if _ncsv.exists():
        _df = _pd.read_csv(_ncsv)
        dashboard_results["noise_levels"]  = _df["noise_level"].tolist()
        dashboard_results["rms_classical"] = _df["rms_classical"].tolist()
        dashboard_results["rms_cnn"]       = _df["rms_cnn"].tolist()
    _rcsv = _bf / "benchmark_reconstruction.csv"
    if _rcsv.exists():
        _df2 = _pd.read_csv(_rcsv)
        dashboard_results["zernike_variance"] = (_df2["rms_classical"].values ** 2).tolist()
    # Load benchmark CSVs into dashboard if they exist
    import pandas as _pd
    _rd = results_dir
    _bf = Path(config["paths"]["results_dir"])
    for _csv, _keys in [
        ("benchmark_noise.csv",       {"noise_levels": "noise_level", "rms_classical": "rms_classical", "rms_cnn": "rms_cnn"}),
        ("benchmark_reconstruction.csv", {"zernike_variance": None}),
        ("benchmark_dropout.csv",     {}),
    ]:
        _p = _bf / _csv
        if _p.exists():
            _df = _pd.read_csv(_p)
            if _csv == "benchmark_noise.csv":
                dashboard_results["noise_levels"]  = _df["noise_level"].tolist()
                dashboard_results["rms_classical"] = _df["rms_classical"].tolist()
                dashboard_results["rms_cnn"]       = _df["rms_cnn"].tolist()
            if _csv == "benchmark_reconstruction.csv" and "rms_classical" in _df.columns:
                dashboard_results["zernike_variance"] = _df["rms_classical"].values ** 2
    if "residual_phase" not in dashboard_results and "rms_open_loop" in results:
        dashboard_results["residual_phase"] = results.get("last_residual_phase",
            np.zeros((sim_cfg["grid_size"], sim_cfg["grid_size"])))
    if "actuator_commands" in results:
        dashboard_results["actuator_commands"] = results["actuator_commands"]
    fig_dashboard = create_dashboard(dashboard_results, config)
    export_dashboard_html(fig_dashboard, str(results_dir / "dashboard.html"))
    print(f"Dashboard saved to {results_dir / 'dashboard.html'}")
    # Export each panel as PNG
    import plotly.io as _pio
    _panel_titles = [
        "strehl_timeseries", "rms_vs_noise", "zernike_spectrum",
        "r0_accuracy", "actuator_commands", "residual_phase"
    ]
    for _i, _title in enumerate(_panel_titles):
        _row = _i // 3 + 1
        _col = _i % 3 + 1
        _pfig = create_dashboard(dashboard_results, config)
        _pfig.update_layout(
            annotations=[a for a in _pfig.layout.annotations if a.text == _pfig.layout.annotations[_i].text]
        )
        _pio.write_image(_pfig, str(results_dir / f"panel_{_title}.png"), width=600, height=400, scale=2)
        print(f"Saved panel_{_title}.png")
    # Also save full dashboard as PNG
    _pio.write_image(fig_dashboard, str(results_dir / "dashboard_full.png"), width=1200, height=800, scale=2)


def plt_subplots_helper():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(8, 4))


def run_generate_bmp(config: dict, n_frames: int, output_dir: str,
                     reference_output: str | None) -> None:
    """Generate synthetic SH-WFS BMP frames for pipeline testing."""
    from sim.generate_bmp_frames import generate_bmp_frames
    generate_bmp_frames(
        config,
        n_frames=n_frames,
        output_dir=output_dir,
        reference_output=reference_output,
        verbose=True,
    )


def run_real_data(config: dict, bmp_dir: str, reference: str | None = None) -> None:
    """
    Process a directory of real SH-WFS .bmp frames end-to-end:
      load → centroid → slopes → Zernike reconstruction → r0/tau0 → actuator map.
    """
    from data.load_real_frames import RealSHWFSLoader
    from reconstruction.zernike import zernike_basis
    from reconstruction.classical import ModalReconstructor
    from sim.shwfs import SHWFSSensor
    from actuator.dm_command import DMController
    from temporal.turbulence_param import TurbulenceParameterEstimator
    from viz.plot_utils import plot_phase_map

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time as _time_real

    sim_cfg = config["sim"]
    N = sim_cfg["grid_size"]
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading real BMP frames from: {bmp_dir}")
    loader = RealSHWFSLoader(config, bmp_dir=bmp_dir, reference_frame_path=reference)
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
    basis = zernike_basis(sim_cfg["n_zernike"], N)
    modal_recon = ModalReconstructor(
        sensor, basis, sim_cfg["n_zernike"],
        config["reconstruction"]["svd_condition_number"]
    )
    dm = DMController(config)
    mask = dm.mask

    all_zernike = np.zeros((n_frames, sim_cfg["n_zernike"]))
    all_phase_maps = np.zeros((n_frames, N, N))
    all_actuator_maps = []

    t0 = _time_real.perf_counter()
    for k in range(n_frames):
        z = modal_recon.reconstruct(slopes_x[k], slopes_y[k])
        all_zernike[k] = z
        phase = modal_recon.get_reconstructed_phase(z, basis)
        all_phase_maps[k] = phase
        cmd = dm.zernike_to_commands(z, basis, mask)
        all_actuator_maps.append(cmd)
    dt_ms = (_time_real.perf_counter() - t0) * 1000.0 / max(n_frames, 1)
    print(f"Reconstruction: {dt_ms:.3f} ms/frame")

    estimator = TurbulenceParameterEstimator(
        wavelength=sim_cfg["wavelength_m"],
        D=sim_cfg["aperture_diameter_m"],
        dt=sim_cfg["dt_s"],
        n_subapertures=sim_cfg["n_subapertures"],
    )
    params = estimator.fit(all_zernike)
    r0 = params["r0_m"]
    # FIX: compute per-mode tau0 and report the median (not tip/tilt only).
    # Tip/tilt has the longest tau0 and is not representative of the AO bandwidth.
    from profiling.temporal_psd import compute_tau0_per_mode
    _tau0_per_mode = compute_tau0_per_mode(all_zernike, sim_cfg["dt_s"])
    if len(_tau0_per_mode) > 0:
        tau0 = float(np.median(_tau0_per_mode))
    else:
        tau0 = 1.0 / params["greenwood_freq_hz"] if params["greenwood_freq_hz"] > 0 else 0.001
    print(f"\nTurbulence characterization:")
    print(f"  r0   = {r0:.4f} m  (Fried parameter)")
    print(f"  tau0 = {tau0*1000:.3f} ms  (coherence time)")

    np.savez(
        results_dir / "real_reconstruction.npz",
        zernike_coeffs=all_zernike,
        phase_maps=all_phase_maps,
        actuator_maps=np.stack(all_actuator_maps),
        r0_m=r0,
        tau0_s=tau0,
    )
    print(f"Saved reconstruction to {results_dir / 'real_reconstruction.npz'}")

    fig, axes = plt.subplots(1, min(3, n_frames), figsize=(12, 4))
    for idx, ax in enumerate(np.atleast_1d(axes)):
        plot_phase_map(all_phase_maps[idx], ax=ax, title=f"Frame {idx}")
    fig.suptitle(f"Reconstructed wavefront — r0={r0:.3f}m  tau0={tau0*1000:.1f}ms")
    fig.savefig(results_dir / "real_phase_maps.png", dpi=120, bbox_inches="tight")
    print(f"Phase maps saved to {results_dir / 'real_phase_maps.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BAH2026 AO Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str,
                        choices=["all", "sim", "train", "eval", "demo", "genbmp", "real"],
                        default="all")
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







