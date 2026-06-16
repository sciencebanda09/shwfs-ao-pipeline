"""
sim/generate_bmp_frames.py
==========================
Generate a time-series of synthetic SH-WFS frames as .bmp files,
suitable for testing the real-data ingestion pipeline before ISRO
provides actual laboratory frames.

Each .bmp frame is a (n_sub * pix_per_sub) x (n_sub * pix_per_sub)
grayscale image where each sub-aperture tile contains a simulated
diffraction-limited spot shifted by the local wavefront slope, with
photon and readout noise applied — identical to what a real science
camera would record.

The simulator reuses SHWFSSensor.simulate_spot_image() so the spot
PSF and centroid-shift scaling are physically consistent with the
reconstruction pipeline's assumptions.

Usage
-----
    # From repo root:
    python3 -m sim.generate_bmp_frames --config config.yaml \\
        --n_frames 200 --output_dir data/synthetic_bmp/ \\
        --reference_output data/synthetic_bmp/reference.bmp

    # Then ingest with the real-frame loader:
    python3 pipeline.py --config config.yaml --mode real \\
        --bmp_dir data/synthetic_bmp/ \\
        --reference data/synthetic_bmp/reference.bmp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _save_bmp(array: np.ndarray, path: str) -> None:
    """Save a 2D float array as an 8-bit grayscale BMP. Requires Pillow."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to write BMP files: pip install Pillow")
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(path)


def slopes_to_wfs_frame(
    sensor,
    slopes_x: np.ndarray,
    slopes_y: np.ndarray,
    flux_photons: float = 1000.0,
    readout_noise_e: float = 3.0,
    rng: np.random.Generator | None = None,
    scale: float = 200.0,
) -> np.ndarray:
    """
    Render a full SH-WFS detector frame from a (n_sub, n_sub) slope array.

    Uses SHWFSSensor.simulate_spot_image() for each valid subaperture
    so the PSF shape and centroid-shift relationship match the simulation
    pipeline exactly.

    Parameters
    ----------
    sensor : SHWFSSensor
    slopes_x, slopes_y : np.ndarray, shape (n_sub, n_sub), radians/pixel
    flux_photons : float
        Mean photon flux per subaperture per frame (for Poisson noise).
    readout_noise_e : float
        RMS readout noise in electrons (Gaussian).
    rng : np.random.Generator, optional
        For reproducibility. Uses default_rng() if None.
    scale : float
        Peak DN value of a noiseless spot (maps normalised PSF to 8-bit range).

    Returns
    -------
    frame : np.ndarray, shape (n_sub*pix_per_sub, n_sub*pix_per_sub), float32
        Raw detector frame in DN (digital numbers), NOT yet clipped to uint8.
        Caller clips and saves as BMP.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sub = sensor.n_sub
    pps = sensor.pix_per_sub
    H = W = n_sub * pps
    frame = np.zeros((H, W), dtype=np.float64)

    valid = sensor.get_valid_subaperture_mask()

    for i in range(n_sub):
        for j in range(n_sub):
            y0 = i * pps
            x0 = j * pps

            if not valid[i, j]:
                # Dark subaperture — just readout noise
                noise = rng.normal(0.0, readout_noise_e, (pps, pps))
                frame[y0:y0+pps, x0:x0+pps] = noise
                continue

            sx = float(slopes_x[i, j])
            sy = float(slopes_y[i, j])

            # Physically correct spot PSF via FFT propagation in SHWFSSensor
            spot_norm = sensor.simulate_spot_image(sx, sy)  # shape (pps, pps), sums to 1

            # Scale to photon counts
            spot_photons = spot_norm * flux_photons

            # Poisson (shot) noise
            spot_noisy = rng.poisson(np.maximum(spot_photons, 0)).astype(np.float64)

            # Readout noise (Gaussian, RMS = readout_noise_e)
            spot_noisy += rng.normal(0.0, readout_noise_e, (pps, pps))

            # Rescale from photon counts to 8-bit DN range
            # (so peak≈scale DN for a bright spot)
            spot_dn = spot_noisy * (scale / max(flux_photons, 1.0))

            frame[y0:y0+pps, x0:x0+pps] = spot_dn

    return frame.astype(np.float32)


def generate_reference_frame(sensor, config: dict, scale: float = 200.0) -> np.ndarray:
    """
    Generate a flat (zero-slope) reference WFS frame with minimal noise.

    This mimics what you would record in the lab by pointing at a point
    source through a flat wavefront (e.g. using an internal calibration
    source).

    Returns
    -------
    frame : np.ndarray, shape (n_sub*pps, n_sub*pps), float32
    """
    n_sub = sensor.n_sub
    pps = sensor.pix_per_sub
    slopes_zero_x = np.zeros((n_sub, n_sub), dtype=np.float32)
    slopes_zero_y = np.zeros((n_sub, n_sub), dtype=np.float32)
    noise_cfg = config.get("noise", {})
    # Low noise for reference frame (10x more photons, half readout noise)
    return slopes_to_wfs_frame(
        sensor,
        slopes_zero_x,
        slopes_zero_y,
        flux_photons=noise_cfg.get("flux_photons_per_frame", 1000) * 10,
        readout_noise_e=noise_cfg.get("readout_noise_e", 3.0) * 0.5,
        rng=np.random.default_rng(0),
        scale=scale,
    )


def generate_bmp_frames(
    config: dict,
    n_frames: int,
    output_dir: str,
    reference_output: str | None = None,
    seed: int = 42,
    scale: float = 200.0,
    verbose: bool = True,
) -> None:
    """
    Generate a full time-series of synthetic SH-WFS BMP frames.

    Steps
    -----
    1. Simulate atmospheric turbulence (same MultiLayerAtmosphere as pipeline).
    2. Propagate each phase screen through SHWFSSensor to get ideal slopes.
    3. Apply photon + readout noise using slopes_to_wfs_frame().
    4. Render each frame as a grayscale BMP.
    5. Optionally write a clean reference frame.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    n_frames : int
        Number of frames to generate.
    output_dir : str
        Output directory for frame BMP files (frame_0000.bmp, frame_0001.bmp, ...).
    reference_output : str, optional
        Path to write the flat reference frame BMP.
        If None, no reference frame is written.
    seed : int
        RNG seed for reproducibility.
    scale : float
        Peak DN value for spot rendering (200 = good for 8-bit).
    verbose : bool
    """
    from sim.shwfs import SHWFSSensor
    from sim.turbulence import build_atmosphere_from_config
    from sim.phase_screen import apply_aperture_mask

    sim_cfg = config["sim"]
    noise_cfg = config.get("noise", {})
    N = sim_cfg["grid_size"]
    pixel_scale = sim_cfg["aperture_diameter_m"] / N

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    np_seed = int(rng.integers(0, 2**31))
    np.random.seed(np_seed)

    atmosphere = build_atmosphere_from_config(config, N=N, pixel_scale=pixel_scale, seed=seed)
    sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )

    dt = sim_cfg["dt_s"]
    flux = noise_cfg.get("flux_photons_per_frame", 1000)
    readout = noise_cfg.get("readout_noise_e", 3.0)

    if verbose:
        print(f"Generating {n_frames} synthetic SH-WFS BMP frames → {out_path}/")
        print(f"  n_sub={sim_cfg['n_subapertures']}, pix_per_sub={sim_cfg['detector_pixels_per_subaperture']}")
        print(f"  r0={config['turbulence']['r0_m']}m, dt={dt*1000:.1f}ms, flux={flux} ph/sub/frame")

    # Optional reference frame (zero slopes, low noise)
    if reference_output is not None:
        ref_frame = generate_reference_frame(sensor, config, scale=scale)
        _save_bmp(ref_frame, reference_output)
        if verbose:
            print(f"  Reference frame → {reference_output}")

    # Time-series frames
    frame_rng = np.random.default_rng(seed + 1)
    for k in range(n_frames):
        atmosphere.evolve(dt)
        phase = apply_aperture_mask(atmosphere.get_integrated_phase_radians(), "circular")
        slopes_x, slopes_y = sensor.propagate(phase)

        frame = slopes_to_wfs_frame(
            sensor, slopes_x, slopes_y,
            flux_photons=flux,
            readout_noise_e=readout,
            rng=frame_rng,
            scale=scale,
        )

        bmp_path = out_path / f"frame_{k:04d}.bmp"
        _save_bmp(frame, str(bmp_path))

        if verbose and (k == 0 or (k + 1) % 50 == 0 or k == n_frames - 1):
            print(f"  [{k+1}/{n_frames}] {bmp_path.name}  "
                  f"slope_rms_x={slopes_x[sensor.valid_mask].std():.4f} rad/px")

    if verbose:
        frame_h = sim_cfg["n_subapertures"] * sim_cfg["detector_pixels_per_subaperture"]
        print(f"\nDone. Each frame: {frame_h}×{frame_h} px, 8-bit grayscale BMP.")
        print(f"Ingest with:")
        print(f"  python3 pipeline.py --config config.yaml --mode real \\")
        print(f"      --bmp_dir {output_dir}/ \\")
        if reference_output:
            print(f"      --reference {reference_output}")


# ---------------------------------------------------------------------------
# CLI: python3 -m sim.generate_bmp_frames --config config.yaml --n_frames 200
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic SH-WFS BMP frames for pipeline testing"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--n_frames", type=int, default=200, help="Number of frames to generate")
    parser.add_argument("--output_dir", default="data/synthetic_bmp",
                        help="Output directory for BMP files")
    parser.add_argument("--reference_output", default="data/synthetic_bmp/reference.bmp",
                        help="Path for the flat reference BMP (pass '' to skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=200.0,
                        help="Peak DN value for spot rendering (default 200)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ref_out = args.reference_output if args.reference_output else None
    generate_bmp_frames(
        cfg,
        n_frames=args.n_frames,
        output_dir=args.output_dir,
        reference_output=ref_out,
        seed=args.seed,
        scale=args.scale,
        verbose=True,
    )
