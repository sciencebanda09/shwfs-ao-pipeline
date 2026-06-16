"""Unit tests for synthetic BMP frame generator."""
import numpy as np
import pytest
import yaml
import tempfile
import os
from pathlib import Path


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_slopes_to_wfs_frame_shape():
    """Frame shape matches n_sub * pix_per_sub."""
    config = _load_config()
    from sim.shwfs import SHWFSSensor
    from sim.generate_bmp_frames import slopes_to_wfs_frame

    sim_cfg = config["sim"]
    sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    n_sub = sim_cfg["n_subapertures"]
    pps = sim_cfg["detector_pixels_per_subaperture"]
    sx = np.zeros((n_sub, n_sub), dtype=np.float32)
    sy = np.zeros((n_sub, n_sub), dtype=np.float32)

    frame = slopes_to_wfs_frame(sensor, sx, sy, flux_photons=500, readout_noise_e=3.0,
                                 rng=np.random.default_rng(0))
    expected_size = n_sub * pps
    assert frame.shape == (expected_size, expected_size), \
        f"Expected ({expected_size},{expected_size}), got {frame.shape}"


def test_slopes_to_wfs_frame_finite():
    """All frame pixels must be finite."""
    config = _load_config()
    from sim.shwfs import SHWFSSensor
    from sim.generate_bmp_frames import slopes_to_wfs_frame

    sim_cfg = config["sim"]
    sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    n_sub = sim_cfg["n_subapertures"]
    sx = np.random.default_rng(1).uniform(-0.05, 0.05, (n_sub, n_sub)).astype(np.float32)
    sy = np.random.default_rng(2).uniform(-0.05, 0.05, (n_sub, n_sub)).astype(np.float32)

    frame = slopes_to_wfs_frame(sensor, sx, sy, flux_photons=500, readout_noise_e=3.0,
                                 rng=np.random.default_rng(3))
    assert np.isfinite(frame).all(), "Frame contains non-finite values"


def test_reference_frame_spots_brighter_than_turbulence_frame():
    """Reference frame (zero slope, 10x flux) should have higher mean DN than a turbulence frame."""
    config = _load_config()
    from sim.shwfs import SHWFSSensor
    from sim.generate_bmp_frames import slopes_to_wfs_frame, generate_reference_frame

    sim_cfg = config["sim"]
    sensor = SHWFSSensor(
        n_subapertures=sim_cfg["n_subapertures"],
        pixels_per_subaperture=sim_cfg["detector_pixels_per_subaperture"],
        focal_length=sim_cfg["focal_length_m"],
        pitch=sim_cfg["mla_pitch_m"],
        wavelength=sim_cfg["wavelength_m"],
    )
    n_sub = sim_cfg["n_subapertures"]

    ref_frame = generate_reference_frame(sensor, config, scale=200.0)
    sx = np.random.default_rng(10).uniform(-0.05, 0.05, (n_sub, n_sub)).astype(np.float32)
    sy = np.random.default_rng(11).uniform(-0.05, 0.05, (n_sub, n_sub)).astype(np.float32)
    turb_frame = slopes_to_wfs_frame(sensor, sx, sy, flux_photons=1000, readout_noise_e=3.0,
                                      rng=np.random.default_rng(12), scale=200.0)

    # Reference uses 10x flux → should be brighter on average in valid subaperture centres
    pps = sim_cfg["detector_pixels_per_subaperture"]
    valid = sensor.get_valid_subaperture_mask()
    ref_vals, turb_vals = [], []
    for i in range(n_sub):
        for j in range(n_sub):
            if not valid[i, j]:
                continue
            y0, x0 = i * pps, j * pps
            ref_vals.append(ref_frame[y0:y0+pps, x0:x0+pps].max())
            turb_vals.append(turb_frame[y0:y0+pps, x0:x0+pps].max())
    ref_mean = float(np.mean(ref_vals))
    turb_mean = float(np.mean(turb_vals))

    assert ref_mean > turb_mean, \
        f"Reference peak mean ({ref_mean:.2f}) should exceed turbulence peak mean ({turb_mean:.2f})"


def test_generate_bmp_frames_writes_files():
    """generate_bmp_frames() creates the expected number of BMP files."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    config = _load_config()
    from sim.generate_bmp_frames import generate_bmp_frames

    n_frames = 5
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "reference.bmp")
        generate_bmp_frames(
            config,
            n_frames=n_frames,
            output_dir=tmp,
            reference_output=ref_path,
            seed=99,
            verbose=False,
        )
        bmp_files = sorted(Path(tmp).glob("frame_*.bmp"))
        assert len(bmp_files) == n_frames, \
            f"Expected {n_frames} BMP files, found {len(bmp_files)}"
        assert Path(ref_path).exists(), "Reference BMP not written"

        # Each file must be readable and the right shape
        sim_cfg = config["sim"]
        expected_size = sim_cfg["n_subapertures"] * sim_cfg["detector_pixels_per_subaperture"]
        for bmp in bmp_files:
            img = np.array(Image.open(str(bmp)))
            assert img.shape == (expected_size, expected_size), \
                f"{bmp.name}: wrong shape {img.shape}, expected ({expected_size},{expected_size})"
            assert img.dtype == np.uint8


def test_round_trip_zero_slope():
    """
    Round-trip: generate zero-slope BMP frames, load with RealSHWFSLoader,
    slopes should be near zero (within noise tolerance).
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    config = _load_config()
    from sim.generate_bmp_frames import generate_bmp_frames

    n_frames = 10
    with tempfile.TemporaryDirectory() as tmp:
        # Override turbulence so slopes are tiny (set r0 very large → weak turbulence)
        import copy
        cfg = copy.deepcopy(config)
        cfg["turbulence"]["r0_m"] = 5.0  # very weak turbulence → slopes ≈ 0
        cfg["noise"]["flux_photons_per_frame"] = 10000  # high SNR

        ref_path = os.path.join(tmp, "reference.bmp")
        generate_bmp_frames(cfg, n_frames=n_frames, output_dir=tmp,
                             reference_output=ref_path, seed=7, verbose=False)

        from data.load_real_frames import RealSHWFSLoader
        loader = RealSHWFSLoader(cfg, bmp_dir=tmp, reference_frame_path=ref_path)
        sx, sy = loader.process_all(verbose=False)

    assert sx.shape[0] == n_frames
    # With very large r0, slopes should be small in magnitude (< 0.5 rad)
    valid_mask = loader.valid_mask
    slope_rms = np.sqrt((sx[:, valid_mask]**2 + sy[:, valid_mask]**2).mean())
    assert slope_rms < 0.5, f"Expected small slopes for r0=5m, got RMS={slope_rms:.4f} rad"
