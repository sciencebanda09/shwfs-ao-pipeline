"""Unit tests for the real BMP frame loader (no actual BMP files needed)."""
import numpy as np
import pytest
import yaml
import os
import tempfile


def _make_fake_bmp(path: str, n_sub: int, pix_per_sub: int, noise_std: float = 5.0):
    """Write a synthetic SH-WFS BMP frame using PIL."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    h = w = n_sub * pix_per_sub
    frame = np.zeros((h, w), dtype=np.uint8)

    for i in range(n_sub):
        for j in range(n_sub):
            y0 = i * pix_per_sub
            x0 = j * pix_per_sub
            cy_c = y0 + pix_per_sub // 2
            cx_c = x0 + pix_per_sub // 2
            for dy in range(pix_per_sub):
                for dx in range(pix_per_sub):
                    r2 = (y0 + dy - cy_c) ** 2 + (x0 + dx - cx_c) ** 2
                    frame[y0 + dy, x0 + dx] = min(255, int(200 * np.exp(-r2 / 2.0)))

    rng = np.random.default_rng(42)
    frame = np.clip(
        frame.astype(np.float32) + rng.normal(0, noise_std, frame.shape), 0, 255
    ).astype(np.uint8)
    Image.fromarray(frame, mode="L").save(path)


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_loader_basic():
    config = _load_config()
    n_sub = config["sim"]["n_subapertures"]
    pps = config["sim"]["detector_pixels_per_subaperture"]

    with tempfile.TemporaryDirectory() as tmp:
        n_frames = 5
        for k in range(n_frames):
            _make_fake_bmp(os.path.join(tmp, f"frame_{k:04d}.bmp"), n_sub, pps)

        from data.load_real_frames import RealSHWFSLoader
        loader = RealSHWFSLoader(config, bmp_dir=tmp)
        sx, sy = loader.process_all(verbose=False)

    assert sx.shape == (n_frames, n_sub, n_sub)
    assert sy.shape == (n_frames, n_sub, n_sub)
    assert np.isfinite(sx).all()
    assert np.isfinite(sy).all()


def test_loader_centroid_near_zero_for_flat():
    config = _load_config()
    n_sub = config["sim"]["n_subapertures"]
    pps = config["sim"]["detector_pixels_per_subaperture"]

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "reference.bmp")
        _make_fake_bmp(ref_path, n_sub, pps, noise_std=0.0)

        for k in range(3):
            _make_fake_bmp(os.path.join(tmp, f"frame_{k:04d}.bmp"), n_sub, pps, noise_std=0.0)

        from data.load_real_frames import RealSHWFSLoader
        loader = RealSHWFSLoader(config, bmp_dir=tmp, reference_frame_path=ref_path)
        sx, sy = loader.process_all(verbose=False)

    assert np.abs(sx).max() < 1e-3, f"max sx={np.abs(sx).max():.6f}"
    assert np.abs(sy).max() < 1e-3, f"max sy={np.abs(sy).max():.6f}"


def test_wcog_centroiding():
    from data.load_real_frames import centroid_wcog
    pps = 8
    tile = np.zeros((pps, pps), dtype=np.float32)
    tile[3, 4] = 100.0
    cx, cy = centroid_wcog(tile)
    assert np.isfinite(cx) and np.isfinite(cy)
    assert 3.5 < cx < 4.5, f"cx={cx}"
