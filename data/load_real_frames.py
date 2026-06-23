"""
data/load_real_frames.py
========================
Load a time-series of real Shack-Hartmann WFS frames from .bmp files,
extract sub-aperture spot images, compute centroids, and convert to
slope arrays in the same format used by the simulation pipeline.

Usage
-----
    from data.load_real_frames import RealSHWFSLoader
    loader = RealSHWFSLoader(config, bmp_dir="path/to/bmps")
    slopes_x, slopes_y = loader.process_all()   # (n_frames, n_sub, n_sub) each
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _load_bmp(path: str) -> np.ndarray:
    """Load a BMP file as a 2D float32 array (grayscale). Tries PIL then OpenCV."""
    if _PIL_AVAILABLE:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32)
    if _CV2_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"cv2 could not read: {path}")
        return img.astype(np.float32)
    raise ImportError(
        "Install Pillow (`pip install Pillow`) or OpenCV (`pip install opencv-python`) "
        "to load BMP files."
    )


def centroid_cog_threshold(tile: np.ndarray, threshold_sigma: float = 3.0) -> tuple[float, float]:
    """
    Centre-of-gravity centroid with background subtraction and threshold.

    1. Estimate background from the 4 corner pixels.
    2. Subtract background; clip to zero.
    3. Zero out pixels below threshold_sigma * background_std.
    4. Compute intensity-weighted centroid.

    Returns (cx, cy) in pixel coordinates within the tile.
    Returns tile centre if the tile is empty or all background.
    """
    rows, cols = tile.shape
    cx_default, cy_default = (cols - 1) / 2.0, (rows - 1) / 2.0

    corners = np.array([tile[0, 0], tile[0, -1], tile[-1, 0], tile[-1, -1]], dtype=np.float64)
    bg_mean = corners.mean()
    bg_std = max(corners.std(), 1.0)   # floor at 1 DN to avoid division by zero

    img = (tile.astype(np.float64) - bg_mean).clip(0)
    mask = img > (threshold_sigma * bg_std)
    img = img * mask

    total = img.sum()
    if total <= 0:
        return cx_default, cy_default

    y_idx, x_idx = np.indices(tile.shape)
    cx = float((img * x_idx).sum() / total)
    cy = float((img * y_idx).sum() / total)
    return cx, cy


def centroid_wcog(tile: np.ndarray, weight_fwhm_px: float = 2.0) -> tuple[float, float]:
    """
    Weighted centre-of-gravity centroid using a Gaussian weight centred on the
    brightest pixel. More robust than plain CoG for low-SNR spots.

    Returns (cx, cy) in pixel coordinates within the tile.
    """
    rows, cols = tile.shape
    cx_default, cy_default = (cols - 1) / 2.0, (rows - 1) / 2.0

    img = tile.astype(np.float64).clip(0)
    if img.max() <= 0:
        return cx_default, cy_default

    total0 = img.sum()
    if total0 <= 0:
        return cx_default, cy_default
    y_idx, x_idx = np.indices(tile.shape)
    cx0 = float((img * x_idx).sum() / total0)
    cy0 = float((img * y_idx).sum() / total0)

    sigma = weight_fwhm_px / (2 * np.sqrt(2 * np.log(2)))
    W = np.exp(-((x_idx - cx0) ** 2 + (y_idx - cy0) ** 2) / (2 * sigma ** 2))
    weighted = img * W
    total_w = weighted.sum()
    if total_w <= 0:
        return cx_default, cy_default

    cx = float((weighted * x_idx).sum() / total_w)
    cy = float((weighted * y_idx).sum() / total_w)
    return cx, cy


class RealSHWFSLoader:
    """
    Load real SH-WFS frames from .bmp files and produce slope arrays.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.  Keys used:
          sim.n_subapertures, sim.detector_pixels_per_subaperture,
          sim.mla_pitch_m, sim.focal_length_m,
          noise.centroiding_method ('cog' | 'wcog')
    bmp_dir : str
        Directory containing .bmp frame files (sorted alphabetically = time order).
    reference_frame_path : str, optional
        Path to a flat (unaberrated) reference .bmp frame.  If None, the
        mean of all frames is used as the reference.
    pixel_size_m : float, optional
        Physical detector pixel size in meters.  Used to convert centroid
        displacements (pixels) to slopes (rad/m).  If None, defaults to
        mla_pitch / pixels_per_subaperture.
    """

    def __init__(
        self,
        config: dict,
        bmp_dir: str,
        reference_frame_path: Optional[str] = None,
        pixel_size_m: Optional[float] = None,
    ):
        sim_cfg = config["sim"]
        noise_cfg = config.get("noise", {})

        self.n_sub = sim_cfg["n_subapertures"]
        self.pix_per_sub = sim_cfg["detector_pixels_per_subaperture"]
        self.pitch_m = sim_cfg["mla_pitch_m"]
        self.focal_length_m = sim_cfg["focal_length_m"]
        self.centroiding_method = noise_cfg.get("centroiding_method", "cog")

        self.pixel_size_m = pixel_size_m or (self.pitch_m / self.pix_per_sub)

        # px_to_rad converts centroid displacement (subaperture pixels) to slope
        # in the same units as SHWFSSensor.propagate() — i.e. rad per full-phase-
        # grid index.  propagate() calls np.gradient on the (N x N) phase screen,
        # so its slopes are in rad / (D/N) meters = rad * (N/D) per meter.
        # simulate_spot_image() shifts a spot by slope * pix_per_sub / (2*pi)
        # subaperture pixels for a slope in rad/subaperture-pixel.  Converting:
        #   slope [rad/full-grid-index] = slope [rad/sub-px] * (sub-px-m / grid-px-m)
        #                               = slope [rad/sub-px] * pitch_m*N / (pix_per_sub*D_m)
        # and slope [rad/sub-px] = centroid_disp_px * 2*pi / pix_per_sub
        # => px_to_rad = (2*pi / pps) * pitch_m * N / (pps * D_m)
        #              = 2*pi * pitch_m * N / (pps^2 * D_m)
        N_grid = sim_cfg.get("grid_size", self.n_sub * self.pix_per_sub)
        D_m    = sim_cfg.get("aperture_diameter_m", self.pitch_m * self.n_sub)
        self.px_to_rad = (
            2.0 * np.pi * self.pitch_m * N_grid
            / (self.pix_per_sub ** 2 * D_m)
        )

        self.bmp_dir = Path(bmp_dir)
        self.reference_frame_path = reference_frame_path

        self.frame_paths = sorted(self.bmp_dir.glob("*.bmp"))
        if not self.frame_paths:
            self.frame_paths = sorted(self.bmp_dir.glob("*.BMP"))
        if not self.frame_paths:
            raise FileNotFoundError(f"No .bmp files found in {bmp_dir}")

        # Exclude reference frame from time-series if it lives in the same dir
        if reference_frame_path is not None:
            ref_p = Path(reference_frame_path).resolve()
            self.frame_paths = [p for p in self.frame_paths if p.resolve() != ref_p]

        coords = (np.arange(self.n_sub) + 0.5) / self.n_sub * 2 - 1
        cx, cy = np.meshgrid(coords, coords)
        self.valid_mask = (cx ** 2 + cy ** 2) <= 1.0

        self._ref_centroids: Optional[np.ndarray] = None

        # Optional C extension for fast centroiding
        try:
            import centroid_cog as _c_cog
            self._c_ext = _c_cog
            print("C centroiding extension loaded — using fast C CoG path.")
        except ImportError:
            self._c_ext = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_all(self, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Process every .bmp frame in bmp_dir and return slope arrays.

        Returns
        -------
        slopes_x : np.ndarray, shape (n_frames, n_sub, n_sub), radians
        slopes_y : np.ndarray, shape (n_frames, n_sub, n_sub), radians
            Zero for invalid (vignetted) subapertures.
        """
        n_frames = len(self.frame_paths)
        if verbose:
            print(f"Found {n_frames} BMP frames in {self.bmp_dir}")

        self._ref_centroids = self._build_reference_centroids(verbose)

        slopes_x = np.zeros((n_frames, self.n_sub, self.n_sub), dtype=np.float32)
        slopes_y = np.zeros((n_frames, self.n_sub, self.n_sub), dtype=np.float32)

        t0 = time.perf_counter()
        for k, path in enumerate(self.frame_paths):
            frame = _load_bmp(str(path))
            sx, sy = self._frame_to_slopes(frame)
            slopes_x[k] = sx
            slopes_y[k] = sy

        dt_ms = (time.perf_counter() - t0) * 1000.0 / max(n_frames, 1)
        if verbose:
            print(f"Centroiding throughput: {dt_ms:.2f} ms/frame  "
                  f"({'PASS' if dt_ms < 10 else 'WARN — exceeds 10 ms target'})")

        return slopes_x, slopes_y

    def process_frame(self, bmp_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Process a single BMP frame. Reference must already be built."""
        if self._ref_centroids is None:
            self._ref_centroids = self._build_reference_centroids(verbose=False)
        frame = _load_bmp(bmp_path)
        return self._frame_to_slopes(frame)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_tile(self, frame: np.ndarray, i: int, j: int) -> np.ndarray:
        """Extract the (i, j) subaperture tile from a full WFS frame."""
        h, w = frame.shape
        tile_h = h // self.n_sub
        tile_w = w // self.n_sub
        y0, y1 = i * tile_h, (i + 1) * tile_h
        x0, x1 = j * tile_w, (j + 1) * tile_w
        tile = frame[y0:y1, x0:x1]
        if tile.shape != (self.pix_per_sub, self.pix_per_sub):
            if _CV2_AVAILABLE:
                tile = cv2.resize(tile, (self.pix_per_sub, self.pix_per_sub),
                                  interpolation=cv2.INTER_AREA)
            elif _PIL_AVAILABLE:
                pil_tile = Image.fromarray(tile).resize(
                    (self.pix_per_sub, self.pix_per_sub), Image.LANCZOS
                )
                tile = np.array(pil_tile, dtype=np.float32)
        return tile

    def _centroid(self, tile: np.ndarray) -> tuple[float, float]:
        """Dispatch to selected centroiding algorithm."""
        if self.centroiding_method == "wcog":
            return centroid_wcog(tile)
        return centroid_cog_threshold(tile)

    def _build_reference_centroids(self, verbose: bool = True) -> np.ndarray:
        """
        Compute reference (unaberrated) centroid positions.

        If reference_frame_path is given, use that frame.
        Otherwise use the mean of all frames as a proxy flat reference.

        Returns
        -------
        ref : np.ndarray, shape (n_sub, n_sub, 2), pixel coordinates
        """
        if self.reference_frame_path is not None and Path(self.reference_frame_path).exists():
            if verbose:
                print(f"Using reference frame: {self.reference_frame_path}")
            ref_frame = _load_bmp(self.reference_frame_path)
        else:
            if verbose:
                print("No reference frame provided — using mean of all frames as flat reference.")
            ref_frame = np.zeros_like(_load_bmp(str(self.frame_paths[0])), dtype=np.float64)
            for p in self.frame_paths:
                ref_frame += _load_bmp(str(p)).astype(np.float64)
            ref_frame = (ref_frame / len(self.frame_paths)).astype(np.float32)

        ref = np.zeros((self.n_sub, self.n_sub, 2), dtype=np.float64)
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                if not self.valid_mask[i, j]:
                    continue
                tile = self._extract_tile(ref_frame, i, j)
                cx, cy = self._centroid(tile)
                ref[i, j, 0] = cx
                ref[i, j, 1] = cy
        return ref

    def _frame_to_slopes(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert one WFS frame to (slopes_x, slopes_y) arrays in radians.

        slope_x[i,j] = (centroid_x[i,j] - ref_x[i,j]) * px_to_rad
        slope_y[i,j] = (centroid_y[i,j] - ref_y[i,j]) * px_to_rad
        """
        sx = np.zeros((self.n_sub, self.n_sub), dtype=np.float32)
        sy = np.zeros((self.n_sub, self.n_sub), dtype=np.float32)

        # Fast path: C extension for CoG (all subapertures in one C call)
        if self._c_ext is not None and self.centroiding_method == "cog":
            frame_f32 = frame.astype(np.float32)
            cx_list, cy_list = self._c_ext.cog_batch(
                frame_f32.tobytes(), self.n_sub, self.pix_per_sub, 3.0
            )
            cx_arr = np.array(cx_list, dtype=np.float64).reshape(self.n_sub, self.n_sub)
            cy_arr = np.array(cy_list, dtype=np.float64).reshape(self.n_sub, self.n_sub)
            for i in range(self.n_sub):
                for j in range(self.n_sub):
                    if not self.valid_mask[i, j]:
                        continue
                    sx[i, j] = (cx_arr[i, j] - self._ref_centroids[i, j, 0]) * self.px_to_rad
                    sy[i, j] = (cy_arr[i, j] - self._ref_centroids[i, j, 1]) * self.px_to_rad
            return sx, sy

        # Fallback: pure Python per-tile loop
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                if not self.valid_mask[i, j]:
                    continue
                tile = self._extract_tile(frame, i, j)
                cx, cy = self._centroid(tile)
                sx[i, j] = (cx - self._ref_centroids[i, j, 0]) * self.px_to_rad
                sy[i, j] = (cy - self._ref_centroids[i, j, 1]) * self.px_to_rad
        return sx, sy


# ---------------------------------------------------------------------------
# CLI: python3 -m data.load_real_frames --bmp_dir ./frames/ --config config.yaml
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Process real SH-WFS BMP frames")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bmp_dir", required=True, help="Directory of .bmp WFS frames")
    parser.add_argument("--reference", default=None, help="Path to flat reference .bmp frame")
    parser.add_argument("--output", default="data/real_slopes.npz", help="Output .npz path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = RealSHWFSLoader(config, bmp_dir=args.bmp_dir, reference_frame_path=args.reference)
    sx, sy = loader.process_all(verbose=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, slopes_x=sx, slopes_y=sy)
    print(f"Saved slopes to {args.output}  shape: {sx.shape}")
