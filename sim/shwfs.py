"""
sim/shwfs.py
============
Shack-Hartmann Wavefront Sensor (SH-WFS) simulation: subaperture
geometry, spot propagation, and centroid-based slope measurement.
"""

from __future__ import annotations

import numpy as np

from sim.noise import centroid_cog


def _k_shift(n: int) -> float:
    """
    Centroid-shift conversion constant K_SHIFT (pixels per rad/pixel tilt).

    With the 4x oversampled FFT in SHWFSSensor.simulate_spot_image, a
    linear phase ramp of magnitude t (rad/pixel) over an n-pixel pupil
    shifts the focal-plane spot by t * K_SHIFT pixels, where

        K_SHIFT = n / (2*pi)

    This follows directly from the discrete Fourier shift theorem.
    """
    return n / (2.0 * np.pi)


class SHWFSSensor:
    """
    Shack-Hartmann Wavefront Sensor.

    Parameters
    ----------
    n_subapertures : int
        Number of subapertures across the pupil diameter.
    pixels_per_subaperture : int
        Detector pixels per subaperture side.
    focal_length : float
        Microlens focal length (m).
    pitch : float
        Subaperture pitch (m).
    wavelength : float
        Operating wavelength (m).
    """

    def __init__(
        self,
        n_subapertures: int,
        pixels_per_subaperture: int,
        focal_length: float,
        pitch: float,
        wavelength: float,
    ):
        self.n_sub = n_subapertures
        self.pix_per_sub = pixels_per_subaperture
        self.focal_length = focal_length
        self.pitch = pitch
        self.wavelength = wavelength

        # Subaperture center coordinates in normalized aperture units [-1,1]
        coords = (np.arange(n_subapertures) + 0.5) / n_subapertures * 2 - 1
        self.cx, self.cy = np.meshgrid(coords, coords)

        # Valid subapertures: centers lying within the unit circle
        self.valid_mask = (self.cx ** 2 + self.cy ** 2) <= 1.0

        self._reference_spots = self.generate_reference_spots()

    def generate_reference_spots(self) -> np.ndarray:
        """
        Return reference spot centroid positions for every subaperture.

        Returns
        -------
        ref_spots : np.ndarray, shape (n_sub, n_sub, 2)
            Each entry is the (x, y) pixel coordinate of the subaperture
            center on its local detector tile.
        """
        center = (self.pix_per_sub - 1) / 2.0
        ref = np.full((self.n_sub, self.n_sub, 2), center, dtype=float)
        return ref

    def get_valid_subaperture_mask(self) -> np.ndarray:
        """Boolean mask of valid (illuminated) subapertures."""
        return self.valid_mask

    def get_subaperture_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (x, y) center positions of all valid subapertures in
        normalized aperture coordinates [-1, 1].
        """
        return self.cx[self.valid_mask], self.cy[self.valid_mask]

    def propagate(self, phase_screen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate a phase screen through the SH-WFS and compute slope
        measurements for every subaperture.

        For each valid subaperture:
          1. Extract the local phase tile.
          2. Compute the local mean gradient (tilt).
          3. Simulate the focal-plane spot via FFT of the tilted pupil.
          4. Compute the spot centroid (centre-of-gravity).

        Parameters
        ----------
        phase_screen : np.ndarray, shape (N, N), radians

        Returns
        -------
        slopes_x, slopes_y : np.ndarray, shape (n_sub, n_sub)
            Local wavefront slopes (radians per meter), zero for invalid
            subapertures.
        """
        N = phase_screen.shape[0]
        tile = N // self.n_sub

        slopes_x = np.zeros((self.n_sub, self.n_sub))
        slopes_y = np.zeros((self.n_sub, self.n_sub))

        for i in range(self.n_sub):
            for j in range(self.n_sub):
                if not self.valid_mask[i, j]:
                    continue

                y0, y1 = i * tile, (i + 1) * tile
                x0, x1 = j * tile, (j + 1) * tile
                sub_phase = phase_screen[y0:y1, x0:x1]

                # Local tilt = mean gradient of phase across the tile
                grad_y, grad_x = np.gradient(sub_phase)
                slopes_x[i, j] = np.mean(grad_x)
                slopes_y[i, j] = np.mean(grad_y)

        return slopes_x, slopes_y

    # Oversampling factor for zero-padded FFT.  4x gives sub-pixel
    # accuracy over the realistic tilt range (|tilt| < ~0.15 rad/pix).
    _OVERSAMPLE = 4

    def simulate_spot_image(self, tilt_x: float, tilt_y: float) -> np.ndarray:
        """
        Simulate a single subaperture spot image given local tip/tilt
        phase gradients, via an oversampled FFT of a tilted pupil function.

        The pupil is zero-padded by a factor of ``_OVERSAMPLE`` before the
        FFT so that the Fourier-shift theorem is satisfied for sub-pixel
        tilts.  A linear phase ramp tilt_x * X + tilt_y * Y (rad/pixel
        of the n x n pupil grid, with X centred on 0) produces a focal-plane
        spot shifted by

            shift_pixels = tilt * n / (2*pi)   [in the oversampled grid]

        which after binning back to the n x n output is still linear in tilt.
        The effective pixel-per-rad conversion constant for the returned
        n x n image is therefore

            K_SHIFT = n / (2*pi)

        consistent with the inverse conversion used in _apply_realistic_noise.

        Parameters
        ----------
        tilt_x, tilt_y : float
            Local phase gradients (radians/pixel within the subaperture
            pupil grid).

        Returns
        -------
        spot_image : np.ndarray, shape (pix_per_sub, pix_per_sub)
            Normalized intensity image (sums to 1).
        """
        n = self.pix_per_sub
        os = self._OVERSAMPLE
        N_pad = n * os  # oversampled grid size

        # Pixel-centred coordinate grid for the pupil (n x n)
        x = np.arange(n) - n / 2.0
        X, Y = np.meshgrid(x, x)

        pupil_amp = np.ones((n, n))
        pupil_phase = tilt_x * X + tilt_y * Y
        pupil = pupil_amp * np.exp(1j * pupil_phase)

        # Zero-pad pupil into centre of N_pad x N_pad array
        padded = np.zeros((N_pad, N_pad), dtype=complex)
        pad_start = (N_pad - n) // 2
        padded[pad_start: pad_start + n, pad_start: pad_start + n] = pupil

        # ifftshift centres pupil before FFT; fftshift centres PSF after
        focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
        intensity_os = np.abs(focal) ** 2  # shape (N_pad, N_pad)

        # Bin back to n x n by summing os x os blocks
        intensity = intensity_os.reshape(n, os, n, os).sum(axis=(1, 3))

        total = intensity.sum()
        if total > 0:
            intensity = intensity / total
        return intensity

    def slopes_to_displacement(self, slopes: np.ndarray) -> np.ndarray:
        """
        Convert local wavefront slopes (radians/pixel-equivalent) into
        detector spot displacements (pixels):

        displacement = slope * focal_length / pitch

        Parameters
        ----------
        slopes : np.ndarray
            Slope array (any shape).

        Returns
        -------
        displacement : np.ndarray, same shape as slopes
        """
        return slopes * self.focal_length / self.pitch
