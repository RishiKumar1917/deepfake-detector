"""Frequency-domain analysis using Fast Fourier Transform (FFT).

Deepfake generation artefacts often manifest as characteristic patterns in
the frequency spectrum of an image.  This module provides helpers that:

* compute the 2-D FFT magnitude spectrum of a grayscale image, and
* return a 1-D feature vector derived from the spectrum that can complement
  spatial features produced by the CNN backbone.
"""

import numpy as np
import cv2
from PIL import Image


def compute_fft_spectrum(image_path: str, size: int = 224) -> np.ndarray:
    """Return the log-magnitude FFT spectrum of an image as a 2-D array.

    The image is converted to grayscale, resized to *size* × *size*, and the
    zero-frequency component is shifted to the centre so that low-frequency
    energy appears in the middle of the plot.

    Args:
        image_path: Path to the source image.
        size: Side length (pixels) to which the image is resized before FFT.

    Returns:
        2-D float32 array of shape ``(size, size)`` containing the
        log-magnitude spectrum.
    """
    img = Image.open(image_path).convert("L")        # grayscale
    img = img.resize((size, size))
    gray = np.array(img, dtype=np.float32)

    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)              # log(1 + |F|)
    return log_magnitude.astype(np.float32)


def extract_frequency_features(image_path: str, n_bins: int = 64) -> np.ndarray:
    """Extract a compact 1-D frequency feature vector from an image.

    The log-magnitude spectrum is divided into *n_bins* radial frequency bands
    and the mean energy in each band is returned.  This captures the radial
    energy distribution without being sensitive to image rotation.

    Args:
        image_path: Path to the source image.
        n_bins: Number of radial frequency bins.

    Returns:
        1-D float32 array of length *n_bins*.
    """
    spectrum = compute_fft_spectrum(image_path)
    h, w = spectrum.shape
    cx, cy = w // 2, h // 2

    # Build a distance map from the centre.
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    # Bin the spectrum values by radial distance.
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    features = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        values = spectrum[mask]
        features[i] = values.mean() if values.size > 0 else 0.0

    return features


def get_fft_visualization(image_path: str, size: int = 224) -> np.ndarray:
    """Return an 8-bit grayscale image of the normalised FFT spectrum.

    Useful for side-by-side visualisation in the output heatmap figure.
    """
    spectrum = compute_fft_spectrum(image_path, size=size)
    norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)
