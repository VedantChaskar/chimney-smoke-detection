"""
Module 2: Atmospheric Light Estimation.

Estimates the airlight vector A — the color the scene converges to in
optically thick smoke/haze.  Uses the Dark Channel Prior heuristic
(He et al., 2009): the brightest pixels in the dark channel correspond
to the densest haze, and their original color approximates A.
"""

import numpy as np
import cv2

from .dark_channel import dark_channel


def estimate_airlight(
    image: np.ndarray,
    smoke_mask: np.ndarray,
    patch_size: int = 15,
    top_frac: float = 0.001,
) -> np.ndarray:
    """
    Estimate the atmospheric light vector A.

    Algorithm:
        1. Compute the dark channel of the full image.
        2. Among pixels INSIDE the smoke mask, pick the top `top_frac`
           fraction by dark-channel brightness.
        3. Among those candidate pixels, select the one with the highest
           overall intensity in the original image.
        4. A = that pixel's BGR color.

    Fallback: if the smoke mask covers < 1% of the image, use the global
    top-1% brightest pixels instead (ignores mask).

    Args:
        image:      (H, W, 3) uint8 BGR (or float32 [0,1]).
        smoke_mask: (H, W) binary mask.
        patch_size: DCP patch size for dark channel computation.
        top_frac:   Fraction of brightest dark-channel pixels to consider.

    Returns:
        A: (3,) float32 in [0, 1] — atmospheric light [B, G, R].
    """
    # Normalise to [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    mask = smoke_mask.astype(bool)
    h, w = img.shape[:2]
    n_pixels = h * w
    n_mask = mask.sum()

    # Choose candidate region: inside mask, or global fallback
    if n_mask < 0.01 * n_pixels:
        # Fallback: global
        candidate_flat_indices = np.arange(n_pixels)
    else:
        candidate_flat_indices = np.flatnonzero(mask)

    # Dark channel values at candidate positions
    dc = dark_channel(img, patch_size)
    dc_flat = dc.ravel()
    dc_candidates = dc_flat[candidate_flat_indices]

    # Top fraction by dark-channel brightness
    k = max(1, int(top_frac * len(dc_candidates)))
    top_k_local = np.argsort(dc_candidates)[-k:]
    top_k_global = candidate_flat_indices[top_k_local]

    # Among those, pick the pixel with highest mean intensity
    img_flat = img.reshape(-1, 3)
    intensities = img_flat[top_k_global].mean(axis=1)
    best_local = np.argmax(intensities)
    best_pixel = top_k_global[best_local]

    A = img_flat[best_pixel].astype(np.float32)
    # Safety: clamp to [0.05, 1] to avoid divide-by-zero later
    A = np.clip(A, 0.05, 1.0)
    return A
