"""
Module 3b: Contrast Attenuation Ratio.

Reference-free opacity cross-check: smoke attenuates scene contrast.
We compare local gradient magnitude inside the smoke to the boundary
region outside it.
"""

import numpy as np
import cv2

from ..utils.mask_ops import compute_boundary_band


def _local_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel Sobel gradient magnitude on a grayscale image.

    Args:
        gray: (H, W) float32 in [0, 1].

    Returns:
        magnitude: (H, W) float32.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def compute_contrast_ratio(
    image_gray: np.ndarray,
    smoke_mask: np.ndarray,
    boundary_width: int = 25,
    reliable_threshold: float = 10.0,
) -> dict:
    """
    Estimate opacity from contrast attenuation.

    Smoke reduces local gradient magnitude relative to clear regions.
    opacity_estimate = 1 - (contrast_inside / contrast_boundary)

    Args:
        image_gray:         (H, W) uint8 grayscale image.
        smoke_mask:         (H, W) binary mask.
        boundary_width:     Width of the boundary band in pixels.
        reliable_threshold: Minimum mean boundary gradient to trust the result.
                            Below this (e.g. sky background), flag as unreliable.

    Returns:
        dict with keys:
            contrast_inside   : float — mean gradient inside smoke mask
            contrast_boundary : float — mean gradient in boundary band
            contrast_ratio    : float — inside / boundary
            opacity_estimate  : float in [0, 1]
            reliable          : bool  — False if background is too uniform
    """
    gray = image_gray.astype(np.float32)

    mask = smoke_mask.astype(bool)
    boundary = compute_boundary_band(smoke_mask, boundary_width).astype(bool)

    grad = _local_gradient_magnitude(gray)

    inside_vals  = grad[mask]
    boundary_vals = grad[boundary]

    eps = 1e-6
    contrast_inside   = float(inside_vals.mean())  if inside_vals.size   > 0 else 0.0
    contrast_boundary = float(boundary_vals.mean()) if boundary_vals.size > 0 else 0.0

    ratio = contrast_inside / (contrast_boundary + eps)
    opacity = float(np.clip(1.0 - ratio, 0.0, 1.0))

    reliable = (contrast_boundary >= reliable_threshold) and (boundary_vals.size > 50)

    return {
        "contrast_inside":   contrast_inside,
        "contrast_boundary": contrast_boundary,
        "contrast_ratio":    ratio,
        "opacity_estimate":  opacity,
        "reliable":          reliable,
    }
