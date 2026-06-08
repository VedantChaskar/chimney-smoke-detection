"""
Module 3c: Color Channel Variance Analysis (Color Collapse).

Dense smoke collapses RGB channel differences — it pushes pixel colors
toward the airlight color (near-white/grey).  We measure how much the
per-pixel channel spread (max - min across R,G,B) is reduced inside
the smoke compared to the surrounding boundary.
"""

import numpy as np

from ..utils.mask_ops import compute_boundary_band


def compute_color_collapse(
    image: np.ndarray,
    smoke_mask: np.ndarray,
    boundary_width: int = 25,
) -> dict:
    """
    Estimate opacity from color channel collapse.

    opacity_estimate = 1 - (mean_spread_inside / mean_spread_boundary)

    Args:
        image:          (H, W, 3) uint8 BGR image.
        smoke_mask:     (H, W) binary mask.
        boundary_width: Width of the boundary band in pixels.

    Returns:
        dict with keys:
            spread_inside     : float — mean per-pixel channel spread inside smoke
            spread_boundary   : float — mean per-pixel channel spread in boundary
            collapse_ratio    : float — inside / boundary
            opacity_estimate  : float in [0, 1]
            reliable          : bool  — False if boundary spread is near 0
    """
    img = image.astype(np.float32)

    mask     = smoke_mask.astype(bool)
    boundary = compute_boundary_band(smoke_mask, boundary_width).astype(bool)

    # Per-pixel channel spread: max(R,G,B) - min(R,G,B)
    channel_max = img.max(axis=2)   # (H, W)
    channel_min = img.min(axis=2)   # (H, W)
    spread      = channel_max - channel_min  # (H, W)

    eps = 1e-6
    spread_inside   = float(spread[mask].mean())    if mask.sum()     > 0 else 0.0
    spread_boundary = float(spread[boundary].mean()) if boundary.sum() > 0 else 0.0

    collapse_ratio  = spread_inside / (spread_boundary + eps)
    opacity         = float(np.clip(1.0 - collapse_ratio, 0.0, 1.0))

    # Reliable only when boundary has meaningful colour variation
    reliable = (spread_boundary >= 5.0) and (boundary.sum() > 50)

    return {
        "spread_inside":    spread_inside,
        "spread_boundary":  spread_boundary,
        "collapse_ratio":   collapse_ratio,
        "opacity_estimate": opacity,
        "reliable":         reliable,
    }
