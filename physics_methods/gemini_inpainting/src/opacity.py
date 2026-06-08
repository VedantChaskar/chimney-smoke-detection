# src/opacity.py
"""
Opacity formulas for Method 2.

Core principle (identical to Method 1, different ground truth source):
    L_smoke      = brightness of smoke pixels in the ORIGINAL image
    L_background = brightness of the SAME pixels in the INPAINTED image
                   (synthetic ground truth — Gemini's reconstruction of what
                    was behind the smoke)

Formula selection:
    Dark smoke:
        Formula 1 — opacity = alpha * (1 - L_smoke / L_background)

    Light smoke + dark background region available:
        Formula 2 — temporal light-smoke formula:
            opacity = (L_smoke - L_bg_at_smoke) / (L_bright_gt - L_bg_at_smoke) * 100
            where L_bright_gt = mean brightness of bright region in inpainted image

    Light smoke, no usable dark background:
        Formula 1 as fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def compute_opacity(
    original_bgr: np.ndarray,
    inpainted_bgr: np.ndarray,
    smoke_mask_dict: dict,
    best_regions: dict,
    smoke_type: str,
    alpha: float = cfg.ALPHA_DEFAULT,
) -> dict:
    """
    Compute opacity % by comparing original vs inpainted image at smoke pixels.

    Args:
        original_bgr   : (H, W, 3) original smoke image
        inpainted_bgr  : (H, W, 3) Gemini inpainted synthetic ground truth
        smoke_mask_dict: best smoke mask dict
        best_regions   : full classified regions dict (for background_dark)
        smoke_type     : "dark" or "light"
        alpha          : calibration constant

    Returns dict: opacity_pct, ringelman, formula_used, reason
    """
    gray_orig     = cv2.cvtColor(original_bgr,  cv2.COLOR_BGR2GRAY)
    gray_inpainted = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2GRAY)

    smoke_seg    = smoke_mask_dict["segmentation"]
    smoke_pixels = gray_orig[smoke_seg]

    if len(smoke_pixels) == 0:
        return _res(None, "none", "Smoke mask is empty")

    L_smoke = float(np.mean(smoke_pixels))

    # Background = inpainted image at the same smoke pixel locations
    bg_pixels    = gray_inpainted[smoke_seg]
    L_background = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else None

    if L_background is None or L_background == 0:
        return _res(None, "none", "Inpainted background is zero or empty")

    # ── Dark smoke — Formula 1 ────────────────────────────────────────────────
    # Also use Formula 1 when smoke is darker than inpainted background (e.g. thin
    # grey smoke against a bright sky — classified as "light" by absolute brightness
    # but actually absorbs/blocks light relative to background).
    if smoke_type == "dark" or L_smoke <= L_background:
        opacity = alpha * (1.0 - L_smoke / L_background)
        formula = "formula_1_dark_smoke" if smoke_type == "dark" else "formula_1_absorbing_smoke"
        return _res(float(np.clip(opacity, 0, 100)), formula, "OK")

    # ── Light smoke — Formula 2 (temporal brightening) ───────────────────────
    #
    # White/grey smoke scatters light and BRIGHTENS its background, so
    # L_smoke > L_background (inpainted = what was there before the smoke).
    #
    # Measure how much of the "remaining brightness headroom" the smoke filled:
    #   opacity = (L_smoke - L_background) / (L_ceiling - L_background) * 100
    #
    # L_ceiling is the maximum expected brightness for this scene.
    # Priority:
    #   1. Bright background region from the ORIGINAL image (sky, open wall)
    #   2. 95th percentile of the original image
    #   3. Hard ceiling of 255
    #
    # Using the ORIGINAL (not inpainted) for L_ceiling is important because
    # OpenCV inpainting may not reproduce bright sky correctly, making the
    # inpainted percentile collapse to ≈ L_background.

    bright_mask = best_regions.get("background_bright")
    L_ceiling   = None

    if bright_mask is not None:
        bright_pixels = gray_orig[bright_mask["segmentation"]]   # from ORIGINAL
        if len(bright_pixels) > 0:
            L_ceiling = float(np.mean(bright_pixels))

    if L_ceiling is None:
        L_ceiling = float(np.percentile(gray_orig, 95))

    # Ensure ceiling is strictly above background
    if L_ceiling <= L_background + 1:
        L_ceiling = 255.0

    denom = L_ceiling - L_background
    if denom > 1:
        opacity = ((L_smoke - L_background) / denom) * 100
        return _res(float(np.clip(opacity, 0, 100)), "formula_2_light_smoke", "OK")

    # Absolute fallback: use 255 ceiling
    opacity = ((L_smoke - L_background) / (255.0 - L_background)) * 100
    return _res(float(np.clip(opacity, 0, 100)),
                "formula_2_light_smoke_255_ceiling",
                "Scene has no bright reference — used absolute ceiling 255")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _res(opacity_pct, formula_used, reason) -> dict:
    return {
        "opacity_pct":  opacity_pct,
        "ringelman":    _to_ringelman(opacity_pct),
        "formula_used": formula_used,
        "reason":       reason,
    }


def _to_ringelman(opacity_pct) -> int | None:
    if opacity_pct is None:
        return None
    if opacity_pct <= 0:
        return 0
    for grade, thresh in enumerate(cfg.RINGELMAN_THRESHOLDS, start=1):
        if opacity_pct <= thresh:
            return grade
    return 5
