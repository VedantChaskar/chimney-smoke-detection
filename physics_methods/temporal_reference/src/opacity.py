# src/opacity.py
"""
Opacity formulas and Ringelman mapping.

Formula selection:
    Dark smoke (smoke_dark found):
        Formula 1 — black smoke opacity.
        L_background = ground truth brightness at smoke pixel locations (temporal).

    Light smoke (smoke_light found) + dark background in ground truth:
        Formula 2 — white smoke full formula.
        L_bright = ground truth brightness at smoke pixel locations.
        L_dark   = ground truth L_dark (dark-background region mean).

    Light smoke + no usable dark background:
        Formula 1 as fallback (approximation).

    No smoke detected:
        opacity = 0.0, grade = 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from .ground_truth import GroundTruth


def compute_opacity(
    smoke_bgr: np.ndarray,
    best_regions: dict,
    ground_truth: GroundTruth,
    alpha: float = cfg.ALPHA_DEFAULT,
) -> dict:
    """
    Compute opacity % for one smoke image using temporal ground truth comparison.

    Args:
        smoke_bgr    : (H, W, 3) aligned smoke image
        best_regions : dict from segment_smoke_image()
        ground_truth : GroundTruth object
        alpha        : calibration constant

    Returns dict with keys:
        opacity_pct  : float 0–100 or None
        ringelman    : int 0–5 or None
        formula_used : str
        smoke_type   : "dark" | "light" | "none"
        reason       : str
    """
    gray = cv2.cvtColor(smoke_bgr, cv2.COLOR_BGR2GRAY)

    # ── No smoke ──────────────────────────────────────────────────────────────
    if best_regions["smoke_dark"] is None and best_regions["smoke_light"] is None:
        return _result(0.0, "none", "none", "No smoke region detected")

    # ── Dark smoke — Formula 1 ────────────────────────────────────────────────
    if best_regions["smoke_dark"] is not None:
        smoke_seg    = best_regions["smoke_dark"]["segmentation"]
        smoke_pixels = gray[smoke_seg]
        if len(smoke_pixels) == 0:
            return _result(None, "none", "dark", "smoke_dark mask is empty")

        L_smoke = float(np.mean(smoke_pixels))
        L_bg    = ground_truth.get_brightness_at_mask(smoke_seg)

        if L_bg is None or L_bg == 0:
            return _result(None, "none", "dark",
                           "Ground truth brightness at smoke location is zero or empty")

        opacity = alpha * (1.0 - L_smoke / L_bg)
        return _result(float(np.clip(opacity, 0, 100)), "formula_1_dark_smoke", "dark", "OK")

    # ── Light smoke ───────────────────────────────────────────────────────────
    smoke_seg    = best_regions["smoke_light"]["segmentation"]
    smoke_pixels = gray[smoke_seg]
    if len(smoke_pixels) == 0:
        return _result(None, "none", "light", "smoke_light mask is empty")

    L_smoke       = float(np.mean(smoke_pixels))
    L_bg_at_smoke = ground_truth.get_brightness_at_mask(smoke_seg)   # temporal lookup
    L_bright      = ground_truth.L_bright   # GT sky / bright-region mean

    # Formula 2 — temporal light-smoke formula.
    #
    # Physical meaning: white/grey smoke brightens the area it covers.
    #   Numerator   = how much the smoke brightened the area vs the clean GT frame.
    #   Denominator = maximum possible brightening (pure sky brightness - clean bg).
    #   Result      = 0%   → no brightening (no smoke / transparent smoke)
    #                100%  → smoke as bright as open sky (fully opaque white smoke)
    #
    # Using gt.L_bright (sky mean) as the ceiling avoids the denominator-collapse
    # that occurs when L_bg_at_smoke ≈ L_dark (dark-roof smoke locations).
    if L_bg_at_smoke is not None and L_bright is not None:
        denom = L_bright - L_bg_at_smoke
        if denom > 1:   # guard against degenerate sky-over-sky case
            opacity = ((L_smoke - L_bg_at_smoke) / denom) * 100
            return _result(float(np.clip(opacity, 0, 100)),
                           "formula_2_light_smoke", "light", "OK")

    # Formula 1 fallback — treat light smoke as attenuating a bright background
    if L_bg_at_smoke is not None and L_bg_at_smoke > 0:
        opacity = alpha * (1.0 - L_smoke / L_bg_at_smoke)
        return _result(float(np.clip(opacity, 0, 100)),
                       "formula_1_fallback_light_smoke", "light",
                       "No usable sky reference — used Formula 1 approximation")

    return _result(None, "none", "light",
                   "Could not obtain background reference from ground truth")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result(opacity_pct, formula_used, smoke_type, reason) -> dict:
    return {
        "opacity_pct":  opacity_pct,
        "ringelman":    _to_ringelman(opacity_pct),
        "formula_used": formula_used,
        "smoke_type":   smoke_type,
        "reason":       reason,
    }


def _to_ringelman(opacity_pct) -> int | None:
    if opacity_pct is None:
        return None
    for grade, threshold in enumerate(cfg.RINGELMAN_THRESHOLDS):
        if opacity_pct < threshold:
            return grade
    return 5
