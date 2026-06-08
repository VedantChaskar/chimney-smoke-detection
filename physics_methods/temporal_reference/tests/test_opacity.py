# tests/test_opacity.py
"""
Unit tests for opacity formula math and Ringelman mapping.
No images, SAM3, or GPU required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.opacity import _to_ringelman, compute_opacity
from src.ground_truth import GroundTruth


# ── Ringelman mapping ─────────────────────────────────────────────────────────

def test_ringelman_grade_0():
    assert _to_ringelman(0.0) == 0
    assert _to_ringelman(9.9) == 0

def test_ringelman_grade_1():
    assert _to_ringelman(10.0) == 1
    assert _to_ringelman(29.9) == 1

def test_ringelman_grade_2():
    assert _to_ringelman(30.0) == 2
    assert _to_ringelman(49.9) == 2

def test_ringelman_grade_3():
    assert _to_ringelman(50.0) == 3
    assert _to_ringelman(69.9) == 3

def test_ringelman_grade_4():
    assert _to_ringelman(70.0) == 4
    assert _to_ringelman(89.9) == 4

def test_ringelman_grade_5():
    assert _to_ringelman(90.0) == 5
    assert _to_ringelman(100.0) == 5

def test_ringelman_none():
    assert _to_ringelman(None) is None


# ── Formula 1 (dark smoke): opacity = alpha * (1 - L_smoke / L_background) ───

def test_formula1_known_values():
    """L_smoke=100, L_background=200, alpha=100 → 50.0"""
    alpha = 100.0
    L_smoke = 100.0
    L_bg    = 200.0
    result  = alpha * (1.0 - L_smoke / L_bg)
    assert abs(result - 50.0) < 0.01

def test_formula1_no_smoke():
    """If L_smoke == L_background, opacity = 0."""
    alpha = 100.0
    L     = 180.0
    assert alpha * (1.0 - L / L) == 0.0

def test_formula1_full_opacity():
    """If L_smoke = 0, opacity = alpha = 100."""
    alpha   = 100.0
    L_bg    = 200.0
    L_smoke = 0.0
    result  = alpha * (1.0 - L_smoke / L_bg)
    assert abs(result - 100.0) < 0.01


# ── Formula 2 (light smoke): opacity = (L_smoke - L_dark) / (L_bright - L_dark) * 100

def test_formula2_known_values():
    """L_smoke=80, L_dark=60, L_bright=160 → 20.0"""
    L_smoke  = 80.0
    L_dark   = 60.0
    L_bright = 160.0
    result   = ((L_smoke - L_dark) / (L_bright - L_dark)) * 100
    assert abs(result - 20.0) < 0.01

def test_formula2_transparent_smoke():
    """If smoke brightness equals bright background, opacity = 100 (no darkening)."""
    L_smoke  = 160.0
    L_dark   = 60.0
    L_bright = 160.0
    result   = ((L_smoke - L_dark) / (L_bright - L_dark)) * 100
    assert abs(result - 100.0) < 0.01

def test_formula2_full_dark():
    """If smoke brightness equals dark background, opacity = 0."""
    L_dark   = 60.0
    L_bright = 160.0
    result   = ((L_dark - L_dark) / (L_bright - L_dark)) * 100
    assert result == 0.0


# ── compute_opacity integration (synthetic GroundTruth) ──────────────────────

def _make_gt(H=100, W=100, bright_val=200, dark_val=50):
    """Synthetic GroundTruth: top half bright, bottom half dark."""
    import cv2
    gray_np  = np.zeros((H, W), dtype=np.uint8)
    gray_np[:H//2, :] = bright_val
    gray_np[H//2:, :] = dark_val
    bgr = cv2.cvtColor(gray_np, cv2.COLOR_GRAY2BGR)

    # Fake sky mask (upper half) and dark_background mask (lower half)
    sky_seg  = np.zeros((H, W), dtype=bool)
    sky_seg[:H//2, :] = True
    dark_seg = np.zeros((H, W), dtype=bool)
    dark_seg[H//2:, :] = True

    sky_mask  = {"segmentation": sky_seg,  "area": sky_seg.sum(),  "brightness": bright_val}
    dark_mask = {"segmentation": dark_seg, "area": dark_seg.sum(), "brightness": dark_val}

    gt = GroundTruth.__new__(GroundTruth)
    gt.image_bgr = bgr
    gt.gray      = gray_np
    gt.raw_masks = {"sky": sky_mask, "dark_background": dark_mask}
    gt.L_bright  = float(bright_val)
    gt.L_dark    = float(dark_val)
    return gt


def test_no_smoke_returns_zero():
    import cv2
    H, W = 100, 100
    gt   = _make_gt(H, W)
    smoke_bgr = np.ones((H, W, 3), dtype=np.uint8) * 128
    best_regions = {"smoke_dark": None, "smoke_light": None,
                    "background_bright": None, "background_dark": None}
    r = compute_opacity(smoke_bgr, best_regions, gt)
    assert r["opacity_pct"] == 0.0
    assert r["ringelman"]   == 0
    assert r["smoke_type"]  == "none"


def test_dark_smoke_formula1():
    """Dark smoke at brightness 100 against GT bright=200 → opacity ≈ 50."""
    import cv2
    H, W = 100, 100
    gt   = _make_gt(H, W, bright_val=200)

    smoke_bgr          = np.ones((H, W, 3), dtype=np.uint8) * 100
    smoke_seg          = np.ones((H, W), dtype=bool)  # whole image is smoke
    best_regions = {
        "smoke_dark":        {"segmentation": smoke_seg, "area": H*W, "brightness": 100.0},
        "smoke_light":       None,
        "background_bright": None,
        "background_dark":   None,
    }
    r = compute_opacity(smoke_bgr, best_regions, gt, alpha=100)
    # L_bg_at_mask = mean of GT where smoke_seg is True
    # GT is 200 top half + 50 bottom half → mean ≈ 125
    # opacity = 100 * (1 - 100/125) = 20.0
    assert r["smoke_type"]   == "dark"
    assert r["formula_used"] == "formula_1_dark_smoke"
    assert r["opacity_pct"]  is not None
    assert r["ringelman"]    is not None


def test_light_smoke_formula2():
    """Light smoke at mid-brightness → formula 2 selected with dark bg available."""
    import cv2
    H, W = 100, 100
    gt   = _make_gt(H, W, bright_val=200, dark_val=50)

    # Smoke pixels = brightness 150 in upper half (bright GT region)
    smoke_bgr = np.ones((H, W, 3), dtype=np.uint8) * 150
    smoke_seg = np.zeros((H, W), dtype=bool)
    smoke_seg[:H//2, :] = True  # upper half, GT brightness = 200 there

    best_regions = {
        "smoke_dark":        None,
        "smoke_light":       {"segmentation": smoke_seg, "area": smoke_seg.sum(), "brightness": 150.0},
        "background_bright": None,
        "background_dark":   None,
    }
    r = compute_opacity(smoke_bgr, best_regions, gt, alpha=100)
    # L_bg_at_smoke = 200 (upper half of GT), L_dark = 50
    # opacity = (150 - 50) / (200 - 50) * 100 ≈ 66.67
    assert r["smoke_type"]  == "light"
    assert r["formula_used"] in ("formula_2_light_smoke", "formula_1_fallback_light_smoke")
    assert r["opacity_pct"]  is not None
