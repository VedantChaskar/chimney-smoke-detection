# tests/test_opacity.py
"""
Unit tests for Method 2 — opacity formulas, Ringelman mapping,
smoke validator, and mask generator.
No GPU, SAM3, or Gemini API required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import cv2

from src.opacity        import _to_ringelman, compute_opacity
from src.smoke_validator import validate_smoke
from src.mask_generator  import generate_pixel_mask


# ── Ringelman mapping ─────────────────────────────────────────────────────────

def test_ringelman_grade_0():
    assert _to_ringelman(0.0) == 0

def test_ringelman_grade_1():
    assert _to_ringelman(0.1) == 1
    assert _to_ringelman(20.0) == 1

def test_ringelman_grade_2():
    assert _to_ringelman(20.1) == 2
    assert _to_ringelman(40.0) == 2

def test_ringelman_grade_3():
    assert _to_ringelman(40.1) == 3
    assert _to_ringelman(60.0) == 3

def test_ringelman_grade_4():
    assert _to_ringelman(60.1) == 4
    assert _to_ringelman(80.0) == 4

def test_ringelman_grade_5():
    assert _to_ringelman(80.1) == 5
    assert _to_ringelman(100.0) == 5

def test_ringelman_none():
    assert _to_ringelman(None) is None


# ── compute_opacity (synthetic inpainted GT) ──────────────────────────────────

def _make_smoke_image(H=100, W=100, smoke_val=80, bg_val=160):
    """Dark smoke over a medium background."""
    orig      = np.ones((H, W, 3), dtype=np.uint8) * smoke_val
    inpainted = np.ones((H, W, 3), dtype=np.uint8) * bg_val
    seg       = np.ones((H, W), dtype=bool)
    smoke_mask = {"segmentation": seg, "area": H * W, "brightness": smoke_val}
    best_regions = {
        "smoke_dark": smoke_mask, "smoke_light": None,
        "background_bright": None, "background_dark": None,
    }
    return orig, inpainted, smoke_mask, best_regions


def test_dark_smoke_formula1():
    """Dark smoke at 80, inpainted bg at 160, alpha=100 → opacity=50%."""
    orig, inp, sm, br = _make_smoke_image(smoke_val=80, bg_val=160)
    r = compute_opacity(orig, inp, sm, br, "dark", alpha=100)
    assert r["formula_used"] == "formula_1_dark_smoke"
    assert abs(r["opacity_pct"] - 50.0) < 0.1
    assert r["ringelman"] == 3


def test_no_smoke_zero_opacity():
    """When smoke brightness equals inpainted background, opacity = 0."""
    orig, inp, sm, br = _make_smoke_image(smoke_val=160, bg_val=160)
    r = compute_opacity(orig, inp, sm, br, "dark", alpha=100)
    assert r["opacity_pct"] == 0.0
    assert r["ringelman"] == 0


def test_light_smoke_formula2():
    """Light smoke over dark area: smoke brightens it; formula 2 calculates ratio."""
    H, W    = 100, 100
    seg     = np.ones((H, W), dtype=bool)

    # Original: smoke region is bright (light smoke brightening a dark background)
    orig = np.ones((H, W, 3), dtype=np.uint8) * 160
    # Inpainted: without smoke, that region was dark (80), sky is 220
    inpainted = np.ones((H, W, 3), dtype=np.uint8) * 80
    # Bright background region in inpainted
    bright_seg = np.zeros((H, W), dtype=bool)
    bright_seg[:10, :] = True
    inpainted_with_sky = inpainted.copy()
    inpainted_with_sky[:10, :] = 220

    smoke_mask = {"segmentation": seg, "area": H * W, "brightness": 160.0}
    best_regions = {
        "smoke_dark":  None,
        "smoke_light": smoke_mask,
        "background_bright": {"segmentation": bright_seg, "area": bright_seg.sum(), "brightness": 220.0},
        "background_dark":   None,
    }

    r = compute_opacity(orig, inpainted_with_sky, smoke_mask, best_regions, "light", alpha=100)
    # L_smoke=160, L_bg_at_smoke=80 (non-sky inpainted), L_bright≈220 (sky)
    # opacity = (160 - 80) / (220 - 80) * 100 ≈ 57.1%
    assert r["formula_used"] in ("formula_2_light_smoke", "formula_1_fallback_light_smoke")
    assert r["opacity_pct"] is not None
    assert r["ringelman"] is not None


def test_empty_smoke_mask_returns_none():
    H, W = 100, 100
    seg  = np.zeros((H, W), dtype=bool)   # empty mask
    orig      = np.ones((H, W, 3), dtype=np.uint8) * 100
    inpainted = np.ones((H, W, 3), dtype=np.uint8) * 200
    smoke_mask = {"segmentation": seg, "area": 0, "brightness": 0.0}
    r = compute_opacity(orig, inpainted, smoke_mask, {}, "dark")
    assert r["opacity_pct"] is None
    assert r["ringelman"]   is None


# ── Smoke validator ────────────────────────────────────────────────────────────

def _fake_mask(area_pct=0.05, confidence=0.5, H=100, W=100):
    seg  = np.zeros((H, W), dtype=bool)
    n    = int(H * W * area_pct)
    seg.ravel()[:n] = True
    return {"segmentation": seg, "area": int(seg.sum()), "brightness": 100.0,
            "confidence": confidence}


def test_validator_smoke_present():
    mask = _fake_mask(area_pct=0.05, confidence=0.5)
    best = {"smoke_dark": mask, "smoke_light": None,
            "background_bright": None, "background_dark": None}
    v = validate_smoke(best, (100, 100, 3))
    assert v["smoke_present"] is True
    assert v["smoke_type"] == "dark"


def test_validator_no_smoke_region():
    best = {"smoke_dark": None, "smoke_light": None,
            "background_bright": None, "background_dark": None}
    v = validate_smoke(best, (100, 100, 3))
    assert v["smoke_present"] is False


def test_validator_low_confidence():
    mask = _fake_mask(area_pct=0.05, confidence=0.1)
    best = {"smoke_dark": mask, "smoke_light": None,
            "background_bright": None, "background_dark": None}
    v = validate_smoke(best, (100, 100, 3))
    assert v["smoke_present"] is False


def test_validator_too_small():
    mask = _fake_mask(area_pct=0.005, confidence=0.8)   # < MIN_AREA_RATIO (0.02)
    best = {"smoke_dark": mask, "smoke_light": None,
            "background_bright": None, "background_dark": None}
    v = validate_smoke(best, (100, 100, 3))
    assert v["smoke_present"] is False


# ── Mask generator ────────────────────────────────────────────────────────────

def _make_seg(H=100, W=100):
    seg = np.zeros((H, W), dtype=bool)
    seg[20:60, 30:70] = True
    return {"segmentation": seg, "area": int(seg.sum()), "brightness": 150.0, "confidence": 0.5}


def test_pixel_mask_shape():
    m = _make_seg()
    mask = generate_pixel_mask(m, (100, 100, 3))
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert mask.max() == 255


def test_pixel_mask_covers_segmentation():
    m    = _make_seg()
    mask = generate_pixel_mask(m, (100, 100, 3))
    # All pixels in the original seg must be 255 in the pixel mask (may be dilated)
    seg  = m["segmentation"]
    assert np.all(mask[seg] == 255)


