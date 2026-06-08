"""
src/opacity/formula.py

Brightness-ratio opacity formulas from Chen et al.

Formula 1 — dark / black smoke against a bright background:
    opacity (%) = alpha * (1.0 - L_smoke / L_bright)

Formula 2 — white / grey smoke against any background:
    opacity (%) = ((L_smoke - L_dark) / (L_bright - L_dark)) * 100

`alpha` is a calibration constant (default 100).  When alpha=100 the formula
is the standard Ringelmann reference (each 20% opacity = one grade).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg

from .brightness import get_region_brightness


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    """Division that returns `fallback` when denominator is zero or near-zero."""
    return num / denom if abs(denom) > 1e-6 else fallback


def _fix_inverted_bright(image_bgr, L_smoke: float, L_bright: float) -> tuple:
    """
    Sanity-check the bright reference.

    If the detected sky region is dimmer than the smoke (L_bright ≤ L_smoke),
    the detected region is not a reliable bright reference — replace it with
    the image 95th-percentile brightness, which is always the actual ceiling.

    Returns:
        (L_bright_corrected, was_corrected: bool)
    """
    if L_bright > L_smoke:
        return L_bright, False          # reference is sensible — leave it

    import cv2
    import numpy as np
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    L_bright_p95 = float(np.percentile(gray, 95))
    # Only use the p95 if it's actually brighter than smoke;
    # if even p95 ≤ L_smoke the image is extremely unusual — keep p95 anyway.
    return L_bright_p95, True


# ── Core formulas ──────────────────────────────────────────────────────────────

def formula_dark_smoke(L_smoke: float, L_bright: float, alpha: float = cfg.ALPHA_DEFAULT) -> float:
    """
    Chen et al. Formula 1 — dark smoke against a bright background.

    opacity (%) = alpha * (1 - L_smoke / L_bright)

    Args:
        L_smoke:  mean brightness of smoke region (0–255).
        L_bright: mean brightness of bright background (sky, 0–255).
        alpha:    calibration constant (default 100).

    Returns:
        Opacity in [0, 100] %.
    """
    ratio   = _safe_div(L_smoke, L_bright, fallback=1.0)
    opacity = alpha * (1.0 - ratio)
    return float(max(0.0, min(100.0, opacity)))


def formula_white_smoke(L_smoke: float, L_bright: float, L_dark: float) -> float:
    """
    Chen et al. Formula 2 — white / grey smoke between dark and bright backgrounds.

    opacity (%) = ((L_smoke - L_dark) / (L_bright - L_dark)) * 100

    Args:
        L_smoke:  mean brightness of smoke region (0–255).
        L_bright: mean brightness of bright background (0–255).
        L_dark:   mean brightness of dark background (0–255).

    Returns:
        Opacity in [0, 100] %.
    """
    span    = L_bright - L_dark
    diff    = L_smoke  - L_dark
    opacity = _safe_div(diff, span, fallback=0.5) * 100.0
    return float(max(0.0, min(100.0, opacity)))


# ── High-level entry point ─────────────────────────────────────────────────────

def compute_opacity(image_bgr, regions: dict, alpha: float = cfg.ALPHA_DEFAULT) -> dict:
    """
    Compute opacity (%) from classified region mask dicts.

    Selects the appropriate formula based on the smoke type
    (smoke_dark → Formula 1; smoke_light → Formula 2).

    Args:
        image_bgr: (H, W, 3) BGR image.
        regions:   dict from region_classifier.classify_regions(), keys:
                       smoke_dark, smoke_light,
                       background_bright, background_dark
                   each value is a mask dict or None.
        alpha:     calibration constant for Formula 1 (default 100).

    Returns:
        dict with keys:
            "opacity"        : float [0, 100] — final opacity %
            "formula"        : "formula_1" | "formula_2" | "fallback" | "none"
            "smoke_type"     : "dark" | "light" | "none"
            "L_smoke"        : float or None
            "L_bright"       : float or None
            "L_dark"         : float or None
            "reliable"       : bool — False if key references were missing
    """
    result = {
        "opacity":   0.0,
        "formula":   "none",
        "smoke_type": "none",
        "L_smoke":   None,
        "L_bright":  None,
        "L_dark":    None,
        "reliable":  False,
    }

    # ── 1. Extract brightness references ──────────────────────────────────────
    L_bright = get_region_brightness(image_bgr, regions.get("background_bright"))
    L_dark   = get_region_brightness(image_bgr, regions.get("background_dark"))

    result["L_bright"] = L_bright
    result["L_dark"]   = L_dark

    # ── 2. Dark smoke → Formula 1 ─────────────────────────────────────────────
    smoke_dark = regions.get("smoke_dark")
    if smoke_dark is not None:
        L_smoke = get_region_brightness(image_bgr, smoke_dark)
        result["L_smoke"]    = L_smoke
        result["smoke_type"] = "dark"

        if L_smoke is not None and L_bright is not None:
            # Fix 1: if detected sky is dimmer than smoke, use image p95 instead
            L_bright, corrected = _fix_inverted_bright(image_bgr, L_smoke, L_bright)
            result["L_bright"]  = L_bright
            result["opacity"]   = formula_dark_smoke(L_smoke, L_bright, alpha=alpha)
            result["formula"]   = "formula_1"
            result["reliable"]  = not corrected
        elif L_smoke is not None:
            # No bright background — use image p95 as proxy
            import cv2, numpy as np
            gray     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            L_bright = float(np.percentile(gray, 95))
            result["L_bright"]  = L_bright
            result["opacity"]   = formula_dark_smoke(L_smoke, L_bright, alpha=alpha)
            result["formula"]   = "formula_1"
            result["reliable"]  = False
        return result

    # ── 3. Light / grey smoke → Formula 2 ────────────────────────────────────
    smoke_light = regions.get("smoke_light")
    if smoke_light is not None:
        L_smoke = get_region_brightness(image_bgr, smoke_light)
        result["L_smoke"]    = L_smoke
        result["smoke_type"] = "light"

        if L_smoke is not None and L_bright is not None and L_dark is not None:
            # Fix 1: if detected sky is dimmer than smoke, use image p95 instead
            L_bright, corrected = _fix_inverted_bright(image_bgr, L_smoke, L_bright)
            result["L_bright"]  = L_bright
            result["opacity"]   = formula_white_smoke(L_smoke, L_bright, L_dark)
            result["formula"]   = "formula_2"
            result["reliable"]  = not corrected
        elif L_smoke is not None:
            # Missing one or both background references — use image-derived fallbacks
            import cv2, numpy as np
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if L_bright is None:
                L_bright = float(np.percentile(gray, 95))
                result["L_bright"] = L_bright
            if L_dark is None:
                L_dark = float(np.percentile(gray, 5))
                result["L_dark"] = L_dark
            result["opacity"]  = formula_white_smoke(L_smoke, L_bright, L_dark)
            result["formula"]  = "formula_2"
            result["reliable"] = False   # proxy references
        return result

    # ── 4. No smoke detected ───────────────────────────────────────────────────
    result["formula"] = "none"
    return result
