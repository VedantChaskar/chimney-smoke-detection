# src/smoke_validator.py
"""
Validates smoke detection before running the expensive inpainting step.

Two conditions must both pass:
  1. SAM3 confidence >= SMOKE_MIN_CONFIDENCE
  2. Smoke region covers >= SMOKE_MIN_AREA_RATIO of the image
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def validate_smoke(best_regions: dict, image_shape: tuple) -> dict:
    """
    Determine whether detected smoke is real and significant enough
    to justify running Gemini inpainting.

    Args:
        best_regions : dict from segment_image()
        image_shape  : (H, W, C)

    Returns dict with keys:
        smoke_present  : bool
        smoke_type     : "dark" | "light" | "none"
        smoke_mask     : the winning mask dict or None
        coverage_ratio : float
        reason         : str
    """
    img_area = image_shape[0] * image_shape[1]

    smoke_mask = best_regions.get("smoke_dark") or best_regions.get("smoke_light")
    smoke_type = (
        "dark"  if best_regions.get("smoke_dark")  is not None else
        "light" if best_regions.get("smoke_light") is not None else
        "none"
    )

    if smoke_mask is None:
        return _no_smoke("SAM3 found no smoke region")

    # ── Check 1: confidence score ─────────────────────────────────────────────
    confidence = smoke_mask.get("confidence", 0.0)
    if confidence < cfg.SMOKE_MIN_CONFIDENCE:
        return _no_smoke(
            f"SAM3 confidence {confidence:.2f} below threshold {cfg.SMOKE_MIN_CONFIDENCE}",
            smoke_type, smoke_mask["area"] / img_area,
        )

    # ── Check 2: minimum area coverage ───────────────────────────────────────
    coverage = smoke_mask["area"] / img_area
    if coverage < cfg.SMOKE_MIN_AREA_RATIO:
        return _no_smoke(
            f"Smoke coverage {coverage:.3f} below minimum {cfg.SMOKE_MIN_AREA_RATIO}",
            smoke_type, coverage,
        )

    return {
        "smoke_present":  True,
        "smoke_type":     smoke_type,
        "smoke_mask":     smoke_mask,
        "coverage_ratio": coverage,
        "reason":         "OK",
    }


def _no_smoke(reason: str, smoke_type: str = "none", coverage: float = 0.0) -> dict:
    return {
        "smoke_present":  False,
        "smoke_type":     smoke_type,
        "smoke_mask":     None,
        "coverage_ratio": coverage,
        "reason":         reason,
    }
