# src/segmenter.py
"""
SAM3 smoke segmentation for Method 2.

Unlike Method 1, there is no ground-truth frame to diff against.
Detection relies entirely on SAM3 text-prompt segmentation + HSV fallback.

Returns the four region types the opacity formulas need:
    smoke_dark        — dark / black smoke
    smoke_light       — white / grey smoke
    background_bright — sky, bright surfaces
    background_dark   — rooftop, chimney, dark surfaces
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from .sam3_loader import SAM3Segmenter


def segment_image(
    image_bgr: np.ndarray,
    segmenter: SAM3Segmenter,
) -> tuple[list, dict]:
    """
    Segment image and classify regions.

    Returns
    -------
    classified_masks : list (not used downstream, kept for API compatibility)
    best_regions     : dict with keys smoke_dark, smoke_light,
                       background_bright, background_dark
    """
    raw  = segmenter.segment(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    classified = {
        "smoke_dark":        None,
        "smoke_light":       None,
        "background_bright": None,
        "background_dark":   None,
    }

    classified["background_bright"] = raw.get("sky")
    classified["background_dark"]   = raw.get("dark_background")

    smoke = raw.get("smoke")
    if smoke is not None:
        classified = _classify_smoke(smoke, gray, classified)
    else:
        # HSV fallback keys
        if raw.get("smoke_black") is not None:
            classified["smoke_dark"]  = raw["smoke_black"]
        if raw.get("smoke_white") is not None:
            classified["smoke_light"] = raw["smoke_white"]
        if raw.get("background_bright") is not None and classified["background_bright"] is None:
            classified["background_bright"] = raw["background_bright"]

    smoke_found = (classified["smoke_dark"] is not None
                   or classified["smoke_light"] is not None)
    print(f"[Segmenter] smoke={smoke_found} "
          f"(dark={classified['smoke_dark'] is not None}, "
          f"light={classified['smoke_light'] is not None})")

    return list(classified.values()), classified


def _classify_smoke(smoke_dict: dict, gray: np.ndarray, classified: dict) -> dict:
    seg        = smoke_dict["segmentation"]
    brightness = float(gray[seg].mean()) if seg.any() else 0.0
    smoke_copy = {**smoke_dict, "brightness": brightness}

    if brightness <= cfg.BRIGHTNESS_DARK_MAX:
        classified["smoke_dark"]  = smoke_copy
    else:
        # mid or bright brightness → light/grey/white smoke
        classified["smoke_light"] = smoke_copy

    return classified
