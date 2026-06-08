# src/smoke_segmenter.py
"""
Smoke segmentation for the aligned smoke image.

Strategy (two-stage):
  1. SAM3 text-prompt → primary smoke mask (most precise boundaries).
  2. Temporal difference fallback — if SAM3 finds nothing, compare the aligned
     smoke image pixel-by-pixel against the ground truth.  Regions that changed
     significantly (and are in the upper portion of the frame) are smoke.

This second stage is particularly faithful to Method 1's premise: we HAVE the
clean reference frame, so we can directly detect what changed.

Final classification into four types for the opacity formula:
    smoke_dark        — dark / black smoke
    smoke_light       — white / grey smoke (including bright plumes)
    background_bright — sky, bright surfaces
    background_dark   — rooftop, chimney, trees
"""

from __future__ import annotations

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from .sam3_loader import SAM3Segmenter
from .ground_truth import GroundTruth


def segment_smoke_image(
    image_bgr: np.ndarray,
    segmenter: SAM3Segmenter,
    ground_truth: GroundTruth | None = None,
) -> tuple[dict, dict]:
    """
    Segment and classify a smoke image.

    Args:
        image_bgr    : aligned smoke image (H, W, 3)
        segmenter    : loaded SAM3Segmenter
        ground_truth : GroundTruth object (used for temporal-diff fallback)

    Returns
    -------
    classified_masks / best_regions : dict with keys
        smoke_dark, smoke_light, background_bright, background_dark
        Each value is a mask dict or None.
    """
    raw  = segmenter.segment(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = image_bgr.shape[:2]

    classified = {
        "smoke_dark":        None,
        "smoke_light":       None,
        "background_bright": None,
        "background_dark":   None,
    }

    # ── Background regions from SAM3 ─────────────────────────────────────────
    classified["background_bright"] = raw.get("sky")
    classified["background_dark"]   = raw.get("dark_background")

    # ── Smoke mask — SAM3 primary ─────────────────────────────────────────────
    smoke = raw.get("smoke")
    if smoke is None:
        # HSV fallback keys (when SAM3 is disabled)
        smoke_black = raw.get("smoke_black")
        smoke_white = raw.get("smoke_white")
        if smoke_black is not None:
            classified["smoke_dark"]  = smoke_black
        if smoke_white is not None:
            classified["smoke_light"] = smoke_white
        if raw.get("background_bright") is not None:
            classified["background_bright"] = raw["background_bright"]
    else:
        classified = _classify_smoke_mask(smoke, gray, classified)

    # ── Temporal-difference fallback ─────────────────────────────────────────
    # If SAM3 found no smoke AND we have the ground truth, use pixel-level
    # temporal comparison to detect changed regions.
    smoke_found = (classified["smoke_dark"] is not None
                   or classified["smoke_light"] is not None)

    if not smoke_found and ground_truth is not None:
        td_mask = _temporal_diff_mask(image_bgr, ground_truth, h, w)
        if td_mask is not None:
            classified = _classify_smoke_mask(td_mask, gray, classified)
            if (classified["smoke_dark"] is not None
                    or classified["smoke_light"] is not None):
                print("[SmokeSegmenter] smoke found via temporal-diff fallback")

    smoke_found = (classified["smoke_dark"] is not None
                   or classified["smoke_light"] is not None)
    print(f"[SmokeSegmenter] smoke detected: {smoke_found} "
          f"(dark={classified['smoke_dark'] is not None}, "
          f"light={classified['smoke_light'] is not None})")

    return classified, classified


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_smoke_mask(smoke_dict: dict, gray: np.ndarray, classified: dict) -> dict:
    """Sub-classify a smoke mask dict into smoke_dark or smoke_light."""
    seg        = smoke_dict["segmentation"]
    brightness = float(gray[seg].mean()) if seg.any() else 0.0
    smoke_copy = {**smoke_dict, "brightness": brightness}

    if brightness <= cfg.BRIGHTNESS_DARK_MAX:
        classified["smoke_dark"]  = smoke_copy
    elif brightness >= cfg.BRIGHTNESS_BRIGHT_MIN:
        # Very bright region — if it's SAM3-found smoke that outshone sky,
        # treat as light smoke (dense white plume)
        classified["smoke_light"] = smoke_copy
    else:
        classified["smoke_light"] = smoke_copy

    return classified


def _temporal_diff_mask(
    smoke_bgr: np.ndarray,
    ground_truth: GroundTruth,
    h: int,
    w: int,
) -> dict | None:
    """
    Detect smoke by finding pixels that changed significantly between the
    ground truth and the smoke image.

    Steps:
      1. Convert both to grayscale.
      2. Compute absolute per-pixel difference.
      3. Keep pixels with difference > threshold AND in upper 70% of frame
         (smoke rises; the lower portion is typically solid structure).
      4. Morphological cleanup (close small gaps, remove tiny specks).
      5. Return the largest connected component as the smoke mask dict.
    """
    img_area = h * w
    gt_gray  = ground_truth.gray

    # Resize GT gray if needed (should match after alignment, but just in case)
    if gt_gray.shape != smoke_bgr.shape[:2]:
        gt_gray = cv2.resize(gt_gray, (w, h), interpolation=cv2.INTER_LINEAR)

    smoke_gray = cv2.cvtColor(smoke_bgr, cv2.COLOR_BGR2GRAY)
    diff       = cv2.absdiff(smoke_gray, gt_gray)

    # Adaptive threshold — use Otsu on the diff image (often bimodal)
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Restrict to upper 70% (smoke is above the chimney line)
    binary[int(h * 0.70):] = 0

    # Morphological close — fill gaps within smoke plume
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Remove small noise blobs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Pick largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return None

    # stats[0] is background — skip it
    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    best_area  = stats[best_label, cv2.CC_STAT_AREA]

    if best_area / img_area < cfg.MIN_MASK_AREA_RATIO:
        return None
    if best_area / img_area > cfg.MAX_MASK_AREA_RATIO:
        return None

    seg       = (labels == best_label)
    gray_vals = cv2.cvtColor(smoke_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "segmentation": seg.astype(bool),
        "area":         int(best_area),
        "brightness":   float(gray_vals[seg].mean()),
        "confidence":   0.5,
        "prompt_used":  "temporal_diff",
    }
