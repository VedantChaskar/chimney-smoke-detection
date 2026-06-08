"""
src/segmentation/region_classifier.py

Classifies the three SAM3-detected regions (smoke, sky, dark_background)
into the four named types the opacity formulas need:

    smoke_dark        — dark / black smoke plume
    smoke_light       — white / grey smoke plume
    background_bright — sky or any bright surface
    background_dark   — rooftop, trees, or any dark surface

The smoke region is classified as dark vs light based on mean brightness.
Sky and dark-background are renamed to the formula's expected keys.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg


# ── Edge / texture helper ──────────────────────────────────────────────────────

def get_edge_variance(image_bgr: np.ndarray, mask_dict: dict) -> float:
    """
    Mean absolute Laplacian response inside a mask.

    Smoke is diffuse → low edge variance.
    Solid objects (chimney, rooftop) are sharp → high edge variance.
    """
    if mask_dict is None:
        return 0.0
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    seg   = mask_dict["segmentation"]
    vals  = np.abs(edges[seg])
    return float(vals.mean()) if len(vals) > 0 else 0.0


def get_vertical_center(mask_dict: dict, image_height: int) -> float:
    """Normalised vertical centre of a mask (0 = top, 1 = bottom)."""
    if mask_dict is None:
        return 0.5
    ys = np.where(mask_dict["segmentation"])[0]
    return float(np.mean(ys)) / image_height if len(ys) > 0 else 0.5


# ── Classifier ─────────────────────────────────────────────────────────────────

def classify_regions(image_bgr: np.ndarray, raw_regions: dict) -> dict:
    """
    Convert SAM3-detected regions into the four named region types.

    SAM3 now returns smoke type directly via descriptive text prompts
    (smoke_black / smoke_white), so no brightness-based reclassification
    is needed for smoke — just annotate with edge_variance / vertical_center
    and rename to the formula keys (smoke_dark / smoke_light).

    Args:
        image_bgr:   (H, W, 3) BGR image.
        raw_regions: dict from SAM3Segmenter.segment_all_regions(), with keys:
                       "smoke_black", "smoke_white",
                       "background_bright", "dark_background"
                     Each value is a mask dict (or None).

    Returns:
        dict with keys:
            smoke_dark        : mask dict or None
            smoke_light       : mask dict or None
            background_bright : mask dict or None
            background_dark   : mask dict or None
        Each mask dict has the same structure as SAM3 output, plus
        "region_type", "edge_variance", "vertical_center" added.
    """
    h = image_bgr.shape[0]
    result = {
        "smoke_dark":        None,
        "smoke_light":       None,
        "background_bright": None,
        "background_dark":   None,
    }

    # ── smoke_black → smoke_dark ──────────────────────────────────────────────
    smoke_black = raw_regions.get("smoke_black")
    if smoke_black is not None:
        ev = get_edge_variance(image_bgr, smoke_black)
        vc = get_vertical_center(smoke_black, h)
        result["smoke_dark"] = {
            **smoke_black,
            "edge_variance":  ev,
            "vertical_center": vc,
            "region_type":    "smoke_dark",
        }

    # ── smoke_white → smoke_light ─────────────────────────────────────────────
    smoke_white = raw_regions.get("smoke_white")
    if smoke_white is not None:
        ev = get_edge_variance(image_bgr, smoke_white)
        vc = get_vertical_center(smoke_white, h)
        result["smoke_light"] = {
            **smoke_white,
            "edge_variance":  ev,
            "vertical_center": vc,
            "region_type":    "smoke_light",
        }

    # ── Keep only the higher-confidence smoke type ────────────────────────────
    # When SAM3 returns both smoke_black and smoke_white for the same plume,
    # the one with lower confidence is likely a mis-detection.  Keeping both
    # would let the pipeline apply the wrong formula.
    if result["smoke_dark"] is not None and result["smoke_light"] is not None:
        conf_dark  = result["smoke_dark"].get("confidence",  0.0)
        conf_light = result["smoke_light"].get("confidence", 0.0)
        if conf_dark >= conf_light:
            result["smoke_light"] = None
        else:
            result["smoke_dark"] = None

    # ── background_bright (passed through) ───────────────────────────────────
    bg_bright = raw_regions.get("background_bright")
    if bg_bright is not None:
        ev = get_edge_variance(image_bgr, bg_bright)
        vc = get_vertical_center(bg_bright, h)
        result["background_bright"] = {
            **bg_bright,
            "edge_variance":  ev,
            "vertical_center": vc,
            "region_type":    "background_bright",
        }

    # ── dark_background → background_dark ────────────────────────────────────
    dark_bg = raw_regions.get("dark_background")
    if dark_bg is not None:
        ev = get_edge_variance(image_bgr, dark_bg)
        vc = get_vertical_center(dark_bg, h)
        result["background_dark"] = {
            **dark_bg,
            "edge_variance":  ev,
            "vertical_center": vc,
            "region_type":    "background_dark",
        }

    # ── Fallback: derive missing backgrounds from image statistics ────────────
    result = _fill_missing_backgrounds(image_bgr, result)

    return result


def _fill_missing_backgrounds(image_bgr: np.ndarray, result: dict) -> dict:
    """
    If SAM3 couldn't find a background region, derive it from the image:
      - bright background: top-25% rows, high-brightness pixels
      - dark background:   bottom-25% rows, low-brightness pixels
    Only fills regions that are still None.
    """
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape

    if result["background_bright"] is None:
        # Upper quarter of image, above brightness threshold
        region = np.zeros((h, w), dtype=bool)
        region[:h // 4, :] = gray[:h // 4, :] >= cfg.BRIGHTNESS_BRIGHT_MIN
        if region.sum() > 100:
            result["background_bright"] = {
                "segmentation":  region,
                "area":          int(region.sum()),
                "brightness":    float(gray[region].mean()),
                "confidence":    0.0,
                "prompt_used":   "image_stats_fallback",
                "edge_variance": 0.0,
                "vertical_center": 0.1,
                "region_type":   "background_bright",
            }

    if result["background_dark"] is None:
        # Lower quarter of image, below brightness threshold
        region = np.zeros((h, w), dtype=bool)
        region[3 * h // 4:, :] = gray[3 * h // 4:, :] <= cfg.BRIGHTNESS_DARK_MAX
        if region.sum() > 100:
            result["background_dark"] = {
                "segmentation":  region,
                "area":          int(region.sum()),
                "brightness":    float(gray[region].mean()),
                "confidence":    0.0,
                "prompt_used":   "image_stats_fallback",
                "edge_variance": 0.0,
                "vertical_center": 0.9,
                "region_type":   "background_dark",
            }

    return result


def summarise_regions(regions: dict) -> dict:
    """Return a JSON-serialisable summary of detected regions."""
    summary = {}
    for k, v in regions.items():
        if v is None:
            summary[k] = None
        else:
            summary[k] = {
                "area":           v["area"],
                "brightness":     round(v["brightness"], 1),
                "confidence":     round(v.get("confidence", 0.0), 3),
                "prompt_used":    v.get("prompt_used", ""),
                "edge_variance":  round(v.get("edge_variance", 0.0), 2),
                "vertical_center": round(v.get("vertical_center", 0.5), 3),
            }
    return summary
