"""
src/opacity/brightness.py

Brightness extraction from masked image regions.
"""

import numpy as np
import cv2
from typing import Optional


def get_region_brightness(image_bgr: np.ndarray, mask_dict: Optional[dict]) -> Optional[float]:
    """
    Returns mean grayscale brightness (0–255) of all pixels within a mask.

    Args:
        image_bgr: (H, W, 3) BGR image.
        mask_dict: Region mask dict with key "segmentation" (H, W) bool array,
                   or None.

    Returns:
        float in [0, 255], or None if mask is None or empty.
    """
    if mask_dict is None:
        return None

    seg = mask_dict["segmentation"]
    if not seg.any():
        return None

    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[seg]
    return float(np.mean(pixels))


def get_region_brightness_stats(
    image_bgr: np.ndarray, mask_dict: Optional[dict]
) -> Optional[dict]:
    """
    Returns brightness statistics (mean, median, std, p10, p90) for a region.
    Useful for debugging and calibration.
    """
    if mask_dict is None:
        return None

    seg = mask_dict["segmentation"]
    if not seg.any():
        return None

    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[seg].astype(np.float32)

    return {
        "mean":   float(np.mean(pixels)),
        "median": float(np.median(pixels)),
        "std":    float(np.std(pixels)),
        "p10":    float(np.percentile(pixels, 10)),
        "p90":    float(np.percentile(pixels, 90)),
        "count":  int(len(pixels)),
    }
