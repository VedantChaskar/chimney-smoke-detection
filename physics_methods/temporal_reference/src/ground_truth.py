# src/ground_truth.py
"""
Processes the ground truth (no-smoke) image once at startup.
Produces a GroundTruth object that all subsequent smoke image comparisons reference.

The key idea (paper's temporal approach):
    L_background is NOT taken from adjacent regions within the smoke image.
    It is taken from the SAME pixel coordinates in the clean ground truth image.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .sam3_loader import SAM3Segmenter
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class GroundTruth:
    """
    Reference data derived from the no-smoke ground truth image.

    Attributes
    ----------
    image_bgr   : (H, W, 3) raw image
    gray        : (H, W) grayscale
    raw_masks   : dict from SAM3Segmenter.segment() — smoke/sky/dark_background
    L_bright    : mean brightness of sky/bright region (float or None)
    L_dark      : mean brightness of dark background region (float or None)
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        raw_masks: dict,
    ) -> None:
        self.image_bgr = image_bgr
        self.gray      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.raw_masks = raw_masks

        sky_mask  = raw_masks.get("sky")
        dark_mask = raw_masks.get("dark_background")

        self.L_bright = self._mean_at(sky_mask)
        self.L_dark   = self._mean_at(dark_mask)

    # ── Core temporal-comparison method (paper's approach) ────────────────────

    def get_brightness_at_mask(self, smoke_mask_seg: np.ndarray) -> float | None:
        """
        Return mean brightness of ground-truth pixels at the given mask locations.

        This is the paper's temporal comparison: instead of sampling background
        from a neighboring region in the same smoke image, we look up those
        exact pixels in the clean reference frame.

        Args:
            smoke_mask_seg: bool (H, W) array, same spatial size as ground truth

        Returns:
            float mean gray value, or None if mask is empty
        """
        if smoke_mask_seg is None or not smoke_mask_seg.any():
            return None

        # Resize mask if smoke image was slightly different resolution
        if smoke_mask_seg.shape != self.gray.shape:
            smoke_mask_seg = cv2.resize(
                smoke_mask_seg.astype(np.uint8),
                (self.gray.shape[1], self.gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        pixels = self.gray[smoke_mask_seg]
        return float(np.mean(pixels)) if len(pixels) > 0 else None

    # ── Private ───────────────────────────────────────────────────────────────

    def _mean_at(self, mask_dict: dict | None) -> float | None:
        if mask_dict is None:
            return None
        seg    = mask_dict["segmentation"]
        pixels = self.gray[seg]
        return float(np.mean(pixels)) if len(pixels) > 0 else None


def load_ground_truth(image_path: str, segmenter: SAM3Segmenter) -> GroundTruth:
    """
    Load and segment the ground truth image.

    Args:
        image_path : path to ground_truth.jpg
        segmenter  : loaded SAM3Segmenter

    Returns:
        GroundTruth object
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Ground truth image not found: {image_path}")

    print(f"[GroundTruth] Loaded {image_path} — shape {image.shape}")

    raw_masks = segmenter.segment(image)

    sky_found  = raw_masks.get("sky") is not None
    dark_found = raw_masks.get("dark_background") is not None
    print(f"[GroundTruth] sky (bright bg) found: {sky_found}")
    print(f"[GroundTruth] dark_background found:  {dark_found}")

    return GroundTruth(image, raw_masks)
