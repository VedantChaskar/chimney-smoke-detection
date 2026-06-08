# src/mask_generator.py
"""
Pixel-level inpainting mask for Method 2.

Exact SAM3 segmentation, slightly dilated to cover fringe pixels.
Mask is a uint8 array: 255 = inpaint, 0 = keep.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def generate_pixel_mask(smoke_mask_dict: dict, image_shape: tuple) -> np.ndarray:
    """
    Pixel-level mask from SAM3 segmentation.

    Returns (H, W) uint8 — 255 = inpaint, 0 = keep.
    """
    H, W  = image_shape[:2]
    seg   = smoke_mask_dict["segmentation"].astype(np.uint8) * 255

    if cfg.PIXEL_MASK_DILATION_PX > 0:
        ksize  = cfg.PIXEL_MASK_DILATION_PX * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        seg    = cv2.dilate(seg, kernel, iterations=1)

    return seg
