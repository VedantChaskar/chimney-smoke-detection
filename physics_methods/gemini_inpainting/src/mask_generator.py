# src/mask_generator.py
"""
Two inpainting mask types for Method 2.

Method 2a — Pixel mask:
    Exact SAM3 segmentation, slightly dilated to cover fringe pixels.

Method 2b — Bounding box mask:
    Rectangle encompassing the smoke region, slightly expanded.
    Guarantees all smoke is covered even if SAM3 under-segmented.

Both masks are uint8 arrays: 255 = inpaint, 0 = keep.
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
    Method 2a: pixel-level mask from SAM3 segmentation.

    Returns (H, W) uint8 — 255 = inpaint, 0 = keep.
    """
    H, W  = image_shape[:2]
    seg   = smoke_mask_dict["segmentation"].astype(np.uint8) * 255

    if cfg.PIXEL_MASK_DILATION_PX > 0:
        ksize  = cfg.PIXEL_MASK_DILATION_PX * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        seg    = cv2.dilate(seg, kernel, iterations=1)

    return seg


def generate_bbox_mask(smoke_mask_dict: dict, image_shape: tuple) -> np.ndarray:
    """
    Method 2b: bounding box mask computed from the segmentation.

    The SAM3 text-prompt API may not always populate a 'bbox' field, so we
    always derive the bounding box from the boolean segmentation mask directly.

    Returns (H, W) uint8 — 255 = inpaint, 0 = keep.
    """
    H, W  = image_shape[:2]
    seg   = smoke_mask_dict["segmentation"].astype(bool)

    ys, xs = np.where(seg)
    if len(xs) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    bw = x_max - x_min
    bh = y_max - y_min
    expand_x = int(bw * cfg.BBOX_MASK_EXPAND_RATIO)
    expand_y = int(bh * cfg.BBOX_MASK_EXPAND_RATIO)

    x1 = max(0, x_min - expand_x)
    y1 = max(0, y_min - expand_y)
    x2 = min(W, x_max + expand_x + 1)
    y2 = min(H, y_max + expand_y + 1)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def get_bbox_coords(smoke_mask_dict: dict, image_shape: tuple) -> tuple[int, int, int, int]:
    """
    Return (x1, y1, x2, y2) expanded bbox coordinates.
    Used by the annotator.
    """
    H, W  = image_shape[:2]
    seg   = smoke_mask_dict["segmentation"].astype(bool)
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return (0, 0, W, H)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bw = x_max - x_min
    bh = y_max - y_min
    expand_x = int(bw * cfg.BBOX_MASK_EXPAND_RATIO)
    expand_y = int(bh * cfg.BBOX_MASK_EXPAND_RATIO)
    return (
        max(0, x_min - expand_x),
        max(0, y_min - expand_y),
        min(W, x_max + expand_x + 1),
        min(H, y_max + expand_y + 1),
    )
