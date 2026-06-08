"""
Morphological operations and mask utilities.
"""

import numpy as np
import cv2


def morphological_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply morphological closing to a binary mask to fill small holes.

    Args:
        mask: (H, W) bool or uint8 binary mask.
        ksize: Square kernel size.

    Returns:
        (H, W) uint8 mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = mask.astype(np.uint8) * 255
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return (closed > 0).astype(np.uint8)


def morphological_open(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply morphological opening to remove small islands."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = mask.astype(np.uint8) * 255
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return (opened > 0).astype(np.uint8)


def compute_boundary_band(mask: np.ndarray, width: int = 25) -> np.ndarray:
    """
    Compute the boundary band around a binary mask:
        boundary = dilate(mask, width) AND NOT mask

    This gives a ring of pixels just outside the smoke region, used as a
    contrast/color reference for the unattenuated scene.

    Args:
        mask: (H, W) binary mask (bool or uint8).
        width: Dilation radius in pixels.

    Returns:
        (H, W) uint8 boundary band mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1)
    )
    m = mask.astype(np.uint8)
    dilated = cv2.dilate(m, kernel)
    boundary = dilated & (~m.astype(bool)).astype(np.uint8)
    return boundary.astype(np.uint8)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    Args:
        mask: (H, W) uint8 or bool.

    Returns:
        (H, W) uint8 mask with only the largest component.
    """
    m = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if num_labels <= 1:
        return m
    # stats[:, cv2.CC_STAT_AREA] gives area; skip background (label 0)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def dilate_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Dilate a binary mask by ksize pixels."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel)


def erode_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Erode a binary mask by ksize pixels."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1)
    )
    return cv2.erode(mask.astype(np.uint8), kernel)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return tight bounding box (x1, y1, x2, y2) of a binary mask.
    Returns (0, 0, 0, 0) if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
