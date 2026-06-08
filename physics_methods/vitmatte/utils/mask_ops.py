"""Morphological operations and mask utilities."""

import numpy as np
import cv2


def morphological_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = (mask.astype(np.uint8)) * 255
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return (closed > 0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 0:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel)


def erode_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 0:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1)
    )
    return cv2.erode(mask.astype(np.uint8), kernel)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if num_labels <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def mask_perimeter(mask: np.ndarray) -> float:
    """Compute mask boundary perimeter in pixels."""
    m = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sum(len(c) for c in contours)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
