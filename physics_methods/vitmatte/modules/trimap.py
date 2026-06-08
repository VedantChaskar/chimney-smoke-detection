"""
Module 2: Trimap Generation.

Converts a SAM3 binary/soft mask into a trimap for alpha matting:
    0   = definite background
    128 = unknown (transition zone — matting resolves this)
    255 = definite foreground
"""

import numpy as np
import cv2

from ..utils.mask_ops import dilate_mask, erode_mask, mask_perimeter


# ── Standard trimap ────────────────────────────────────────────────────────────

def generate_trimap(
    binary_mask: np.ndarray,
    soft_mask: np.ndarray = None,
    erosion_size: int = 10,
    dilation_size: int = 20,
    soft_mask_low: float = 0.2,
    soft_mask_high: float = 0.8,
) -> np.ndarray:
    """
    Generate a trimap from a binary (and optional soft) mask.

    Steps:
        1. Erode binary_mask by erosion_size  → definite_fg  (255)
        2. Dilate binary_mask by dilation_size → expanded
        3. unknown = expanded AND NOT definite_fg             (128)
        4. background = NOT expanded                          (0)
        5. If soft_mask provided: expand unknown to include
           pixels where soft_mask ∈ [soft_mask_low, soft_mask_high].

    Args:
        binary_mask:    (H, W) bool/uint8.
        soft_mask:      (H, W) float32 [0,1], SAM3 sigmoid output (optional).
        erosion_size:   Pixels to erode inward for definite FG.
        dilation_size:  Pixels to dilate outward for unknown zone edge.
        soft_mask_low:  Lower bound of soft-mask uncertainty range.
        soft_mask_high: Upper bound of soft-mask uncertainty range.

    Returns:
        trimap: (H, W) uint8 with values {0, 128, 255}.
    """
    m = binary_mask.astype(np.uint8)

    definite_fg = erode_mask(m, erosion_size)
    expanded    = dilate_mask(m, dilation_size)

    trimap = np.zeros_like(m, dtype=np.uint8)
    trimap[expanded.astype(bool)]    = 128   # unknown covers dilated region
    trimap[definite_fg.astype(bool)] = 255   # FG overrides

    # Expand unknown using soft mask uncertainty
    if soft_mask is not None:
        uncertain = ((soft_mask >= soft_mask_low) &
                     (soft_mask <= soft_mask_high))
        # Only expand unknown — never override definite_fg
        trimap[uncertain & (trimap != 255)] = 128

    return trimap


# ── Adaptive trimap ────────────────────────────────────────────────────────────

def adaptive_trimap(
    binary_mask: np.ndarray,
    soft_mask: np.ndarray,
    image: np.ndarray,
    soft_mask_low: float = 0.2,
    soft_mask_high: float = 0.8,
) -> np.ndarray:
    """
    Automatically choose erosion/dilation sizes based on:
        - Mask boundary complexity (perimeter/sqrt(area) ratio).
        - Edge gradient along the mask boundary in the input image.

    Gradual/complex boundaries → wider unknown zone.
    Sharp/simple boundaries → narrower unknown zone.

    Returns trimap: (H, W) uint8 with values {0, 128, 255}.
    """
    m = binary_mask.astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return np.zeros_like(m, dtype=np.uint8)

    # ── Complexity ratio ──────────────────────────────────────────────────
    perim = mask_perimeter(m)
    complexity = perim / max(np.sqrt(area), 1.0)
    # complexity ~12 for a perfect circle; higher for irregular shapes

    # ── Boundary gradient ─────────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # Boundary band: pixels at mask edge
    boundary_ring = (dilate_mask(m, 3).astype(bool) &
                     ~erode_mask(m, 3).astype(bool))
    mean_grad = float(grad[boundary_ring].mean()) if boundary_ring.any() else 30.0

    # ── Derive erosion / dilation sizes ───────────────────────────────────
    # Base sizes relative to sqrt(area)
    base = max(5, int(np.sqrt(area) * 0.04))   # ~4% of plume diameter

    # Complexity factor: more complex → wider unknown zone
    complexity_factor = np.clip(complexity / 20.0, 0.5, 2.5)

    # Gradient factor: lower gradient (gradual edge) → wider zone
    # mean_grad ~30 for typical images; smoke edges are typically 10-50
    gradient_factor = np.clip(40.0 / (mean_grad + 5.0), 0.5, 3.0)

    erosion_size  = max(3,  int(base * 0.8))
    dilation_size = max(8, int(base * complexity_factor * gradient_factor))
    # Cap to avoid over-segmentation
    dilation_size = min(dilation_size, 60)

    return generate_trimap(
        m, soft_mask,
        erosion_size=erosion_size,
        dilation_size=dilation_size,
        soft_mask_low=soft_mask_low,
        soft_mask_high=soft_mask_high,
    )


# ── Utilities ──────────────────────────────────────────────────────────────────

def trimap_stats(trimap: np.ndarray) -> dict:
    """Return fraction of each trimap class."""
    total = trimap.size
    return {
        "fg_fraction":      float((trimap == 255).sum()) / total,
        "unknown_fraction": float((trimap == 128).sum()) / total,
        "bg_fraction":      float((trimap == 0).sum())   / total,
    }


def visualise_trimap(trimap: np.ndarray) -> np.ndarray:
    """Return a colour-coded BGR image of the trimap for debugging."""
    vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
    vis[trimap == 0]   = (50,  50,  50)   # dark grey — BG
    vis[trimap == 128] = (0,  165, 255)   # orange    — unknown
    vis[trimap == 255] = (0,  200,  0)    # green     — FG
    return vis
