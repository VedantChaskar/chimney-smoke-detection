"""
Module 3a: Dark Channel Prior — transmittance estimation.

Reference: He, K., Sun, J., & Tang, X. (2009). Single image haze removal
using dark channel prior. CVPR.
"""

import numpy as np
import cv2


# ── Guided filter (pure NumPy/cv2, no ximgproc required) ─────────────────────

def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 60,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Edge-preserving guided filter (He et al., 2013).

    Both guide and src should be float32 in [0, 1].
    guide: (H, W)   — grayscale guide image
    src:   (H, W)   — signal to filter (e.g., coarse transmittance map)

    Returns:
        filtered: (H, W) float32
    """
    # Box-filter window size
    d = 2 * radius + 1

    def box(x):
        return cv2.boxFilter(x, ddepth=-1, ksize=(d, d),
                             normalize=True, borderType=cv2.BORDER_REFLECT)

    mean_I  = box(guide)
    mean_p  = box(src)
    mean_Ip = box(guide * src)
    mean_II = box(guide * guide)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a)
    mean_b = box(b)

    return (mean_a * guide + mean_b).astype(np.float32)


# ── Dark channel ──────────────────────────────────────────────────────────────

def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an image.

    The dark channel J^dark(x) = min_{c in {R,G,B}} min_{y in Ω(x)} J^c(y)

    Args:
        image: (H, W, 3) float32 in [0, 1] (any channel order).
        patch_size: Local patch size for the min-filter.

    Returns:
        dc: (H, W) float32 dark channel.
    """
    # Per-pixel minimum across channels
    min_channel = np.min(image, axis=2)
    # Local minimum over patch_size × patch_size neighbourhood
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    dc = cv2.erode(min_channel, kernel)   # erosion = local min
    return dc.astype(np.float32)


# ── Transmittance estimation ──────────────────────────────────────────────────

def estimate_transmittance(
    image: np.ndarray,
    A: np.ndarray,
    smoke_mask: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
    t_min: float = 0.1,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
) -> np.ndarray:
    """
    Estimate per-pixel transmittance t(x) using the Dark Channel Prior.

    t(x) = 1 - omega * dc( I(y) / A )   (He et al. Eq. 11)

    Then refined with a guided filter and clamped to [t_min, 1].
    Pixels outside smoke_mask are set to 1.0 (fully transmitting).

    Args:
        image:      (H, W, 3) uint8 or float32 [0,1 or 0,255] BGR/RGB.
        A:          (3,) float — atmospheric light, same channel order.
        smoke_mask: (H, W) binary mask (bool/uint8).
        omega:      Haze-keeping factor (0.95 keeps 5% haze for realism).
        patch_size: DCP patch size.
        t_min:      Minimum transmittance floor.
        guided_radius: Guided-filter radius.
        guided_eps:    Guided-filter regularisation.

    Returns:
        transmittance: (H, W) float32 in [t_min, 1].
    """
    # Normalise to [0, 1] float32
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    A_f = A.astype(np.float32)
    if A_f.max() > 1.5:
        A_f = A_f / 255.0

    # Avoid division by zero
    A_f = np.clip(A_f, 1e-6, 1.0)

    # Step 1: normalise by atmospheric light (per-channel)
    normalised = img / A_f[np.newaxis, np.newaxis, :]   # (H, W, 3)
    normalised = np.clip(normalised, 0.0, 1.0)

    # Step 2: dark channel of normalised image
    dc = dark_channel(normalised, patch_size)

    # Step 3: coarse transmittance
    t_coarse = 1.0 - omega * dc

    # Step 4: guided-filter refinement (guide = grayscale input)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    t_refined = guided_filter(gray_f, t_coarse, radius=guided_radius, eps=guided_eps)

    # Step 5: clamp
    t_refined = np.clip(t_refined, t_min, 1.0)

    # Step 6: outside smoke mask → fully transmitting
    mask = smoke_mask.astype(bool)
    t_refined[~mask] = 1.0

    return t_refined.astype(np.float32)


def transmittance_to_alpha(t: np.ndarray) -> np.ndarray:
    """Convert transmittance to opacity: alpha = 1 - t."""
    return (1.0 - t).astype(np.float32)
