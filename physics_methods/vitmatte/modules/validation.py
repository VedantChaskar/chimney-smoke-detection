"""
Module 4: Alpha Map Validation & Quality Gate.

Runs sanity checks on the ViTMatte output to catch failure modes
(bad trimap, model collapse, low-contrast input, etc.).
Does NOT estimate opacity — it validates the alpha map.
"""

import numpy as np
import cv2

from ..utils.mask_ops import dilate_mask


# ── Top-level validator ────────────────────────────────────────────────────────

def validate_alpha_map(
    image: np.ndarray,
    alpha_map: np.ndarray,
    smoke_mask: np.ndarray,
    trimap: np.ndarray,
) -> dict:
    """
    Run all validation checks and return a consolidated quality report.

    Returns:
        overall_quality  : "high" | "medium" | "low"
        confidence_score : float [0, 1]
        checks           : dict of per-check results
        warnings         : list[str]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    checks = {
        "brightness_consistency": check_brightness_consistency(
            gray, alpha_map, smoke_mask
        ),
        "edge_coherence":         check_edge_coherence(image, alpha_map),
        "alpha_distribution":     check_alpha_distribution(alpha_map, smoke_mask),
        "image_quality":          check_image_quality(image, smoke_mask),
    }

    warnings = []
    score_components = []

    # Brightness consistency
    bc = checks["brightness_consistency"]
    if not bc["consistent"]:
        warnings.append(
            f"Brightness correlation low ({bc['correlation']:.2f}); "
            "matting may be unreliable on this background."
        )
        score_components.append(0.4)
    else:
        score_components.append(1.0)

    # Edge coherence
    ec = checks["edge_coherence"]
    if not ec["coherent"]:
        warnings.append(
            f"Alpha edges don't align with image edges "
            f"(alignment={ec['alignment_score']:.2f}); possible trimap issue."
        )
        score_components.append(0.5)
    else:
        score_components.append(1.0)

    # Alpha distribution
    ad = checks["alpha_distribution"]
    if not ad["plausible"]:
        warnings.append(
            "Alpha distribution is implausible (no gradation or all-zero/all-one). "
            "Try widening the trimap unknown zone."
        )
        score_components.append(0.2)
    elif not ad["has_gradation"]:
        warnings.append("Alpha map lacks gradation (std very low); may be a hard mask.")
        score_components.append(0.6)
    else:
        score_components.append(1.0)

    # Image quality
    iq = checks["image_quality"]
    if iq["overall"] == "poor":
        warnings.append("Poor input quality (too dark/bright, compressed, or smoke too small/large).")
        score_components.append(0.3)
    elif iq["overall"] == "marginal":
        warnings.append("Marginal input quality; results may be less accurate.")
        score_components.append(0.7)
    else:
        score_components.append(1.0)

    confidence_score = float(np.mean(score_components))

    if confidence_score >= 0.80:
        overall_quality = "high"
    elif confidence_score >= 0.55:
        overall_quality = "medium"
    else:
        overall_quality = "low"

    return {
        "overall_quality":  overall_quality,
        "confidence_score": confidence_score,
        "checks":           checks,
        "warnings":         warnings,
    }


# ── Individual checks ──────────────────────────────────────────────────────────

def check_brightness_consistency(
    image_gray: np.ndarray,
    alpha_map: np.ndarray,
    smoke_mask: np.ndarray,
    corr_threshold: float = 0.3,
) -> dict:
    """
    Higher alpha should correlate with brightness deviation from background.

    Uses boundary pixels as the background estimate.
    """
    mask  = smoke_mask.astype(bool)
    boundary = (dilate_mask(smoke_mask.astype(np.uint8), 20).astype(bool) & ~mask)

    if not mask.any() or not boundary.any():
        return {"correlation": 0.0, "smoke_type": "unknown", "consistent": False}

    gray = image_gray.astype(np.float32)
    bg_mean = float(gray[boundary].mean())

    # Deviation from background inside smoke
    deviation = np.abs(gray[mask] - bg_mean)
    alpha_vals = alpha_map[mask]

    # Smoke type: does the smoke make things darker or brighter?
    smoke_mean = float(gray[mask].mean())
    smoke_type = "dark" if smoke_mean < bg_mean else "light"

    if len(alpha_vals) < 10:
        return {"correlation": 0.0, "smoke_type": smoke_type, "consistent": False}

    # Pearson correlation between alpha and |deviation from background|
    corr_matrix = np.corrcoef(alpha_vals, deviation)
    corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

    return {
        "correlation": corr,
        "smoke_type":  smoke_type,
        "consistent":  corr >= corr_threshold,
        "bg_mean":     bg_mean,
        "smoke_mean":  smoke_mean,
    }


def check_edge_coherence(
    image: np.ndarray,
    alpha_map: np.ndarray,
    alpha_grad_threshold: float = 0.05,
    coherence_ratio_threshold: float = 1.5,
) -> dict:
    """
    Alpha gradient edges should align with image gradient edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def sobel_mag(x):
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    alpha_grad = sobel_mag(alpha_map)
    image_grad = sobel_mag(gray)

    high_alpha_grad = alpha_grad > alpha_grad_threshold
    low_alpha_grad  = ~high_alpha_grad

    if not high_alpha_grad.any() or not low_alpha_grad.any():
        return {"alignment_score": 1.0, "coherent": True}

    mean_img_at_alpha_edge    = float(image_grad[high_alpha_grad].mean())
    mean_img_at_non_alpha_edge = float(image_grad[low_alpha_grad].mean())

    alignment_score = (mean_img_at_alpha_edge /
                       (mean_img_at_non_alpha_edge + 1e-6))

    return {
        "alignment_score": alignment_score,
        "coherent":        alignment_score >= coherence_ratio_threshold,
    }


def check_alpha_distribution(
    alpha_map: np.ndarray,
    smoke_mask: np.ndarray,
    min_std: float = 0.05,
) -> dict:
    """
    Check that the alpha distribution is physically plausible for smoke.
    """
    mask = smoke_mask.astype(bool)
    if not mask.any():
        return {
            "mean": 0.0, "std": 0.0, "has_gradation": False,
            "plausible": False, "coverage_fraction": 0.0,
        }

    vals = alpha_map[mask]
    mean = float(vals.mean())
    std  = float(vals.std())
    coverage = float((vals > 0.05).sum()) / len(vals)

    # "Has gradation": meaningful spread of values
    has_gradation = std >= min_std

    # Plausibility checks
    all_zero    = mean < 0.02
    all_one     = mean > 0.98
    no_mid_range = float(((vals > 0.2) & (vals < 0.8)).sum()) / len(vals) < 0.05

    plausible = (not all_zero) and (not all_one) and has_gradation

    return {
        "mean":              mean,
        "std":               std,
        "has_gradation":     has_gradation,
        "plausible":         plausible,
        "coverage_fraction": coverage,
        "all_zero":          all_zero,
        "all_one":           all_one,
        "no_mid_range":      no_mid_range,
    }


def check_image_quality(
    image: np.ndarray,
    smoke_mask: np.ndarray,
    min_mask_ratio: float = 0.005,
    max_mask_ratio: float = 0.80,
    min_bg_std: float = 10.0,
) -> dict:
    """
    Pre-condition checks for reliable opacity estimation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    n_pixels = h * w
    mask = smoke_mask.astype(bool)

    brightness_ok = 40 <= float(gray.mean()) <= 240
    mask_ratio = float(mask.sum()) / n_pixels
    mask_size_ok = min_mask_ratio <= mask_ratio <= max_mask_ratio

    # Background contrast
    bg = ~mask
    bg_std = float(gray[bg].std()) if bg.any() else 0.0
    bg_contrast_ok = bg_std >= min_bg_std

    issues = []
    if not brightness_ok:
        issues.append("brightness")
    if not mask_size_ok:
        issues.append(f"mask_size ({mask_ratio*100:.1f}%)")
    if not bg_contrast_ok:
        issues.append(f"low_bg_contrast (std={bg_std:.1f})")

    if not issues:
        overall = "good"
    elif len(issues) == 1:
        overall = "marginal"
    else:
        overall = "poor"

    return {
        "brightness_ok":     brightness_ok,
        "mask_size_ok":      mask_size_ok,
        "bg_contrast_ok":    bg_contrast_ok,
        "mask_ratio":        mask_ratio,
        "bg_std":            bg_std,
        "overall":           overall,
        "issues":            issues,
    }
