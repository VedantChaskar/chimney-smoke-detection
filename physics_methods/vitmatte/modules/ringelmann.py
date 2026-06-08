"""
Module 5: Alpha → Ringelmann Mapping.

Converts a per-pixel alpha map to a Ringelmann number (0–5).

Standard equivalences:
    R0 = 0%   opacity  (clear)
    R1 = 20%  opacity  (light grey)
    R2 = 40%  opacity  (medium)
    R3 = 60%  opacity  (medium dark)
    R4 = 80%  opacity  (dark)
    R5 = 100% opacity  (black / impenetrable)

Decision boundaries at midpoints: [0.10, 0.30, 0.50, 0.70, 0.90]
"""

import numpy as np

RINGELMANN_CENTERS    = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
RINGELMANN_BOUNDARIES = [0.10, 0.30, 0.50, 0.70, 0.90]


def alpha_to_ringelmann(alpha: float) -> int:
    """Map a single alpha value in [0,1] to Ringelmann 0-5."""
    for i, boundary in enumerate(RINGELMANN_BOUNDARIES):
        if alpha < boundary:
            return i
    return 5


def alpha_to_ringelmann_float(alpha: float) -> float:
    """Continuous Ringelmann scale: R = 5 * alpha, rounded to 2 dp."""
    return round(5.0 * float(alpha), 2)


def compute_ringelmann(
    alpha_map: np.ndarray,
    smoke_mask: np.ndarray,
    method: str = "density_weighted",
) -> dict:
    """
    Summarise a per-pixel alpha map into a Ringelmann number.

    Args:
        alpha_map:   (H, W) float32 [0,1].
        smoke_mask:  (H, W) bool — smoke region.
        method:      One of:
                       "mean"              — mean of all smoke pixels
                       "p75"               — 75th percentile
                       "density_weighted"  — sum(α²)/sum(α)  [recommended]
                       "peak_region"       — mean of densest 10% pixels

    Returns dict with:
        ringelmann          : int 0-5
        ringelmann_float    : float (continuous)
        representative_alpha: float (value used for mapping)
        method_used         : str
        spatial_stats       : dict of per-pixel statistics
    """
    mask = smoke_mask.astype(bool)
    if not mask.any():
        return _empty_result(method)

    vals = alpha_map[mask]

    # ── Spatial statistics ────────────────────────────────────────────────
    spatial_stats = {
        "mean_alpha":        float(vals.mean()),
        "median_alpha":      float(np.median(vals)),
        "p75_alpha":         float(np.percentile(vals, 75)),
        "p90_alpha":         float(np.percentile(vals, 90)),
        "std_alpha":         float(vals.std()),
        "coverage_fraction": float((vals > 0.05).sum()) / len(vals),
    }

    # ── Representative alpha by method ────────────────────────────────────
    if method == "mean":
        rep_alpha = spatial_stats["mean_alpha"]

    elif method == "p75":
        rep_alpha = spatial_stats["p75_alpha"]

    elif method == "density_weighted":
        # sum(α²) / sum(α) — emphasises denser regions smoothly
        alpha_sum = float(vals.sum())
        if alpha_sum < 1e-9:
            rep_alpha = 0.0
        else:
            rep_alpha = float((vals ** 2).sum()) / alpha_sum

    elif method == "peak_region":
        # Mean of the top-10% alpha pixels (or pixels > 0.5, whichever is larger)
        threshold = max(float(np.percentile(vals, 90)), 0.5)
        peak_vals = vals[vals >= threshold]
        rep_alpha = float(peak_vals.mean()) if peak_vals.size > 0 else spatial_stats["p90_alpha"]

    else:
        raise ValueError(f"Unknown Ringelmann method: '{method}'. "
                         "Choose from: mean, p75, density_weighted, peak_region")

    rep_alpha = float(np.clip(rep_alpha, 0.0, 1.0))

    return {
        "ringelmann":           alpha_to_ringelmann(rep_alpha),
        "ringelmann_float":     alpha_to_ringelmann_float(rep_alpha),
        "representative_alpha": rep_alpha,
        "method_used":          method,
        "spatial_stats":        spatial_stats,
    }


def _empty_result(method: str) -> dict:
    return {
        "ringelmann":           0,
        "ringelmann_float":     0.0,
        "representative_alpha": 0.0,
        "method_used":          method,
        "spatial_stats": {
            "mean_alpha": 0.0, "median_alpha": 0.0,
            "p75_alpha":  0.0, "p90_alpha":    0.0,
            "std_alpha":  0.0, "coverage_fraction": 0.0,
        },
    }
