"""
Module 4: Weighted Fusion & Ringelmann Mapping.

Combines per-method opacity estimates into a single fused alpha and maps
it to a Ringelmann number (0–5).
"""

import numpy as np

# Default Ringelmann thresholds: alpha lower bound for each level.
# R0: [0.00, 0.10), R1: [0.10, 0.25), R2: [0.25, 0.45),
# R3: [0.45, 0.65), R4: [0.65, 0.80), R5: [0.80, 1.00]
RINGELMANN_THRESHOLDS = [0.0, 0.10, 0.25, 0.45, 0.65, 0.80]

DEFAULT_WEIGHTS = {
    "dark_channel":  0.60,
    "contrast":      0.25,
    "color_collapse": 0.15,
}


def alpha_to_ringelmann(
    alpha: float,
    thresholds: list[float] = RINGELMANN_THRESHOLDS,
) -> int:
    """Map a fused opacity alpha in [0, 1] to Ringelmann 0–5."""
    for i in range(5, -1, -1):
        if alpha >= thresholds[i]:
            return i
    return 0


def fuse_opacity_estimates(
    alpha_dark_channel: float,
    alpha_contrast: float,
    alpha_color_collapse: float,
    weights: dict = None,
    confidence_flags: dict = None,
    thresholds: list[float] = RINGELMANN_THRESHOLDS,
) -> dict:
    """
    Combine three opacity estimates into a single fused score and
    Ringelmann number.

    Args:
        alpha_dark_channel:  Opacity from dark channel prior (0–1).
        alpha_contrast:      Opacity from contrast attenuation (0–1).
        alpha_color_collapse: Opacity from color channel analysis (0–1).
        weights:             Dict with keys "dark_channel", "contrast",
                             "color_collapse".  Defaults to DEFAULT_WEIGHTS.
        confidence_flags:    Dict with optional boolean fields:
                               "contrast_reliable"      (bool)
                               "color_collapse_reliable" (bool)
                             Unreliable methods have their weight
                             redistributed to dark_channel.
        thresholds:          Ringelmann threshold list (6 values).

    Returns:
        dict with keys:
            fused_alpha  : float in [0, 1]
            ringelmann   : int  in [0, 5]
            confidence   : str  "high" | "medium" | "low"
            per_method   : dict {dark_channel, contrast, color_collapse}
            weights_used : dict — final weights after reliability adjustment
    """
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)
    else:
        weights = dict(weights)

    flags = confidence_flags or {}

    # ── Reliability-based weight redistribution ───────────────────────────
    extra = 0.0
    if not flags.get("contrast_reliable", True):
        extra += weights["contrast"]
        weights["contrast"] = 0.0
    if not flags.get("color_collapse_reliable", True):
        extra += weights["color_collapse"]
        weights["color_collapse"] = 0.0
    weights["dark_channel"] += extra

    # ── Weighted average ──────────────────────────────────────────────────
    estimates = {
        "dark_channel":  alpha_dark_channel,
        "contrast":      alpha_contrast,
        "color_collapse": alpha_color_collapse,
    }

    total_w = sum(weights.values())
    if total_w < 1e-9:
        fused = alpha_dark_channel          # fallback
    else:
        fused = sum(estimates[k] * weights[k] for k in estimates) / total_w

    fused = float(np.clip(fused, 0.0, 1.0))

    # ── Ringelmann mapping ────────────────────────────────────────────────
    ringelmann = alpha_to_ringelmann(fused, thresholds)

    # ── Confidence ───────────────────────────────────────────────────────
    active_estimates = [
        estimates[k] for k, w in weights.items() if w > 0
    ]
    if len(active_estimates) >= 2:
        std = float(np.std(active_estimates))
    else:
        std = 0.0

    if std > 0.25:
        confidence = "low"
    elif std > 0.12:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "fused_alpha":  fused,
        "ringelmann":   ringelmann,
        "confidence":   confidence,
        "per_method":   estimates,
        "weights_used": weights,
    }
