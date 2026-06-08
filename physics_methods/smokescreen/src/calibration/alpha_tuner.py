"""
src/calibration/alpha_tuner.py

Grid-search the calibration constant alpha that minimises MAE between
computed Ringelmann grades and manually labelled ground truth.

Usage
-----
from src.calibration.alpha_tuner import tune_alpha
best_alpha, results = tune_alpha(
    pairs=[
        (image_bgr_1, regions_1, true_grade_1),
        ...
    ]
)
print(f"Best alpha: {best_alpha}")
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg

from src.opacity.formula import compute_opacity
from src.opacity.ringelman import opacity_to_ringelman


def evaluate_alpha(
    pairs: list,
    alpha: float,
    bands: list = None,
) -> float:
    """
    Compute mean absolute error (MAE) for a given alpha value.

    Args:
        pairs:  list of (image_bgr, regions_dict, true_grade) tuples.
        alpha:  calibration constant to evaluate.
        bands:  Ringelmann band boundaries (default cfg.RINGELMAN_BANDS).

    Returns:
        MAE (float).
    """
    if not pairs:
        return float("inf")

    errors = []
    for image_bgr, regions, true_grade in pairs:
        result    = compute_opacity(image_bgr, regions, alpha=alpha)
        pred_grade = opacity_to_ringelman(result["opacity"], bands=bands)
        errors.append(abs(pred_grade - true_grade))

    return float(np.mean(errors))


def tune_alpha(
    pairs: list,
    search_range: tuple = None,
    step: int = None,
    bands: list = None,
    verbose: bool = True,
) -> tuple:
    """
    Grid-search over alpha values; return the one with the lowest MAE.

    Args:
        pairs:        list of (image_bgr, regions_dict, true_grade) tuples.
        search_range: (lo, hi) inclusive int bounds (default cfg.ALPHA_SEARCH_RANGE).
        step:         grid step (default cfg.ALPHA_SEARCH_STEP).
        bands:        Ringelmann band boundaries.
        verbose:      print progress.

    Returns:
        (best_alpha: float, results: list of (alpha, mae) sorted by alpha)
    """
    if search_range is None:
        search_range = cfg.ALPHA_SEARCH_RANGE
    if step is None:
        step = cfg.ALPHA_SEARCH_STEP

    lo, hi = search_range
    alphas  = range(int(lo), int(hi) + 1, int(step))
    results = []

    for alpha in alphas:
        mae = evaluate_alpha(pairs, float(alpha), bands=bands)
        results.append((float(alpha), mae))
        if verbose:
            print(f"  alpha={alpha:5.1f}  MAE={mae:.4f}")

    best_alpha, best_mae = min(results, key=lambda x: x[1])
    if verbose:
        print(f"\n  ✓ Best alpha = {best_alpha}  (MAE = {best_mae:.4f})")

    return best_alpha, results


def save_alpha(alpha: float, path: str = None) -> None:
    """Persist the chosen alpha to a text file for later use."""
    path = path or str(Path(cfg.DATA_DIR) / "best_alpha.txt")
    with open(path, "w") as f:
        f.write(str(alpha))
    print(f"  Saved alpha={alpha} → {path}")


def load_alpha(path: str = None) -> float:
    """Load a previously saved alpha value."""
    path = path or str(Path(cfg.DATA_DIR) / "best_alpha.txt")
    try:
        with open(path) as f:
            return float(f.read().strip())
    except FileNotFoundError:
        return float(cfg.ALPHA_DEFAULT)
