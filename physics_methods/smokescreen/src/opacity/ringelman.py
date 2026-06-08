"""
src/opacity/ringelman.py

Maps opacity percentage (0–100) to a Ringelmann grade (0–5).

Standard Ringelmann chart:
    Grade 0:  0%   — clear
    Grade 1: 20%   — light grey
    Grade 2: 40%   — medium grey
    Grade 3: 60%   — dark grey
    Grade 4: 80%   — very dark
    Grade 5: 100%  — black (opaque)

The config RINGELMAN_BANDS list defines the *upper boundaries* for grades 0–4;
anything above the last boundary is grade 5.

Default bands: [10, 30, 50, 70, 90]
    opacity < 10  → grade 0
    10 ≤ o < 30   → grade 1
    30 ≤ o < 50   → grade 2
    50 ≤ o < 70   → grade 3
    70 ≤ o < 90   → grade 4
    o ≥ 90        → grade 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg


def opacity_to_ringelman(opacity_pct: float, bands: list = None) -> int:
    """
    Convert opacity percentage to a Ringelmann grade (0–5).

    Args:
        opacity_pct: opacity in [0, 100].
        bands:       list of 5 ascending boundary values (default cfg.RINGELMAN_BANDS).

    Returns:
        int in [0, 5].
    """
    if bands is None:
        bands = cfg.RINGELMAN_BANDS

    for grade, upper in enumerate(bands):
        if opacity_pct < upper:
            return grade
    return 5


def ringelman_label(grade: int) -> str:
    """Human-readable label for a Ringelmann grade."""
    labels = {
        0: "Clear (0)",
        1: "Light smoke (1)",
        2: "Medium-light smoke (2)",
        3: "Medium-dark smoke (3)",
        4: "Dark smoke (4)",
        5: "Black / opaque smoke (5)",
    }
    return labels.get(grade, f"Grade {grade}")


def ringelman_description(grade: int) -> dict:
    """Return a summary dict for a Ringelmann grade."""
    return {
        "grade":         grade,
        "label":         ringelman_label(grade),
        "opacity_range": _grade_range(grade),
    }


def _grade_range(grade: int, bands: list = None) -> str:
    bands = bands or cfg.RINGELMAN_BANDS
    if grade == 0:
        return f"< {bands[0]}%"
    elif grade >= 5:
        return f"≥ {bands[-1]}%"
    else:
        return f"{bands[grade - 1]}–{bands[grade]}%"
