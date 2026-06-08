# src/logger.py
"""
CSV logging — one row per processed image.
"""

import os
import sys
from pathlib import Path

_METHOD1_DIR  = Path(__file__).parent.parent
_PROJECT_ROOT = _METHOD1_DIR.parent
sys.path.insert(0, str(_METHOD1_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "utils"))  # for csv_logger
import config as cfg
from csv_logger import init_csv as _init_csv, append_row as _append_row


_HEADERS = [
    "filename",
    "smoke_type",
    "opacity_pct",
    "ringelman",
    "formula_used",
    "alignment_success",
    "num_alignment_matches",
    "reason",
]


def init_csv() -> None:
    """Create CSV with headers. Overwrites any existing file."""
    _init_csv(cfg.RESULTS_CSV, _HEADERS)


def log_result(
    filename: str,
    result: dict,
    alignment_success: bool,
    num_matches: int,
) -> None:
    """Append one result row."""
    row = {
        "filename":              os.path.basename(filename),
        "smoke_type":            result.get("smoke_type", ""),
        "opacity_pct":           (round(result["opacity_pct"], 2)
                                  if result["opacity_pct"] is not None else ""),
        "ringelman":             result.get("ringelman", ""),
        "formula_used":          result.get("formula_used", ""),
        "alignment_success":     alignment_success,
        "num_alignment_matches": num_matches,
        "reason":                result.get("reason", ""),
    }
    _append_row(cfg.RESULTS_CSV, _HEADERS, row)
