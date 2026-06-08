# src/logger.py
import os
import sys
from pathlib import Path

_METHOD2_DIR  = Path(__file__).parent.parent
_PROJECT_ROOT = _METHOD2_DIR.parent
sys.path.insert(0, str(_METHOD2_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "utils"))  # for csv_logger
import config as cfg
from csv_logger import init_csv as _init_csv, append_row as _append_row

_HEADERS = [
    "filename",
    "smoke_present", "smoke_type", "coverage_ratio",
    "opacity_pct", "ringelman", "formula_used",
    "inpaint_ok",
    "reason",
]


def init_csv() -> None:
    _init_csv(cfg.RESULTS_CSV, _HEADERS)


def log(
    filename: str,
    validation: dict,
    result: dict,
    inpaint_ok: bool,
) -> None:
    def _fmt(v):
        return round(v, 2) if v is not None else ""

    row = {
        "filename":       os.path.basename(filename),
        "smoke_present":  validation["smoke_present"],
        "smoke_type":     validation["smoke_type"],
        "coverage_ratio": round(validation["coverage_ratio"], 4),
        "opacity_pct":    _fmt(result.get("opacity_pct")),
        "ringelman":      result.get("ringelman", ""),
        "formula_used":   result.get("formula_used", ""),
        "inpaint_ok":     inpaint_ok,
        "reason":         (result.get("reason") or validation.get("reason", "")),
    }
    _append_row(cfg.RESULTS_CSV, _HEADERS, row)
