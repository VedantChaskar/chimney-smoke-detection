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
    "method_2a_opacity", "method_2a_ringelman", "method_2a_formula",
    "method_2b_opacity", "method_2b_ringelman", "method_2b_formula",
    "method_2a_inpaint_ok", "method_2b_inpaint_ok",
    "reason",
]


def init_csv() -> None:
    _init_csv(cfg.RESULTS_CSV, _HEADERS)


def log(
    filename: str,
    validation: dict,
    result_2a: dict,
    result_2b: dict,
    inpaint_ok_2a: bool,
    inpaint_ok_2b: bool,
) -> None:
    def _fmt(v):
        return round(v, 2) if v is not None else ""

    row = {
        "filename":              os.path.basename(filename),
        "smoke_present":         validation["smoke_present"],
        "smoke_type":            validation["smoke_type"],
        "coverage_ratio":        round(validation["coverage_ratio"], 4),
        "method_2a_opacity":     _fmt(result_2a.get("opacity_pct")),
        "method_2a_ringelman":   result_2a.get("ringelman", ""),
        "method_2a_formula":     result_2a.get("formula_used", ""),
        "method_2b_opacity":     _fmt(result_2b.get("opacity_pct")),
        "method_2b_ringelman":   result_2b.get("ringelman", ""),
        "method_2b_formula":     result_2b.get("formula_used", ""),
        "method_2a_inpaint_ok":  inpaint_ok_2a,
        "method_2b_inpaint_ok":  inpaint_ok_2b,
        "reason":                (result_2a.get("reason") or validation.get("reason", "")),
    }
    _append_row(cfg.RESULTS_CSV, _HEADERS, row)
