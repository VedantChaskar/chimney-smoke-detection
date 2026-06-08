"""
smokescreen/config.py — All constants and tunable parameters.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
TEST_SET_DIR = os.path.join(DATA_DIR, "test_set")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Project root (for SAM3 path resolution)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# ── SAM3 ───────────────────────────────────────────────────────────────────────
# Local SAM3 is the project's text-prompt model (not SAM1 automatic mask generator)
SAM3_CHECKPOINT = ""           # empty → download from HuggingFace
SAM3_CONF_THRESHOLD = 0.25     # minimum confidence for a detected region

# Text prompts used to query each region type.
# Each list is tried in order; the first successful detection is used.
# Keys map directly to the four named region types the opacity formulas need.
SAM3_PROMPTS = {
    "smoke_black": [
        "The visible dark/black smoke plume coming out of the chimney.",
        "dark smoke plume from chimney",
        "black smoke coming out of chimney",
    ],
    "smoke_white": [
        "The visible white or gray smoke plume coming out of the chimney.",
        "white smoke plume from chimney",
        "gray smoke coming out of chimney",
    ],
    "background_bright": [
        "A clean patch of clear sky right next to the smoke — no smoke in it. Same width as your smoke box.",
        "clear sky next to chimney smoke",
        "bright sky background near smoke",
    ],
    "dark_background": [
        "A patch of dark surface visible behind or near the smoke — such as a rooftop edge, dark trees, or a dark wall. Only draw this when clearly visible.",
        "dark rooftop or trees behind chimney smoke",
        "dark wall or roof near chimney",
    ],
}

# ── Region classification (brightness-based) ─────────────────────────────────
# These thresholds are applied to the SAM3-detected smoke region
# to decide whether it is dark smoke or light smoke.
BRIGHTNESS_DARK_MAX    = 85    # smoke brightness below this → smoke_dark
BRIGHTNESS_BRIGHT_MIN  = 170   # background brightness above this → background_bright
# smoke with brightness in (85, 170) → smoke_light (grey/white smoke)

# Fallback HSV-based segmentation (when SAM3 is unavailable)
HSV_SMOKE_MAX_SAT  = 60    # saturation threshold for smoke (low-sat regions)
HSV_SMOKE_MIN_VAL  = 80    # value (brightness) threshold

# Minimum mask size as fraction of image area (ignore tiny spurious detections)
MIN_MASK_AREA_RATIO  = 0.005   # 0.5%
MAX_MASK_AREA_RATIO  = 0.85    # ignore masks covering almost the whole image

# ── Opacity formula ───────────────────────────────────────────────────────────
ALPHA_DEFAULT      = 100       # calibration constant — tuned against test set
ALPHA_SEARCH_RANGE = (70, 130) # grid search bounds
ALPHA_SEARCH_STEP  = 2

# ── Ringelman scale ───────────────────────────────────────────────────────────
# Grade boundaries (opacity %) — grade = 0 below first, 5 above last
RINGELMAN_BANDS = [10, 30, 50, 70, 90]
# Grade 0: < 10%   (clear)
# Grade 1: 10–30%  (light)
# Grade 2: 30–50%  (medium-light)
# Grade 3: 50–70%  (medium-dark)
# Grade 4: 70–90%  (dark)
# Grade 5: > 90%   (opaque / black)
