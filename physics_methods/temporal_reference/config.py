# config.py — Temporal Reference: Stationary Camera Temporal Opacity Detection
import os
from pathlib import Path

# ── Project root (chimney-smoke-detection/) ────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "../../assets/temporal_reference_test"
OUTPUT_FOLDER = "outputs"
ANNOTATED_DIR = os.path.join(OUTPUT_FOLDER, "annotated")
RESULTS_CSV   = os.path.join(OUTPUT_FOLDER, "results.csv")

GROUND_TRUTH_FILENAME = "ground_truth.jpg"

# ── SAM3 Model ─────────────────────────────────────────────────────────────────
# Leave empty to auto-download from HuggingFace on first run.
SAM3_CHECKPOINT = ""
SAM3_DEVICE     = "cuda"

# Confidence threshold for SAM3 text-prompt masks.
# Real chimney-smoke images typically score 0.3–0.5; 0.5 misses most detections.
SAM3_CONF_THRESHOLD = 0.3

# Text prompts per region type — tried in order; first confident mask is used.
SAM3_PROMPTS = {
    "smoke":            ["smoke", "smoke plume", "dark smoke", "smoke cloud", "grey smoke"],
    "sky":              ["sky", "bright sky", "overcast sky", "white sky", "clouds",
                         "bright background", "light background"],
    "dark_background":  ["rooftop", "roof", "chimney", "building", "dark roof",
                         "dark surface", "brick wall"],
}

# Mask area guard-rails (fraction of total image area)
MIN_MASK_AREA_RATIO = 0.01   # ignore tiny fragments
MAX_MASK_AREA_RATIO = 0.90   # ignore masks that cover almost the whole image

# ── Image Alignment ────────────────────────────────────────────────────────────
ALIGNMENT_MAX_FEATURES     = 5000
ALIGNMENT_MATCH_RATIO      = 0.75
ALIGNMENT_MIN_MATCHES      = 10

# ── Region Classification Thresholds ──────────────────────────────────────────
# Grayscale brightness (0–255)
BRIGHTNESS_DARK_MAX   = 85
BRIGHTNESS_BRIGHT_MIN = 170

# Edge sharpness (Laplacian mean abs) — higher = sharper = solid object, not smoke
EDGE_VARIANCE_THRESH = 15.0

# HSV smoke detection (fallback, no SAM3)
HSV_SMOKE_MAX_SAT = 50   # low saturation = grey / white smoke
HSV_SMOKE_MIN_VAL = 40   # avoid pure black (shadow)

# ── Opacity Formula ────────────────────────────────────────────────────────────
ALPHA_DEFAULT = 100

# ── Ringelman Scale Bands ──────────────────────────────────────────────────────
# Opacity % thresholds — grade 0 = below [0], grade 5 = above [4]
RINGELMAN_THRESHOLDS = [10, 30, 50, 70, 90]

# ── Annotation Visual Settings ────────────────────────────────────────────────
ANNOTATION_FONT_SCALE = 1.2
ANNOTATION_THICKNESS  = 2
REGION_COLORS = {
    "smoke_dark":        (0,   0,   220),   # Red
    "smoke_light":       (220, 180, 0  ),   # Cyan
    "background_bright": (0,   200, 0  ),   # Green
    "background_dark":   (200, 0,   200),   # Magenta
}
