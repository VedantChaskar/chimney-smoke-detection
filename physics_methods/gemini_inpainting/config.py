# config.py — Gemini Inpainting: Synthetic Ground Truth via Smoke Inpainting
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from gemini_inpainting/ directory
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "../../assets/temporal_reference_test"
OUTPUT_FOLDER = "outputs"
MASKS_DIR     = os.path.join(OUTPUT_FOLDER, "masks")
INPAINTED_DIR = os.path.join(OUTPUT_FOLDER, "inpainted")
ANNOTATED_DIR = os.path.join(OUTPUT_FOLDER, "annotated")
RESULTS_CSV   = os.path.join(OUTPUT_FOLDER, "results.csv")

# ── SAM3 (local module in project root) ───────────────────────────────────────
SAM3_CHECKPOINT     = ""          # empty → auto-download from HuggingFace
SAM3_DEVICE         = "cuda"
SAM3_CONF_THRESHOLD = 0.15        # lowered to catch thin/transparent smoke

SAM3_PROMPTS = {
    "smoke":           ["smoke", "smoke plume", "dark smoke", "smoke cloud", "grey smoke"],
    "sky":             ["sky", "bright sky", "overcast sky", "white sky", "clouds",
                        "bright background", "light background"],
    "dark_background": ["rooftop", "roof", "chimney", "building", "dark roof",
                        "dark surface", "brick wall"],
}
MIN_MASK_AREA_RATIO = 0.01
MAX_MASK_AREA_RATIO = 0.90

# HSV fallback thresholds
HSV_SMOKE_MAX_SAT  = 50
HSV_SMOKE_MIN_VAL  = 40
BRIGHTNESS_DARK_MAX   = 85
BRIGHTNESS_BRIGHT_MIN = 170

# ── Smoke Validation ───────────────────────────────────────────────────────────
# Minimum SAM3 confidence score to consider a mask as real smoke
SMOKE_MIN_CONFIDENCE  = 0.15
# Minimum fraction of image area that must be smoke to justify inpainting
SMOKE_MIN_AREA_RATIO  = 0.015

# ── Mask Generation ────────────────────────────────────────────────────────────
# Dilation to cover smoke fringe pixels SAM3 may have missed
PIXEL_MASK_DILATION_PX = 8

# ── Gemini API ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Imagen models for the native edit_image API (EDIT_MODE_INPAINT_INSERTION).
# REQUIRES a Vertex AI client (vertexai=True), NOT a standard Gemini API key.
# When using a standard key, this stage fails gracefully and falls through
# to the generate_content path below.
GEMINI_EDIT_MODELS = [
    "imagen-3.0-capability-001",
    "imagen-3.0-capability-preview-0930",
]

# Gemini image-generation models for the visual-compositing generate_content path.
# Confirmed available via standard Gemini API key (verified via models.list()).
# Nano Banana = gemini-2.5-flash-image, Nano Banana 2 = gemini-3.1-flash-image-preview
GEMINI_MODELS = [
    "models/gemini-3.1-flash-image-preview",   # Nano Banana 2 (newest)
    "models/gemini-2.5-flash-image",            # Nano Banana
    "models/gemini-3-pro-image-preview",        # Nano Banana Pro
]

# Max side length for images sent to Gemini (reduces tokens / latency).
# The result is rescaled back to original size after inpainting.
GEMINI_MAX_SIDE = 1024

# Short prompt for the native Imagen edit API (prefers concise instructions)
INPAINTING_EDIT_PROMPT = (
    "Remove the smoke. Infill the area with clear sky while naturally continuing "
    "the texture of the chimney and roof."
)

# Longer prompt for the visual-compositing fallback (Gemini generate_content)
INPAINTING_PROMPT = (
    "You are given a chimney scene image. The bright magenta/pink-coloured region "
    "marks where smoke currently appears. Remove the smoke from that region entirely "
    "and replace it with the natural background — sky, rooftop, or whatever would be "
    "visible if no smoke were present. Match the surrounding lighting, colour temperature, "
    "and texture exactly. Do NOT add any objects, text, or watermarks. "
    "Output the full image with that region naturally reconstructed."
)

# ── Opacity Formula ────────────────────────────────────────────────────────────
ALPHA_DEFAULT        = 100
RINGELMAN_THRESHOLDS = [20, 40, 60, 80]   # grade 0=0%, 1=1-20%, 2=21-40%, 3=41-60%, 4=61-80%, 5=81-100%

# ── Annotation ─────────────────────────────────────────────────────────────────
ANNOTATION_FONT_SCALE = 0.9
ANNOTATION_THICKNESS  = 2
REGION_COLORS = {
    "smoke_dark":        (0,   0,   220),
    "smoke_light":       (220, 180, 0  ),
    "background_bright": (0,   200, 0  ),
    "background_dark":   (200, 0,   200),
}
