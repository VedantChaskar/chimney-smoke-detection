"""
Central configuration for the Smoke Opacity Estimation pipeline.
All tunable parameters live here.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Project root (parent of this file's package)
PROJECT_ROOT = Path(__file__).parent.parent
SAM3_PATH = PROJECT_ROOT / "sam3"
BPE_PATH = SAM3_PATH / "assets" / "bpe_simple_vocab_16e6.txt.gz"


@dataclass
class Config:
    # ── SAM3 ────────────────────────────────────────────────────────────────
    sam_checkpoint: str = ""            # leave empty → downloads from HuggingFace
    sam_model_type: str = "sam3"
    device: str = "cuda"
    sam_conf_threshold: float = 0.25   # confidence threshold for SAM3 detections

    # ── Smoke colour filter (HSV) ─────────────────────────────────────────
    smoke_max_saturation: int = 60     # S channel (0-255 scale)
    smoke_min_value: int = 100         # V channel

    # ── Morphological post-processing ────────────────────────────────────
    morph_close_ksize: int = 5         # kernel for closing binary mask
    boundary_width: int = 25           # width of boundary band in pixels

    # ── Dark Channel Prior ────────────────────────────────────────────────
    dcp_patch_size: int = 15
    dcp_omega: float = 0.95
    dcp_t_min: float = 0.1             # floor on transmittance
    guided_filter_radius: int = 60
    guided_filter_eps: float = 1e-3

    # ── Contrast module ───────────────────────────────────────────────────
    contrast_reliable_threshold: float = 10.0   # min boundary gradient magnitude

    # ── Fusion weights ────────────────────────────────────────────────────
    weight_dark_channel: float = 0.60
    weight_contrast: float = 0.25
    weight_color_collapse: float = 0.15

    # ── Ringelmann thresholds (opacity alpha → Ringelmann 0-5) ───────────
    # R0: [0, 0.10), R1: [0.10, 0.25), R2: [0.25, 0.45),
    # R3: [0.45, 0.65), R4: [0.65, 0.80), R5: [0.80, 1.0]
    ringelmann_thresholds: list = field(
        default_factory=lambda: [0.0, 0.10, 0.25, 0.45, 0.65, 0.80]
    )

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "outputs/smoke_opacity"
    save_panel: bool = True
    save_overlay: bool = True
    save_json: bool = True


# Module-level default instance (import and mutate as needed)
DEFAULT_CONFIG = Config()
