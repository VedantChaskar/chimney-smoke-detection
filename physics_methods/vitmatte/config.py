"""
Central configuration for the VitMatte smoke opacity pipeline.
All tunable parameters live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
SAM3_PATH    = PROJECT_ROOT / "sam3"
BPE_PATH     = SAM3_PATH / "assets" / "bpe_simple_vocab_16e6.txt.gz"

# HuggingFace model cache (avoid re-downloading on every run)
HF_CACHE_DIR = PROJECT_ROOT / "models" / "vitmatte_cache"


@dataclass
class Config:
    # ── SAM3 ─────────────────────────────────────────────────────────────
    sam_checkpoint: str = ""           # empty → HuggingFace download
    sam_conf_threshold: float = 0.25
    device: str = "cuda"

    # Smoke colour filter (HSV, 0-255 scale)
    smoke_max_saturation: int = 60
    smoke_min_value:      int = 100
    morph_close_ksize:    int = 5

    # ── Trimap ───────────────────────────────────────────────────────────
    trimap_erosion_size:   int   = 10    # px eroded for definite FG
    trimap_dilation_size:  int   = 20    # px dilated for unknown zone
    trimap_soft_mask_low:  float = 0.2   # SAM3 soft mask: uncertain range
    trimap_soft_mask_high: float = 0.8
    trimap_adaptive:       bool  = True  # auto-compute erosion/dilation

    # ── Alpha Matting (ViTMatte) ─────────────────────────────────────────
    matting_model:     str = "hustvl/vitmatte-small-composition-1k"
    matting_max_size:  int = 1024        # longest edge resize
    matting_bilateral_d:     int   = 5
    matting_bilateral_sigma: float = 0.1

    # ── Validation ───────────────────────────────────────────────────────
    brightness_corr_threshold: float = 0.3
    edge_coherence_threshold:  float = 1.5
    min_alpha_std:             float = 0.05
    min_mask_ratio:            float = 0.005   # 0.5% of image
    max_mask_ratio:            float = 0.80

    # ── Ringelmann ───────────────────────────────────────────────────────
    ringelmann_method: str = "density_weighted"
    ringelmann_boundaries: List[float] = field(
        default_factory=lambda: [0.10, 0.30, 0.50, 0.70, 0.90]
    )

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "outputs/vitmatte"
    save_panel:   bool = True
    save_overlay: bool = True
    save_alpha:   bool = True
    save_json:    bool = True


DEFAULT_CONFIG = Config()
