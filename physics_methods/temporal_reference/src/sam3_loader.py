# src/sam3_loader.py  (method1 shim)
"""
Thin shim that delegates to the shared SAM3Segmenter in
src/inference/sam3_loader.py, supplying method1's config values.

Uses importlib to load the shared module by file path so that the method1
`src` namespace (active during testing) does not shadow the project root's
`src.inference` package.
"""

import sys
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent   # chimney-smoke-detection/
sys.path.insert(0, str(Path(__file__).parent.parent))   # method1/ → for config
import config as cfg

# Load the shared loader by absolute file path, bypassing `src` namespace.
_spec = importlib.util.spec_from_file_location(
    "_shared_sam3_loader",
    _ROOT / "src" / "inference" / "sam3_loader.py",
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


class SAM3Segmenter(_shared.SAM3Segmenter):
    """Method1 variant — uses conf_threshold=0.3, no vertical smoke guard."""

    def __init__(
        self,
        checkpoint: str = cfg.SAM3_CHECKPOINT,
        device: str = cfg.SAM3_DEVICE,
    ):
        super().__init__(
            checkpoint=checkpoint,
            device=device,
            conf_threshold=cfg.SAM3_CONF_THRESHOLD,
            prompts=cfg.SAM3_PROMPTS,
            min_area_ratio=cfg.MIN_MASK_AREA_RATIO,
            max_area_ratio=cfg.MAX_MASK_AREA_RATIO,
            brightness_bright_min=cfg.BRIGHTNESS_BRIGHT_MIN,
            brightness_dark_max=cfg.BRIGHTNESS_DARK_MAX,
            hsv_smoke_max_sat=cfg.HSV_SMOKE_MAX_SAT,
            hsv_smoke_min_val=cfg.HSV_SMOKE_MIN_VAL,
            smoke_vert_guard=False,  # intentionally absent in method1
        )


def load_segmenter(use_sam: bool = True) -> SAM3Segmenter:
    seg            = SAM3Segmenter.__new__(SAM3Segmenter)
    seg.device     = cfg.SAM3_DEVICE
    seg._model     = None
    seg._processor = None
    if use_sam:
        seg._load(cfg.SAM3_CHECKPOINT)
    return seg
