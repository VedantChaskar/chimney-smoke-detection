"""
src/segmentation/sam3_segmenter.py

SAM3-based region segmentation.

Design note
-----------
This project uses the local SAM3 model (Meta's open-vocabulary detector),
which operates through text and geometric *prompts* — there is no automatic
mask generation mode as in SAM1/SAM2.

Instead of SAM1's `SamAutomaticMaskGenerator`, we query SAM3 with focused
text prompts for each region type we need:

    "smoke"            → smoke plume
    "sky"              → bright background (sky)
    "rooftop" / "roof" → dark background (building surface)

Each query returns a binary mask and a confidence score.  The region
classifier then refines these into the four named types the opacity
formulas require (smoke_dark, smoke_light, background_bright, background_dark).

Fallback: if SAM3 is unavailable, brightness-based HSV thresholding is used.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# ── SAM3 path setup ────────────────────────────────────────────────────────────
# PROJECT_ROOT = chimney-smoke-detection/  (parent of both smokescreen/ and sam3/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
_SAM3_PATH    = _PROJECT_ROOT / "sam3"
# Insert PROJECT_ROOT so that `import sam3` finds the sam3/ package there.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_BPE_PATH = _SAM3_PATH / "assets" / "bpe_simple_vocab_16e6.txt.gz"

# ── Local config (relative import-safe) ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg


class SAM3Segmenter:
    """
    Wraps SAM3 for multi-region smoke scene segmentation.

    Queries SAM3 with separate text prompts for smoke, sky, and dark
    background; returns one mask dict per region type.

    If SAM3 is unavailable, all methods fall back to brightness thresholding.
    """

    def __init__(
        self,
        checkpoint: str = cfg.SAM3_CHECKPOINT,
        device: str = "cuda",
        conf_threshold: float = cfg.SAM3_CONF_THRESHOLD,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self._model = None
        self._processor = None
        self._load_sam3(checkpoint)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_sam3(self, checkpoint: str) -> None:
        try:
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            self._model = build_sam3_image_model(
                bpe_path=str(_BPE_PATH),
                device=self.device,
                eval_mode=True,
                checkpoint_path=checkpoint or None,
                load_from_HF=not bool(checkpoint),
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            self._processor = Sam3Processor(
                self._model,
                device=self.device,
                confidence_threshold=self.conf_threshold,
            )
            print("  ✓ SAM3 segmenter loaded")
        except Exception as e:
            print(f"  ⚠ SAM3 unavailable ({e}); using brightness fallback.")
            self._model = None
            self._processor = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def segment_all_regions(self, image_bgr: np.ndarray) -> dict:
        """
        Detect smoke, sky, and dark-background regions in one image.

        Returns a dict with keys:
            "smoke"           : mask dict (or None)
            "sky"             : mask dict (or None)
            "dark_background" : mask dict (or None)

        Each mask dict contains:
            "segmentation"    : (H, W) bool mask
            "area"            : int pixel count
            "brightness"      : float mean grayscale brightness
            "confidence"      : float SAM3 score
            "prompt_used"     : str
        """
        if self._processor is not None:
            return self._segment_sam3(image_bgr)
        return self._segment_fallback(image_bgr)

    # ── SAM3 path ──────────────────────────────────────────────────────────────

    def _segment_sam3(self, image_bgr: np.ndarray) -> dict:
        import torch
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        h, w = image_bgr.shape[:2]
        img_area = h * w

        results = {}
        prompt_groups = cfg.SAM3_PROMPTS

        with torch.inference_mode():
            with torch.autocast(
                self.device, dtype=torch.bfloat16,
                enabled=(self.device == "cuda")
            ):
                for region_key, prompts in prompt_groups.items():
                    mask_dict = self._query_best_mask(
                        pil, prompts, image_bgr, img_area
                    )
                    results[region_key] = mask_dict

        return results

    def _query_best_mask(
        self,
        pil_image,
        prompts: list[str],
        image_bgr: np.ndarray,
        img_area: int,
    ) -> Optional[dict]:
        """
        Try each prompt in order; return the first confident, size-valid mask.
        """
        import torch

        for prompt in prompts:
            state = self._processor.set_image(pil_image)
            state = self._processor.set_text_prompt(prompt=prompt, state=state)

            if "masks" not in state or state["masks"] is None:
                continue

            masks_t  = state["masks"].float().cpu().numpy()
            scores_t = (state["scores"].float().cpu().numpy()
                        if "scores" in state else np.ones(len(masks_t)))

            best = self._pick_best(masks_t, scores_t, image_bgr, img_area)
            if best is not None:
                best["prompt_used"] = prompt
                return best

        return None

    def _pick_best(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        image_bgr: np.ndarray,
        img_area: int,
    ) -> Optional[dict]:
        """Select the largest size-valid mask above confidence threshold."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        best = None
        best_area = 0

        for raw, score in zip(masks, scores):
            if float(score) < self.conf_threshold:
                continue
            m = raw[0] if raw.ndim == 3 else raw
            if m.max() > 1.01 or m.min() < -0.01:
                m = 1.0 / (1.0 + np.exp(-m))
            binary = (m > 0.5).astype(bool)
            area = int(binary.sum())
            ratio = area / img_area

            if ratio < cfg.MIN_MASK_AREA_RATIO:
                continue
            if ratio > cfg.MAX_MASK_AREA_RATIO:
                continue
            if area > best_area:
                best_area = area
                best = {
                    "segmentation": binary,
                    "area":         area,
                    "brightness":   float(gray[binary].mean()) if binary.any() else 0.0,
                    "confidence":   float(score),
                }
        return best

    # ── Brightness / HSV fallback ──────────────────────────────────────────────

    def _segment_fallback(self, image_bgr: np.ndarray) -> dict:
        """
        Fallback segmentation using HSV colour thresholds when SAM3 is off.

        Smoke:           low saturation, mid-to-high value (grey/white region)
        Sky:             high value in the upper half of the image
        Dark background: low value in the lower half of the image
        """
        h, w = image_bgr.shape[:2]
        img_area = h * w
        hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        def make_dict(binary: np.ndarray, prompt: str) -> Optional[dict]:
            area = int(binary.sum())
            if area / img_area < cfg.MIN_MASK_AREA_RATIO:
                return None
            return {
                "segmentation": binary.astype(bool),
                "area":         area,
                "brightness":   float(gray[binary].mean()) if binary.any() else 0.0,
                "confidence":   0.5,
                "prompt_used":  prompt,
            }

        # Dark smoke: low saturation, low-to-mid brightness (dark grey / black)
        smoke_dark_mask = (
            (hsv[:, :, 1] <= cfg.HSV_SMOKE_MAX_SAT) &
            (hsv[:, :, 2] >= cfg.HSV_SMOKE_MIN_VAL) &
            (gray <= cfg.BRIGHTNESS_DARK_MAX)
        )
        # White/grey smoke: low saturation, mid-to-high brightness
        smoke_white_mask = (
            (hsv[:, :, 1] <= cfg.HSV_SMOKE_MAX_SAT) &
            (hsv[:, :, 2] >= cfg.HSV_SMOKE_MIN_VAL) &
            (gray > cfg.BRIGHTNESS_DARK_MAX)
        )
        # Bright background: high brightness, upper half
        bright_mask = (gray >= cfg.BRIGHTNESS_BRIGHT_MIN)
        bright_mask[h // 2:] = False

        # Dark background: low brightness, lower half
        dark_mask = (gray <= cfg.BRIGHTNESS_DARK_MAX)
        dark_mask[:h // 2] = False

        return {
            "smoke_black":        make_dict(smoke_dark_mask,  "hsv_fallback_dark"),
            "smoke_white":        make_dict(smoke_white_mask, "hsv_fallback_white"),
            "background_bright":  make_dict(bright_mask,      "brightness_threshold"),
            "dark_background":    make_dict(dark_mask,        "brightness_threshold"),
        }


def load_segmenter(
    checkpoint: str = cfg.SAM3_CHECKPOINT,
    device: str = "cuda",
    use_sam: bool = True,
) -> SAM3Segmenter:
    """
    Convenience factory. Set use_sam=False to force HSV fallback
    (useful for unit testing without GPU).
    """
    seg = SAM3Segmenter.__new__(SAM3Segmenter)
    seg.device         = device
    seg.conf_threshold = cfg.SAM3_CONF_THRESHOLD
    seg._model         = None
    seg._processor     = None
    if use_sam:
        seg._load_sam3(checkpoint)
    return seg
