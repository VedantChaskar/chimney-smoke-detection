# src/inference/sam3_loader.py
"""
Shared SAM3 text-prompt segmenter with HSV fallback.

All configuration values are injected through the constructor so this module
has no dependency on any subproject's config.py.  Thin shims in
method1/src/sam3_loader.py and method2/src/sam3_loader.py pass each
subproject's config values when subclassing.

Class-level attribute defaults mirror method2's config (the superset), which
allows the __new__-based stub pattern used by run.py fallback helpers to work
without calling __init__.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── sam3 package path ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent   # chimney-smoke-detection/
_SAM3_PATH    = _PROJECT_ROOT / "sam3"
_BPE_PATH     = _SAM3_PATH / "assets" / "bpe_simple_vocab_16e6.txt.gz"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_PROMPTS = {
    "smoke":           ["smoke", "smoke plume", "dark smoke", "smoke cloud", "grey smoke"],
    "sky":             ["sky", "bright sky", "overcast sky", "white sky", "clouds",
                        "bright background", "light background"],
    "dark_background": ["rooftop", "roof", "chimney", "building", "dark roof",
                        "dark surface", "brick wall"],
}


class SAM3Segmenter:
    """
    Text-prompt SAM3 segmenter with HSV fallback.

    Call segment(image_bgr) to get a dict with keys:
        "smoke"           -> mask dict or None
        "sky"             -> mask dict or None
        "dark_background" -> mask dict or None

    Each mask dict:
        segmentation  : (H, W) bool array
        area          : int
        brightness    : float (mean gray)
        confidence    : float
        prompt_used   : str
    """

    # Class-level defaults — used when __new__ bypasses __init__ (stub/fallback mode)
    device               = "cuda"
    _conf_threshold      = 0.15
    _prompts             = _DEFAULT_PROMPTS
    _min_area_ratio      = 0.01
    _max_area_ratio      = 0.90
    _brightness_bright_min = 170
    _brightness_dark_max   = 85
    _hsv_smoke_max_sat   = 50
    _hsv_smoke_min_val   = 40
    _smoke_vert_guard    = False
    _model               = None
    _processor           = None

    def __init__(
        self,
        checkpoint: str = "",
        device: str = "cuda",
        conf_threshold: float = 0.15,
        prompts: dict | None = None,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.90,
        brightness_bright_min: int = 170,
        brightness_dark_max: int = 85,
        hsv_smoke_max_sat: int = 50,
        hsv_smoke_min_val: int = 40,
        smoke_vert_guard: bool = False,
    ):
        self.device = device
        self._conf_threshold       = conf_threshold
        self._prompts              = prompts or _DEFAULT_PROMPTS
        self._min_area_ratio       = min_area_ratio
        self._max_area_ratio       = max_area_ratio
        self._brightness_bright_min = brightness_bright_min
        self._brightness_dark_max   = brightness_dark_max
        self._hsv_smoke_max_sat    = hsv_smoke_max_sat
        self._hsv_smoke_min_val    = hsv_smoke_min_val
        self._smoke_vert_guard     = smoke_vert_guard
        self._model     = None
        self._processor = None
        self._load(checkpoint)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load(self, checkpoint: str) -> None:
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
                confidence_threshold=self._conf_threshold,
            )
            print("[SAM3] Model loaded")
        except Exception as exc:
            print(f"[SAM3] Unavailable ({exc}) — using HSV fallback")
            self._model     = None
            self._processor = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def segment(self, image_bgr: np.ndarray) -> dict:
        if self._processor is not None:
            results = self._segment_sam3(image_bgr)
            if results.get("sky") is None:
                results["sky"] = self._brightness_sky_fallback(image_bgr)
            return results
        return self._segment_fallback(image_bgr)

    def _brightness_sky_fallback(self, image_bgr: np.ndarray) -> Optional[dict]:
        """Brightest-pixels sky reference for overcast / no-sky-detected scenes."""
        h, w     = image_bgr.shape[:2]
        img_area = h * w
        gray     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        top_region = gray[:int(h * 0.4), :]
        mask_full  = np.zeros((h, w), dtype=bool)
        mask_full[:int(h * 0.4), :] = (top_region >= self._brightness_bright_min)

        area = int(mask_full.sum())
        if area / img_area < self._min_area_ratio:
            threshold = np.percentile(gray, 90)
            mask_full = (gray >= threshold)
            area      = int(mask_full.sum())

        if area == 0:
            return None

        return {
            "segmentation": mask_full,
            "area":         area,
            "brightness":   float(gray[mask_full].mean()),
            "confidence":   0.5,
            "prompt_used":  "brightness_sky_fallback",
        }

    # ── SAM3 path ──────────────────────────────────────────────────────────────

    def _segment_sam3(self, image_bgr: np.ndarray) -> dict:
        import torch
        from PIL import Image as PILImage

        rgb      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil      = PILImage.fromarray(rgb)
        img_area = image_bgr.shape[0] * image_bgr.shape[1]
        results  = {}

        with torch.inference_mode():
            with torch.autocast(
                self.device, dtype=torch.bfloat16,
                enabled=(self.device == "cuda"),
            ):
                for region_key, prompts in self._prompts.items():
                    results[region_key] = self._query_best_mask(
                        pil, prompts, image_bgr, img_area,
                        is_smoke=(region_key == "smoke"),
                    )
        return results

    def _query_best_mask(
        self,
        pil_image,
        prompts,
        image_bgr,
        img_area,
        is_smoke: bool = False,
    ) -> Optional[dict]:
        for prompt in prompts:
            state = self._processor.set_image(pil_image)
            state = self._processor.set_text_prompt(prompt=prompt, state=state)
            if "masks" not in state or state["masks"] is None:
                continue
            masks_arr  = state["masks"].float().cpu().numpy()
            scores_arr = (state["scores"].float().cpu().numpy()
                          if "scores" in state else np.ones(len(masks_arr)))
            best = self._pick_best(masks_arr, scores_arr, image_bgr, img_area,
                                   is_smoke=is_smoke)
            if best is not None:
                best["prompt_used"] = prompt
                return best
        return None

    def _pick_best(
        self,
        masks,
        scores,
        image_bgr,
        img_area,
        is_smoke: bool = False,
    ) -> Optional[dict]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h    = image_bgr.shape[0]
        best = None
        best_area = 0
        for raw, score in zip(masks, scores):
            if float(score) < self._conf_threshold:
                continue
            m = raw[0] if raw.ndim == 3 else raw
            if m.max() > 1.01 or m.min() < -0.01:
                m = 1.0 / (1.0 + np.exp(-m))
            binary = (m > 0.5).astype(bool)
            area   = int(binary.sum())
            ratio  = area / img_area
            if ratio < self._min_area_ratio or ratio > self._max_area_ratio:
                continue

            # Exclude smoke masks whose vertical centre is in the top 15% (sky haze).
            if self._smoke_vert_guard and is_smoke and binary.any():
                vc = float(np.where(binary)[0].mean()) / h
                if vc < 0.15:
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

    # ── HSV fallback ───────────────────────────────────────────────────────────

    def _segment_fallback(self, image_bgr: np.ndarray) -> dict:
        h, w     = image_bgr.shape[:2]
        img_area = h * w
        hsv      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        def make(binary: np.ndarray, label: str) -> Optional[dict]:
            binary = binary.astype(bool)
            area   = int(binary.sum())
            if area / img_area < self._min_area_ratio:
                return None
            return {
                "segmentation": binary,
                "area":         area,
                "brightness":   float(gray[binary].mean()) if binary.any() else 0.0,
                "confidence":   0.5,
                "prompt_used":  label,
            }

        sky_mask           = (gray >= self._brightness_bright_min).copy()
        sky_mask[h // 2:]  = False

        smoke_mask = (
            (hsv[:, :, 1] <= self._hsv_smoke_max_sat) &
            (hsv[:, :, 2] >= self._hsv_smoke_min_val) &
            (~sky_mask) &
            (gray < self._brightness_bright_min)
        )

        dark_mask           = (gray <= self._brightness_dark_max).copy()
        dark_mask[:h // 2]  = False

        return {
            "smoke":           make(smoke_mask,  "hsv_smoke"),
            "sky":             make(sky_mask,    "hsv_sky"),
            "dark_background": make(dark_mask,   "hsv_dark_bg"),
        }
