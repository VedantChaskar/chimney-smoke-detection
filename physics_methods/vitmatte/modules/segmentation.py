"""
Module 1: Smoke Segmentation via SAM3.

Produces binary + soft probability masks for the smoke region.
Falls back to HSV thresholding when SAM3 is unavailable.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from ..utils.mask_ops import morphological_close, largest_connected_component

_SAM3_PATH = Path(__file__).parent.parent.parent.parent / "sam3"
if str(_SAM3_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM3_PATH))
_BPE_PATH = _SAM3_PATH / "assets" / "bpe_simple_vocab_16e6.txt.gz"


class SmokeSegmenter:
    """
    Segment smoke using SAM3 (text prompt "smoke") with HSV fallback.
    """

    def __init__(
        self,
        device: str = "cuda",
        sam_checkpoint: str = "",
        conf_threshold: float = 0.25,
        smoke_max_saturation: int = 60,
        smoke_min_value: int = 100,
        morph_close_ksize: int = 5,
        use_sam: bool = True,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.smoke_max_saturation = smoke_max_saturation
        self.smoke_min_value = smoke_min_value
        self.morph_close_ksize = morph_close_ksize
        self._model = None
        self._processor = None

        if use_sam:
            self._load_sam3(sam_checkpoint)

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
            print(f"  ⚠ SAM3 unavailable ({e}); using HSV fallback.")
            self._model = None
            self._processor = None

    def segment(self, image: np.ndarray, prompt: Optional[dict] = None) -> dict:
        """
        Args:
            image:  (H, W, 3) BGR uint8.
            prompt: optional dict with keys "text" (str) or "box" (x1,y1,x2,y2).

        Returns:
            binary_mask      : (H, W) bool
            soft_mask        : (H, W) float32 [0,1]  — SAM3 logits→sigmoid
            bbox             : (x1, y1, x2, y2)
            mask_area_ratio  : float
            confidence       : float
            method           : "sam3" | "hsv"
        """
        if self._processor is not None:
            result = self._segment_sam3(image, prompt)
            if result["binary_mask"].sum() > 0:
                return result
            print("  ⚠ SAM3 returned empty mask; using HSV fallback.")
        return self._segment_hsv(image)

    # ── SAM3 ─────────────────────────────────────────────────────────────

    def _segment_sam3(self, image: np.ndarray, prompt: Optional[dict]) -> dict:
        import torch
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        text = (prompt or {}).get("text", "smoke")
        box  = (prompt or {}).get("box", None)

        with torch.inference_mode():
            with torch.autocast(self.device, dtype=torch.bfloat16,
                                enabled=(self.device == "cuda")):
                state = self._processor.set_image(pil_img)
                if box is not None:
                    bt = torch.tensor([[*box]], dtype=torch.float32,
                                      device=self.device)
                    state = self._processor.set_box_prompt(bt, state)
                else:
                    state = self._processor.set_text_prompt(
                        prompt=text, state=state)

        h, w = image.shape[:2]
        binary = np.zeros((h, w), dtype=bool)
        soft   = np.zeros((h, w), dtype=np.float32)
        conf   = 0.0

        if "masks" in state and state["masks"] is not None:
            masks  = state["masks"].float().cpu().numpy()
            scores = (state["scores"].float().cpu().numpy()
                      if "scores" in state else np.ones(len(masks)))
            idx = self._pick_best_smoke_mask(image, masks, scores)
            if idx >= 0:
                raw = masks[idx]
                if raw.ndim == 3:
                    raw = raw[0]
                if raw.max() > 1.01 or raw.min() < -0.01:
                    raw = 1.0 / (1.0 + np.exp(-raw))
                soft   = raw.astype(np.float32)
                binary = soft > 0.5
                conf   = float(scores[idx])

        binary = morphological_close(binary, self.morph_close_ksize).astype(bool)
        return self._build_result(binary, soft, conf, "sam3")

    def _pick_best_smoke_mask(self, image, masks, scores):
        if len(masks) == 0:
            return -1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        best_score, best_idx = -1.0, 0
        for i, (mask_raw, conf) in enumerate(zip(masks, scores)):
            raw = mask_raw[0] if mask_raw.ndim == 3 else mask_raw
            if raw.max() > 1.01 or raw.min() < -0.01:
                raw = 1.0 / (1.0 + np.exp(-raw))
            binary = raw > 0.5
            if not binary.any():
                continue
            s = (float(conf)
                 + (1.0 - S[binary].mean() / 255.0) * 0.5
                 + (V[binary].mean() / 255.0) * 0.3)
            if s > best_score:
                best_score, best_idx = s, i
        return best_idx

    # ── HSV fallback ──────────────────────────────────────────────────────

    def _segment_hsv(self, image: np.ndarray) -> dict:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] <= self.smoke_max_saturation) &
                (hsv[:, :, 2] >= self.smoke_min_value))
        mask = morphological_close(mask.astype(np.uint8), self.morph_close_ksize)
        mask = largest_connected_component(mask)
        soft = mask.astype(np.float32)
        return self._build_result(mask.astype(bool), soft, 0.5, "hsv")

    def _build_result(self, binary, soft, conf, method):
        h, w = binary.shape
        ys, xs = np.where(binary)
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())) if len(xs) else (0, 0, 0, 0)
        return {
            "binary_mask":     binary.astype(bool),
            "soft_mask":       soft.astype(np.float32),
            "bbox":            bbox,
            "mask_area_ratio": float(binary.sum()) / (h * w),
            "confidence":      conf,
            "method":          method,
        }
