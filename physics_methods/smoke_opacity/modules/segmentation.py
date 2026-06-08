"""
Module 1: Smoke Segmentation via SAM3.

Uses SAM3's open-vocabulary detection with the text prompt "smoke" to
produce a binary + soft mask of the smoke region.  Falls back to an
HSV-threshold heuristic when SAM3 is unavailable or finds nothing.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from ..utils.mask_ops import morphological_close, compute_boundary_band, largest_connected_component

# SAM3 package is at the project root
_SAM3_PATH = Path(__file__).parent.parent.parent.parent / "sam3"
if str(_SAM3_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM3_PATH))

_BPE_PATH = _SAM3_PATH / "assets" / "bpe_simple_vocab_16e6.txt.gz"


class SmokeSegmenter:
    """
    Segment the smoke region in an image.

    Primary method: SAM3 with the text prompt "smoke".
    Fallback:       HSV colour thresholding (low saturation, high value).
    """

    def __init__(
        self,
        device: str = "cuda",
        sam_checkpoint: str = "",
        conf_threshold: float = 0.25,
        smoke_max_saturation: int = 60,
        smoke_min_value: int = 100,
        morph_close_ksize: int = 5,
        boundary_width: int = 25,
        use_sam: bool = True,
    ):
        """
        Args:
            device:               'cuda' or 'cpu'.
            sam_checkpoint:       Local SAM3 checkpoint path (empty → HuggingFace).
            conf_threshold:       SAM3 confidence threshold.
            smoke_max_saturation: HSV S threshold for smoke colour filter (0-255).
            smoke_min_value:      HSV V threshold for smoke colour filter (0-255).
            morph_close_ksize:    Morphological closing kernel size.
            boundary_width:       Boundary band width in pixels.
            use_sam:              Set False to skip SAM3 and always use HSV fallback.
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.smoke_max_saturation = smoke_max_saturation
        self.smoke_min_value = smoke_min_value
        self.morph_close_ksize = morph_close_ksize
        self.boundary_width = boundary_width

        self._model = None
        self._processor = None

        if use_sam:
            self._load_sam3(sam_checkpoint)

    def _load_sam3(self, checkpoint: str) -> None:
        """Attempt to load SAM3; silently falls back if unavailable."""
        try:
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            if self.device == "cuda":
                import torch
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            checkpoint_path = checkpoint if checkpoint else None
            self._model = build_sam3_image_model(
                bpe_path=str(_BPE_PATH),
                device=self.device,
                eval_mode=True,
                checkpoint_path=checkpoint_path,
                load_from_HF=(not checkpoint_path),
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            self._processor = Sam3Processor(
                self._model,
                device=self.device,
                confidence_threshold=self.conf_threshold,
            )
            print("  ✓ SAM3 smoke segmenter loaded")
        except Exception as e:
            print(f"  ⚠ SAM3 not loaded ({e}); will use HSV fallback.")
            self._model = None
            self._processor = None

    def segment(
        self,
        image: np.ndarray,
        prompt: Optional[dict] = None,
    ) -> dict:
        """
        Segment the smoke region.

        Args:
            image:  (H, W, 3) BGR uint8 numpy array.
            prompt: Optional SAM3 prompt override.
                    Supported keys: "text" (str), "box" (x1,y1,x2,y2).
                    If None and SAM3 is loaded, uses text="smoke".

        Returns:
            dict:
                binary_mask      : (H, W) bool
                soft_mask        : (H, W) float32 in [0, 1]
                bbox             : (x1, y1, x2, y2) int tuple
                mask_area_ratio  : float — fraction of image covered
                method           : str "sam3" | "hsv"
                boundary_band    : (H, W) uint8
        """
        if self._processor is not None:
            result = self._segment_sam3(image, prompt)
            if result["binary_mask"].sum() > 0:
                return result
            # SAM3 returned empty mask — fall through to HSV
            print("  ⚠ SAM3 returned empty mask; using HSV fallback.")

        return self._segment_hsv(image)

    # ── SAM3 path ─────────────────────────────────────────────────────────

    def _segment_sam3(self, image: np.ndarray, prompt: Optional[dict]) -> dict:
        import torch
        from PIL import Image as PILImage

        # SAM3 expects RGB PIL image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        text = "smoke"
        box  = None
        if prompt:
            text = prompt.get("text", text)
            box  = prompt.get("box", None)

        with torch.inference_mode():
            with torch.autocast(self.device, dtype=torch.bfloat16,
                                enabled=(self.device == "cuda")):
                state = self._processor.set_image(pil_img)

                if box is not None:
                    import torch as _torch
                    box_tensor = _torch.tensor(
                        [[box[0], box[1], box[2], box[3]]],
                        dtype=_torch.float32, device=self.device
                    )
                    state = self._processor.set_box_prompt(box_tensor, state)
                else:
                    state = self._processor.set_text_prompt(
                        prompt=text, state=state
                    )

        h, w = image.shape[:2]
        binary = np.zeros((h, w), dtype=bool)
        soft   = np.zeros((h, w), dtype=np.float32)

        if "masks" in state and state["masks"] is not None:
            masks  = state["masks"].float().cpu().numpy()
            scores = state["scores"].float().cpu().numpy() if "scores" in state else np.ones(len(masks))

            # Pick the mask whose region most looks like smoke
            best_idx  = self._pick_best_smoke_mask(image, masks, scores)
            if best_idx >= 0:
                raw = masks[best_idx]
                # raw may be logits (> 1 possible) → sigmoid → [0,1]
                if raw.max() > 1.01 or raw.min() < -0.01:
                    import scipy.special
                    raw = (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
                if raw.ndim == 3:
                    raw = raw[0]
                soft   = raw.astype(np.float32)
                binary = soft > 0.5

        binary = morphological_close(binary, self.morph_close_ksize).astype(bool)
        return self._build_result(binary, soft, "sam3")

    def _pick_best_smoke_mask(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> int:
        """
        Among candidate masks, select the one most likely to be smoke.

        Heuristic: prefer high-confidence masks whose mean HSV has
        low saturation and high value (i.e. white/grey smoke).
        """
        if len(masks) == 0:
            return -1

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        best_score = -1.0
        best_idx   = 0

        for i, (mask_raw, conf) in enumerate(zip(masks, scores)):
            raw = mask_raw
            if raw.ndim == 3:
                raw = raw[0]
            # binarise
            if raw.max() > 1.01 or raw.min() < -0.01:
                raw = (1.0 / (1.0 + np.exp(-raw)))
            binary = raw > 0.5

            if binary.sum() == 0:
                continue

            mean_S = S[binary].mean()
            mean_V = V[binary].mean()

            # Smoke is low-saturation, high-value
            smoke_score = (
                float(conf)
                + (1.0 - mean_S / 255.0) * 0.5
                + (mean_V / 255.0) * 0.3
            )
            if smoke_score > best_score:
                best_score = smoke_score
                best_idx   = i

        return best_idx

    # ── HSV fallback ──────────────────────────────────────────────────────

    def _segment_hsv(self, image: np.ndarray) -> dict:
        """
        Segment smoke via HSV colour thresholding:
        low saturation + high value → probable smoke/haze.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        mask = (S <= self.smoke_max_saturation) & (V >= self.smoke_min_value)
        mask = morphological_close(mask.astype(np.uint8) * 255 > 0,
                                   self.morph_close_ksize)
        mask = largest_connected_component(mask)

        soft   = mask.astype(np.float32)
        binary = mask.astype(bool)
        return self._build_result(binary, soft, "hsv")

    # ── Shared builder ────────────────────────────────────────────────────

    def _build_result(
        self,
        binary: np.ndarray,
        soft: np.ndarray,
        method: str,
    ) -> dict:
        h, w = binary.shape
        n = binary.sum()
        area_ratio = float(n) / (h * w)

        ys, xs = np.where(binary)
        if len(xs):
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        else:
            bbox = (0, 0, 0, 0)

        boundary = compute_boundary_band(binary.astype(np.uint8), self.boundary_width)

        return {
            "binary_mask":     binary.astype(bool),
            "soft_mask":       soft.astype(np.float32),
            "bbox":            bbox,
            "mask_area_ratio": area_ratio,
            "method":          method,
            "boundary_band":   boundary,
        }
