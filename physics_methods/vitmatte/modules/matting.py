"""
Module 3: Deep Alpha Matting via ViTMatte.

ViTMatte (Yao et al., ICCV 2023) — Vision Transformer-based alpha matting.
HuggingFace: hustvl/vitmatte-small-composition-1k
             hustvl/vitmatte-base-composition-1k

Input:  RGB image (H, W, 3)  +  trimap (H, W) with values {0, 128, 255}
Output: alpha map (H, W) float32 [0, 1]
"""

from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch


class AlphaEstimator:
    """
    Wraps ViTMatte for per-pixel smoke opacity estimation.
    """

    def __init__(
        self,
        model_name: str = "hustvl/vitmatte-small-composition-1k",
        device: str = "cuda",
        max_size: int = 1024,
        bilateral_d: int = 5,
        bilateral_sigma: float = 0.1,
        hf_cache_dir: Optional[str | Path] = None,
    ):
        """
        Args:
            model_name:      HuggingFace model ID (or local path).
            device:          'cuda' or 'cpu'.
            max_size:        Longest edge input size (image is resized to this).
            bilateral_d:     Bilateral filter diameter for post-processing.
            bilateral_sigma: Bilateral filter sigma (colour space).
            hf_cache_dir:    Optional local HuggingFace cache directory.
        """
        self.model_name     = model_name
        self.device         = device
        self.max_size       = max_size
        self.bilateral_d    = bilateral_d
        self.bilateral_sigma = bilateral_sigma
        self._model          = None
        self._processor      = None

        cache = str(hf_cache_dir) if hf_cache_dir else None
        self._load(cache)

    # ── Loading ────────────────────────────────────────────────────────────

    def _load(self, cache_dir: Optional[str]) -> None:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        print(f"  Loading ViTMatte: {self.model_name} …")
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        self._processor = VitMatteImageProcessor.from_pretrained(
            self.model_name, **kwargs
        )
        self._model = VitMatteForImageMatting.from_pretrained(
            self.model_name, **kwargs
        )
        self._model.eval()
        self._model.to(self.device)
        print(f"  ✓ ViTMatte loaded on {self.device}")

    # ── Public API ─────────────────────────────────────────────────────────

    def estimate_alpha(
        self,
        image: np.ndarray,
        trimap: np.ndarray,
    ) -> dict:
        """
        Estimate per-pixel alpha (smoke opacity).

        Args:
            image:  (H, W, 3) BGR uint8 or float32.
            trimap: (H, W) uint8 with values {0, 128, 255}.

        Returns:
            alpha_map        : (H, W) float32 [0,1] — per-pixel opacity.
            alpha_map_masked : (H, W) float32 — alpha zeroed outside smoke region.
            detail_map       : None (not exposed by current HF implementation).
        """
        orig_h, orig_w = image.shape[:2]

        # ── Prepare inputs ────────────────────────────────────────────────
        img_rgb, trimap_rs, scale = self._prepare_inputs(image, trimap)

        # ── Forward pass ──────────────────────────────────────────────────
        alpha_rs = self._forward(img_rgb, trimap_rs)

        # ── Resize back to original resolution ───────────────────────────
        alpha = cv2.resize(
            alpha_rs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # ── Post-processing ───────────────────────────────────────────────
        alpha = self._postprocess(alpha, trimap)

        # Masked version: zero outside dilated smoke region
        smoke_region = trimap > 0   # FG + unknown
        alpha_masked = alpha.copy()
        alpha_masked[~smoke_region] = 0.0

        return {
            "alpha_map":        alpha,
            "alpha_map_masked": alpha_masked,
            "detail_map":       None,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _prepare_inputs(
        self,
        image: np.ndarray,
        trimap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Resize image and trimap so the longest edge = max_size.
        Returns (rgb_resized, trimap_resized, scale).
        """
        h, w = image.shape[:2]
        scale = self.max_size / max(h, w)
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_res = cv2.resize(image,  (new_w, new_h), cv2.INTER_LINEAR)
            tri_res = cv2.resize(trimap, (new_w, new_h), cv2.INTER_NEAREST)
        else:
            img_res, tri_res = image.copy(), trimap.copy()
            scale = 1.0

        # BGR → RGB
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        return img_rgb, tri_res, scale

    def _forward(self, img_rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """
        Run ViTMatte forward pass.

        The HuggingFace VitMatteImageProcessor expects:
            images:  PIL Image or numpy RGB (H, W, 3)
            trimaps: numpy (H, W) with values {0, 128, 255}

        Returns alpha_map (H, W) float32 at the resized resolution.
        """
        from PIL import Image as PILImage

        pil_image  = PILImage.fromarray(img_rgb)
        # Processor expects trimap as PIL 'L' image
        pil_trimap = PILImage.fromarray(trimap, mode="L")

        inputs = self._processor(
            images=pil_image,
            trimaps=pil_trimap,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # outputs.alphas: (1, 1, H, W) float32
        alpha = outputs.alphas[0, 0].cpu().float().numpy()
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def _postprocess(self, alpha: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """
        Post-process the alpha map:
            a. Clamp to [0, 1].
            b. Zero out definite background.
            c. Smoke core (definite FG): trust model prediction — do NOT force 1.0,
               because smoke may be semi-transparent even at its densest.
            d. Bilateral filter to smooth noise while preserving edges.
        """
        alpha = np.clip(alpha, 0.0, 1.0)

        # Definite background → always 0
        bg_mask = (trimap == 0)
        alpha[bg_mask] = 0.0

        # Bilateral filter (operates on [0,255] range)
        if self.bilateral_d > 0:
            alpha_u8  = (alpha * 255).astype(np.uint8)
            alpha_u8  = cv2.bilateralFilter(
                alpha_u8,
                d=self.bilateral_d,
                sigmaColor=int(self.bilateral_sigma * 255),
                sigmaSpace=self.bilateral_d * 2,
            )
            alpha = alpha_u8.astype(np.float32) / 255.0

        alpha[bg_mask] = 0.0   # re-apply after filter (filter may bleed)
        return alpha.astype(np.float32)
