"""
src/pipeline.py

End-to-end SmokeScreen inference pipeline.

Steps:
  1. Segment image into smoke / sky / dark-background regions (SAM3 or HSV fallback)
  2. Classify regions as smoke_dark / smoke_light / background_bright / background_dark
  3. Compute opacity % using the brightness-ratio formula
  4. Map opacity → Ringelmann grade (0–5)
  5. Return a structured result dict
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.segmentation.sam3_segmenter import SAM3Segmenter
from src.segmentation.region_classifier import classify_regions, summarise_regions
from src.opacity.formula import compute_opacity
from src.opacity.ringelman import opacity_to_ringelman, ringelman_description


class SmokeScreenPipeline:
    """
    Full inference pipeline: image → Ringelmann grade.

    Parameters
    ----------
    segmenter : SAM3Segmenter or None
        If None, one is created automatically using config defaults.
    alpha : float
        Calibration constant for Formula 1.  Tune with alpha_tuner.py.
    bands : list or None
        Ringelmann band boundaries (defaults to cfg.RINGELMAN_BANDS).
    """

    def __init__(
        self,
        segmenter: SAM3Segmenter = None,
        alpha: float = cfg.ALPHA_DEFAULT,
        bands: list = None,
    ):
        if segmenter is None:
            segmenter = SAM3Segmenter()
        self.segmenter = segmenter
        self.alpha     = alpha
        self.bands     = bands or cfg.RINGELMAN_BANDS

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, image_bgr: np.ndarray) -> dict:
        """
        Run the full pipeline on one image.

        Args:
            image_bgr: (H, W, 3) BGR numpy array.

        Returns:
            result dict with keys:
                grade          : int 0–5
                opacity        : float 0–100
                formula        : str
                smoke_type     : str
                reliable       : bool
                L_smoke        : float or None
                L_bright       : float or None
                L_dark         : float or None
                regions_summary: dict (per-region stats)
                elapsed_s      : float
        """
        t0 = time.time()

        # Step 1: Segment
        raw_regions = self.segmenter.segment_all_regions(image_bgr)

        # Step 2: Classify
        regions = classify_regions(image_bgr, raw_regions)

        # Step 3: Opacity
        opacity_result = compute_opacity(image_bgr, regions, alpha=self.alpha)

        # Step 4: Ringelmann
        grade = opacity_to_ringelman(opacity_result["opacity"], bands=self.bands)
        desc  = ringelman_description(grade)

        # Step 5: Assemble
        result = {
            **desc,
            **opacity_result,
            "regions_summary": summarise_regions(regions),
            "elapsed_s":       round(time.time() - t0, 3),
        }
        return result

    def run_batch(self, image_paths: list) -> list:
        """
        Run the pipeline on multiple images.

        Args:
            image_paths: list of str or Path.

        Returns:
            list of dicts — one per image, with an added "image_path" key.
            If an image fails to load or process, the error is recorded.
        """
        results = []
        for path in image_paths:
            path = str(path)
            img  = cv2.imread(path)
            if img is None:
                results.append({"image_path": path, "error": "Could not load image"})
                continue
            try:
                r = self.run(img)
                r["image_path"] = path
                results.append(r)
            except Exception as e:
                results.append({"image_path": path, "error": str(e)})
        return results


# ── Convenience factory ────────────────────────────────────────────────────────

def create_pipeline(
    use_sam: bool = True,
    alpha: float = cfg.ALPHA_DEFAULT,
    device: str = "cuda",
) -> SmokeScreenPipeline:
    """
    Convenience factory that handles segmenter creation.

    Args:
        use_sam: If False, force HSV fallback (no GPU required).
        alpha:   Calibration constant.
        device:  "cuda" or "cpu".
    """
    from src.segmentation.sam3_segmenter import load_segmenter
    seg = load_segmenter(
        checkpoint=cfg.SAM3_CHECKPOINT,
        device=device,
        use_sam=use_sam,
    )
    return SmokeScreenPipeline(segmenter=seg, alpha=alpha)
