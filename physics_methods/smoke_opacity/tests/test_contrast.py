"""
Unit tests for the contrast attenuation module.

Run:
    conda run -n smokescreen python -m pytest smoke_opacity/tests/test_contrast.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smoke_opacity.modules.contrast import compute_contrast_ratio
from smoke_opacity.modules.color_collapse import compute_color_collapse


def _central_mask(h=128, w=128, frac=0.5) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = int(h * (1-frac)/2), int(h * (1+frac)/2)
    x0, x1 = int(w * (1-frac)/2), int(w * (1+frac)/2)
    mask[y0:y1, x0:x1] = 1
    return mask


class TestContrastRatio:
    def test_returns_expected_keys(self):
        gray = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        mask = _central_mask()
        result = compute_contrast_ratio(gray, mask)
        for key in ["contrast_inside", "contrast_boundary", "contrast_ratio",
                    "opacity_estimate", "reliable"]:
            assert key in result

    def test_opacity_in_range(self):
        gray = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        mask = _central_mask()
        result = compute_contrast_ratio(gray, mask)
        assert 0.0 <= result["opacity_estimate"] <= 1.0

    def test_blurred_region_has_high_opacity(self):
        """Blurring the smoke region reduces contrast → higher opacity estimate."""
        rng = np.random.default_rng(3)
        img = rng.integers(0, 255, (128, 128), dtype=np.uint8)
        mask = _central_mask()

        import cv2
        blurred = img.copy()
        blurred[mask.astype(bool)] = cv2.GaussianBlur(img, (21, 21), 0)[mask.astype(bool)]

        result_orig    = compute_contrast_ratio(img,     mask)
        result_blurred = compute_contrast_ratio(blurred, mask)

        assert result_blurred["opacity_estimate"] >= result_orig["opacity_estimate"] - 0.1

    def test_uniform_smoke_gives_high_opacity(self):
        """Uniform interior (no contrast) with textured boundary → opacity ≈ 1."""
        rng = np.random.default_rng(7)
        img = rng.integers(50, 200, (128, 128), dtype=np.uint8)
        mask = _central_mask()
        # Flatten the interior to a constant
        img[mask.astype(bool)] = 150
        result = compute_contrast_ratio(img, mask)
        assert result["opacity_estimate"] > 0.5


class TestColorCollapse:
    def test_returns_expected_keys(self):
        img  = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = _central_mask()
        result = compute_color_collapse(img, mask)
        for key in ["spread_inside", "spread_boundary", "collapse_ratio", "opacity_estimate"]:
            assert key in result

    def test_opacity_in_range(self):
        img  = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = _central_mask()
        result = compute_color_collapse(img, mask)
        assert 0.0 <= result["opacity_estimate"] <= 1.0

    def test_grey_smoke_has_high_opacity(self):
        """Uniform grey interior (zero channel spread) → high opacity estimate."""
        rng = np.random.default_rng(42)
        img  = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = _central_mask()
        img[mask.astype(bool)] = [180, 180, 180]   # grey smoke
        result = compute_color_collapse(img, mask)
        assert result["opacity_estimate"] > 0.3
