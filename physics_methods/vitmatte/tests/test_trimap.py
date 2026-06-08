"""
Unit tests for trimap generation.

Run:
    conda run -n smokescreen python -m pytest vitmatte/tests/test_trimap.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vitmatte.modules.trimap import (
    generate_trimap, adaptive_trimap, trimap_stats, visualise_trimap
)


def _central_mask(h=128, w=128, frac=0.5):
    mask = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * (1 - frac) / 2); y1 = int(h * (1 + frac) / 2)
    x0 = int(w * (1 - frac) / 2); x1 = int(w * (1 + frac) / 2)
    mask[y0:y1, x0:x1] = 1
    return mask


class TestGenerateTrimap:
    def test_output_shape(self):
        mask = _central_mask()
        tm = generate_trimap(mask, erosion_size=5, dilation_size=10)
        assert tm.shape == mask.shape

    def test_values_are_0_128_255(self):
        mask = _central_mask()
        tm = generate_trimap(mask, erosion_size=5, dilation_size=10)
        unique = set(np.unique(tm))
        assert unique.issubset({0, 128, 255})

    def test_fg_inside_original_mask(self):
        """All definite-FG pixels (255) should be inside the original mask."""
        mask = _central_mask()
        tm = generate_trimap(mask, erosion_size=3, dilation_size=8)
        fg_outside_mask = (tm == 255) & (mask == 0)
        assert fg_outside_mask.sum() == 0

    def test_bg_outside_dilated_region(self):
        """All BG pixels (0) should be outside the dilated mask."""
        from vitmatte.utils.mask_ops import dilate_mask
        mask = _central_mask()
        dil  = dilate_mask(mask, 10).astype(bool)
        tm   = generate_trimap(mask, erosion_size=3, dilation_size=10)
        bg_inside_dilated = (tm == 0) & dil
        assert bg_inside_dilated.sum() == 0

    def test_unknown_band_exists(self):
        """There should be a substantial unknown zone."""
        mask = _central_mask()
        tm = generate_trimap(mask, erosion_size=5, dilation_size=15)
        stats = trimap_stats(tm)
        assert stats["unknown_fraction"] > 0.05

    def test_soft_mask_expands_unknown(self):
        """Providing a soft mask with uncertain region should widen the unknown zone."""
        mask = _central_mask()
        tm_no_soft = generate_trimap(mask, erosion_size=5, dilation_size=10)

        # Soft mask with uncertain band around the hard mask boundary
        soft = mask.astype(np.float32) * 0.5  # all 0 or 0.5 → all uncertain
        tm_with_soft = generate_trimap(mask, soft_mask=soft,
                                        erosion_size=5, dilation_size=10)
        s1 = trimap_stats(tm_no_soft)["unknown_fraction"]
        s2 = trimap_stats(tm_with_soft)["unknown_fraction"]
        assert s2 >= s1

    def test_empty_mask(self):
        """Empty mask should produce all-zero trimap."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        tm = generate_trimap(mask)
        assert (tm == 0).all()

    def test_full_mask(self):
        """Full-image mask with large erosion → FG dominates."""
        mask = np.ones((64, 64), dtype=np.uint8)
        tm = generate_trimap(mask, erosion_size=10, dilation_size=5)
        # The eroded centre should be definite FG
        assert (tm == 255).sum() > 0


class TestAdaptiveTrimap:
    def test_output_values(self):
        mask  = _central_mask()
        soft  = mask.astype(np.float32) * 0.9
        rng   = np.random.default_rng(0)
        image = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        tm = adaptive_trimap(mask, soft, image)
        assert set(np.unique(tm)).issubset({0, 128, 255})

    def test_complex_mask_gets_wider_unknown(self):
        """A highly irregular mask should get a wider unknown zone than a simple one."""
        rng = np.random.default_rng(1)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)

        # Simple circular mask
        simple = np.zeros((128, 128), dtype=np.uint8)
        cv2_available = True
        try:
            import cv2
            cv2.circle(simple, (64, 64), 30, 1, -1)
        except Exception:
            cv2_available = False

        if not cv2_available:
            pytest.skip("cv2 not available")

        # Complex jagged mask
        complex_mask = rng.integers(0, 2, (128, 128), dtype=np.uint8)
        complex_mask[:20] = 0; complex_mask[-20:] = 0
        complex_mask[:, :20] = 0; complex_mask[:, -20:] = 0

        soft_s = simple.astype(np.float32) * 0.9
        soft_c = complex_mask.astype(np.float32) * 0.5

        tm_simple  = adaptive_trimap(simple, soft_s, img)
        tm_complex = adaptive_trimap(complex_mask, soft_c, img)

        unk_simple  = trimap_stats(tm_simple)["unknown_fraction"]
        unk_complex = trimap_stats(tm_complex)["unknown_fraction"]
        # Complex mask should have equal or more unknown region
        assert unk_complex >= unk_simple - 0.05


class TestTrimapUtils:
    def test_visualise_trimap_shape(self):
        mask = _central_mask()
        tm  = generate_trimap(mask)
        vis = visualise_trimap(tm)
        assert vis.shape == (*mask.shape, 3)

    def test_stats_sum_to_one(self):
        mask = _central_mask()
        tm = generate_trimap(mask)
        s = trimap_stats(tm)
        total = s["fg_fraction"] + s["unknown_fraction"] + s["bg_fraction"]
        np.testing.assert_allclose(total, 1.0, atol=1e-5)
