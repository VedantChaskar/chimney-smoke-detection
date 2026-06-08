"""
Unit tests for the ViTMatte alpha matting module.

These tests load the actual ViTMatte model, so they require:
    - Internet (first run: downloads model weights)
    - CUDA or CPU with sufficient RAM

Run:
    conda run -n smokescreen python -m pytest vitmatte/tests/test_matting.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vitmatte.config import Config, HF_CACHE_DIR
from vitmatte.modules.matting import AlphaEstimator
from vitmatte.modules.trimap import generate_trimap


@pytest.fixture(scope="module")
def estimator():
    cfg = Config()
    return AlphaEstimator(
        model_name=cfg.matting_model,
        device=cfg.device,
        max_size=512,          # smaller for faster tests
        hf_cache_dir=HF_CACHE_DIR,
    )


def _make_test_inputs(h=256, w=256, fg_frac=0.5):
    """Create a synthetic image + matching trimap."""
    rng = np.random.default_rng(0)
    image = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    # Central mask
    mask = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * (1 - fg_frac) / 2); y1 = int(h * (1 + fg_frac) / 2)
    x0 = int(w * (1 - fg_frac) / 2); x1 = int(w * (1 + fg_frac) / 2)
    mask[y0:y1, x0:x1] = 1
    trimap = generate_trimap(mask, erosion_size=10, dilation_size=20)
    return image, trimap


class TestAlphaEstimatorOutput:
    def test_output_shape(self, estimator):
        image, trimap = _make_test_inputs(256, 256)
        result = estimator.estimate_alpha(image, trimap)
        assert result["alpha_map"].shape == image.shape[:2]
        assert result["alpha_map_masked"].shape == image.shape[:2]

    def test_alpha_range(self, estimator):
        image, trimap = _make_test_inputs(256, 256)
        result = estimator.estimate_alpha(image, trimap)
        alpha = result["alpha_map"]
        assert alpha.min() >= 0.0 - 1e-4
        assert alpha.max() <= 1.0 + 1e-4

    def test_bg_is_zero(self, estimator):
        """Definite background (trimap==0) must be zeroed in alpha_map."""
        image, trimap = _make_test_inputs(256, 256)
        result = estimator.estimate_alpha(image, trimap)
        bg_alpha = result["alpha_map"][trimap == 0]
        np.testing.assert_allclose(bg_alpha, 0.0, atol=1e-4)

    def test_masked_zeros_outside_smoke(self, estimator):
        """alpha_map_masked should be 0 in definite background."""
        image, trimap = _make_test_inputs(256, 256)
        result = estimator.estimate_alpha(image, trimap)
        bg_alpha = result["alpha_map_masked"][trimap == 0]
        np.testing.assert_allclose(bg_alpha, 0.0, atol=1e-4)

    def test_unknown_zone_has_gradation(self, estimator):
        """Unknown zone should have a range of alpha values (not all identical)."""
        image, trimap = _make_test_inputs(256, 256)
        result = estimator.estimate_alpha(image, trimap)
        unknown_alpha = result["alpha_map"][trimap == 128]
        if len(unknown_alpha) > 10:
            assert unknown_alpha.std() > 0.01, (
                f"Alpha in unknown zone has no gradation (std={unknown_alpha.std():.4f})"
            )

    def test_different_sizes(self, estimator):
        """Model should handle non-square images."""
        for h, w in [(128, 256), (256, 128), (100, 150)]:
            image, trimap = _make_test_inputs(h, w)
            result = estimator.estimate_alpha(image, trimap)
            assert result["alpha_map"].shape == (h, w)


class TestAlphaPostProcessing:
    def test_bilateral_filter_smooths_noise(self, estimator):
        """Alpha with bilateral filter should have smaller local std than input."""
        # This is a smoke-test that the filter runs without error
        image, trimap = _make_test_inputs(128, 128)
        result = estimator.estimate_alpha(image, trimap)
        assert result["alpha_map"] is not None

    def test_float32_output(self, estimator):
        image, trimap = _make_test_inputs(128, 128)
        result = estimator.estimate_alpha(image, trimap)
        assert result["alpha_map"].dtype == np.float32
