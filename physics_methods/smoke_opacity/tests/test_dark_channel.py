"""
Unit tests for the dark channel prior module.

Run:
    conda run -n smokescreen python -m pytest smoke_opacity/tests/test_dark_channel.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smoke_opacity.modules.dark_channel import (
    dark_channel,
    guided_filter,
    estimate_transmittance,
    transmittance_to_alpha,
)


class TestDarkChannel:
    def test_output_shape(self):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        dc = dark_channel(img, patch_size=5)
        assert dc.shape == (64, 64)

    def test_uniform_image(self):
        """Dark channel of a uniform image equals that value."""
        val = 0.5
        img = np.full((32, 32, 3), val, dtype=np.float32)
        dc = dark_channel(img, patch_size=3)
        np.testing.assert_allclose(dc, val, atol=1e-5)

    def test_dark_channel_is_min(self):
        """Dark channel ≤ minimum channel value everywhere."""
        rng = np.random.default_rng(0)
        img = rng.random((64, 64, 3)).astype(np.float32)
        dc = dark_channel(img, patch_size=7)
        channel_min = img.min(axis=2)
        assert np.all(dc <= channel_min + 1e-5)

    def test_zero_image(self):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        dc = dark_channel(img, patch_size=5)
        np.testing.assert_allclose(dc, 0.0, atol=1e-6)


class TestGuidedFilter:
    def test_output_shape(self):
        guide = np.random.rand(64, 64).astype(np.float32)
        src   = np.random.rand(64, 64).astype(np.float32)
        out   = guided_filter(guide, src, radius=8, eps=1e-3)
        assert out.shape == (64, 64)

    def test_preserves_range(self):
        """Output should stay roughly in [0, 1] for [0,1] inputs."""
        guide = np.random.rand(128, 128).astype(np.float32)
        src   = np.clip(np.random.rand(128, 128).astype(np.float32), 0, 1)
        out   = guided_filter(guide, src, radius=20, eps=1e-3)
        assert out.min() >= -0.1
        assert out.max() <= 1.1

    def test_constant_signal_passes_through(self):
        """Constant input → constant output (filtered output ≈ input)."""
        guide = np.random.rand(64, 64).astype(np.float32)
        src   = np.full((64, 64), 0.7, dtype=np.float32)
        out   = guided_filter(guide, src, radius=10, eps=1e-3)
        np.testing.assert_allclose(out, 0.7, atol=0.05)


class TestTransmittance:
    def _make_test_inputs(self):
        rng = np.random.default_rng(1)
        image = (rng.integers(50, 200, (64, 64, 3))).astype(np.uint8)
        A = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        return image, A, mask

    def test_transmittance_range(self):
        img, A, mask = self._make_test_inputs()
        t = estimate_transmittance(img, A, mask)
        assert t.min() >= 0.0 - 1e-5
        assert t.max() <= 1.0 + 1e-5

    def test_outside_mask_is_one(self):
        img, A, mask = self._make_test_inputs()
        t = estimate_transmittance(img, A, mask, t_min=0.1)
        outside = mask == 0
        np.testing.assert_allclose(t[outside], 1.0, atol=1e-5)

    def test_t_min_respected(self):
        img, A, mask = self._make_test_inputs()
        t_min = 0.15
        t = estimate_transmittance(img, A, mask, t_min=t_min)
        assert t[mask.astype(bool)].min() >= t_min - 1e-5

    def test_transmittance_to_alpha(self):
        t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        alpha = transmittance_to_alpha(t)
        np.testing.assert_allclose(alpha, [1.0, 0.5, 0.0], atol=1e-6)
