"""
Unit tests for Ringelmann mapping module.

Run:
    conda run -n smokescreen python -m pytest vitmatte/tests/test_ringelmann.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vitmatte.modules.ringelmann import (
    alpha_to_ringelmann, alpha_to_ringelmann_float,
    compute_ringelmann, RINGELMANN_BOUNDARIES
)


def _central_mask(h=128, w=128, frac=0.5):
    mask = np.zeros((h, w), dtype=bool)
    y0 = int(h * (1-frac)/2); y1 = int(h*(1+frac)/2)
    x0 = int(w * (1-frac)/2); x1 = int(w*(1+frac)/2)
    mask[y0:y1, x0:x1] = True
    return mask


class TestAlphaToRingelmann:
    @pytest.mark.parametrize("alpha,expected", [
        (0.00, 0), (0.05, 0), (0.09, 0),
        (0.10, 1), (0.20, 1), (0.29, 1),
        (0.30, 2), (0.40, 2), (0.49, 2),
        (0.50, 3), (0.60, 3), (0.69, 3),
        (0.70, 4), (0.80, 4), (0.89, 4),
        (0.90, 5), (1.00, 5),
    ])
    def test_boundaries(self, alpha, expected):
        assert alpha_to_ringelmann(alpha) == expected

    def test_monotone(self):
        alphas = np.linspace(0, 1, 500)
        rings  = [alpha_to_ringelmann(a) for a in alphas]
        for i in range(len(rings) - 1):
            assert rings[i] <= rings[i + 1]

    def test_float_scale(self):
        assert alpha_to_ringelmann_float(0.0)  == 0.0
        assert alpha_to_ringelmann_float(0.5)  == 2.5
        assert alpha_to_ringelmann_float(1.0)  == 5.0
        assert alpha_to_ringelmann_float(0.4)  == 2.0


class TestComputeRingelmann:
    def _uniform_alpha(self, value, h=64, w=64):
        alpha = np.full((h, w), value, dtype=np.float32)
        mask  = _central_mask(h, w, frac=0.5)
        return alpha, mask

    def test_returns_expected_keys(self):
        alpha, mask = self._uniform_alpha(0.5)
        result = compute_ringelmann(alpha, mask)
        for key in ["ringelmann", "ringelmann_float", "representative_alpha",
                    "method_used", "spatial_stats"]:
            assert key in result

    def test_empty_mask_returns_r0(self):
        alpha = np.full((64, 64), 0.8, dtype=np.float32)
        mask  = np.zeros((64, 64), dtype=bool)
        result = compute_ringelmann(alpha, mask)
        assert result["ringelmann"] == 0

    @pytest.mark.parametrize("method", ["mean", "p75", "density_weighted", "peak_region"])
    def test_all_methods_work(self, method):
        alpha, mask = self._uniform_alpha(0.5)
        result = compute_ringelmann(alpha, mask, method=method)
        assert 0 <= result["ringelmann"] <= 5

    def test_density_weighted_emphasises_core(self):
        """density_weighted should give higher result than mean for skewed distributions."""
        rng = np.random.default_rng(42)
        alpha = np.zeros((64, 64), dtype=np.float32)
        mask  = _central_mask(64, 64)
        # Mostly low alpha with a bright core
        alpha[mask] = rng.uniform(0.0, 0.2, mask.sum()).astype(np.float32)
        # Dense core patch
        alpha[28:36, 28:36] = 0.9

        r_mean = compute_ringelmann(alpha, mask, method="mean")["representative_alpha"]
        r_dw   = compute_ringelmann(alpha, mask, method="density_weighted")["representative_alpha"]
        assert r_dw >= r_mean

    def test_uniform_alpha_gives_correct_ringelmann(self):
        """Uniform alpha=0.6 → R3 regardless of method."""
        alpha, mask = self._uniform_alpha(0.6)
        for method in ["mean", "p75", "density_weighted"]:
            r = compute_ringelmann(alpha, mask, method=method)
            assert r["ringelmann"] == 3, f"method={method}: got R{r['ringelmann']}"

    def test_spatial_stats_populated(self):
        alpha, mask = self._uniform_alpha(0.4)
        result = compute_ringelmann(alpha, mask)
        stats = result["spatial_stats"]
        for key in ["mean_alpha", "median_alpha", "p75_alpha", "p90_alpha",
                    "std_alpha", "coverage_fraction"]:
            assert key in stats
            assert isinstance(stats[key], float)

    def test_rep_alpha_in_range(self):
        rng = np.random.default_rng(7)
        alpha = rng.random((128, 128)).astype(np.float32)
        mask  = _central_mask(128, 128)
        for method in ["mean", "p75", "density_weighted", "peak_region"]:
            r = compute_ringelmann(alpha, mask, method=method)
            assert 0.0 <= r["representative_alpha"] <= 1.0
