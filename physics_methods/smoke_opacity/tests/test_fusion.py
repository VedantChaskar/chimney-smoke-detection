"""
Unit tests for the fusion and Ringelmann mapping module.

Run:
    conda run -n smokescreen python -m pytest smoke_opacity/tests/test_fusion.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smoke_opacity.modules.fusion import (
    fuse_opacity_estimates,
    alpha_to_ringelmann,
    RINGELMANN_THRESHOLDS,
    DEFAULT_WEIGHTS,
)


class TestAlphaToRingelmann:
    @pytest.mark.parametrize("alpha,expected", [
        (0.00, 0),
        (0.05, 0),
        (0.10, 1),
        (0.24, 1),
        (0.25, 2),
        (0.44, 2),
        (0.45, 3),
        (0.64, 3),
        (0.65, 4),
        (0.79, 4),
        (0.80, 5),
        (1.00, 5),
    ])
    def test_mapping(self, alpha, expected):
        assert alpha_to_ringelmann(alpha) == expected

    def test_monotone(self):
        """R number should be non-decreasing with alpha."""
        alphas = np.linspace(0, 1, 200)
        rings  = [alpha_to_ringelmann(a) for a in alphas]
        for i in range(len(rings) - 1):
            assert rings[i] <= rings[i + 1]


class TestFuseOpacityEstimates:
    def test_basic_output_keys(self):
        result = fuse_opacity_estimates(0.5, 0.5, 0.5)
        for key in ["fused_alpha", "ringelmann", "confidence", "per_method", "weights_used"]:
            assert key in result

    def test_fused_alpha_in_range(self):
        result = fuse_opacity_estimates(0.3, 0.4, 0.2)
        assert 0.0 <= result["fused_alpha"] <= 1.0

    def test_equal_inputs_give_same_fused(self):
        """When all inputs are equal, fused = that value."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = fuse_opacity_estimates(alpha, alpha, alpha)
            np.testing.assert_allclose(result["fused_alpha"], alpha, atol=1e-5)

    def test_unreliable_contrast_redistributes_weight(self):
        """Unreliable contrast should not reduce dark_channel weight."""
        r_reliable   = fuse_opacity_estimates(0.6, 0.1, 0.6,
                                               confidence_flags={"contrast_reliable": True})
        r_unreliable = fuse_opacity_estimates(0.6, 0.1, 0.6,
                                               confidence_flags={"contrast_reliable": False})
        # Unreliable contrast shifts weight to dark_channel (=0.6), making fused higher
        assert r_unreliable["fused_alpha"] >= r_reliable["fused_alpha"] - 0.01

    def test_low_confidence_when_methods_disagree(self):
        """Widely disagreeing methods → low confidence."""
        result = fuse_opacity_estimates(0.0, 1.0, 0.5)
        assert result["confidence"] == "low"

    def test_high_confidence_when_methods_agree(self):
        """Methods close together → high confidence."""
        result = fuse_opacity_estimates(0.5, 0.5, 0.5)
        assert result["confidence"] == "high"

    def test_ringelmann_matches_fused_alpha(self):
        """Ringelmann should match what alpha_to_ringelmann returns."""
        for a_dcp, a_con, a_col in [(0.1, 0.1, 0.1), (0.5, 0.5, 0.5), (0.85, 0.85, 0.85)]:
            result = fuse_opacity_estimates(a_dcp, a_con, a_col)
            expected_ring = alpha_to_ringelmann(result["fused_alpha"])
            assert result["ringelmann"] == expected_ring

    def test_custom_weights(self):
        """Custom weights should change fused alpha."""
        w_dcp_heavy = {"dark_channel": 0.9, "contrast": 0.05, "color_collapse": 0.05}
        r1 = fuse_opacity_estimates(0.9, 0.1, 0.1, weights=w_dcp_heavy)
        assert r1["fused_alpha"] > 0.7  # dominated by dark_channel=0.9
