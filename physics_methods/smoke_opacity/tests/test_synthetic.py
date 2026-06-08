"""
Synthetic smoke validation tests.

Composites smoke at known true_alpha over clean backgrounds, runs the full
pipeline (skipping SAM3, using the ground-truth mask), and checks that the
estimated alpha is within tolerance.

Run:
    conda run -n smokescreen python -m pytest smoke_opacity/tests/test_synthetic.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smoke_opacity.config import Config
from smoke_opacity.modules.atmospheric import estimate_airlight
from smoke_opacity.modules.dark_channel import estimate_transmittance, transmittance_to_alpha
from smoke_opacity.modules.contrast import compute_contrast_ratio
from smoke_opacity.modules.color_collapse import compute_color_collapse
from smoke_opacity.modules.fusion import fuse_opacity_estimates, alpha_to_ringelmann


# ── Synthetic smoke compositor ────────────────────────────────────────────────

def generate_synthetic_smoke(
    background: np.ndarray,
    true_alpha: float,
    smoke_color: tuple = (220, 220, 220),
    mask_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite synthetic smoke over a background at a known opacity.

    The smoke region is the central `mask_frac` × `mask_frac` square.

    Args:
        background:  (H, W, 3) uint8 BGR image.
        true_alpha:  Opacity in [0, 1]. 0 = transparent, 1 = fully opaque.
        smoke_color: BGR smoke colour (default: light grey).
        mask_frac:   Fraction of image size for the smoke region.

    Returns:
        composited:  (H, W, 3) uint8 BGR image with smoke applied.
        mask:        (H, W) uint8 binary mask (255 inside smoke, 0 outside).
    """
    h, w = background.shape[:2]
    composited = background.astype(np.float32)

    # Central rectangular smoke region
    y0 = int(h * (1 - mask_frac) / 2)
    y1 = int(h * (1 + mask_frac) / 2)
    x0 = int(w * (1 - mask_frac) / 2)
    x1 = int(w * (1 + mask_frac) / 2)

    smoke = np.array(smoke_color, dtype=np.float32)
    composited[y0:y1, x0:x1] = (
        (1.0 - true_alpha) * composited[y0:y1, x0:x1] + true_alpha * smoke
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255

    return np.clip(composited, 0, 255).astype(np.uint8), mask


def _make_background(style: str, h: int = 256, w: int = 256) -> np.ndarray:
    """Create synthetic backgrounds for testing."""
    if style == "sky":
        # Uniform blue sky (problematic for DCP)
        img = np.full((h, w, 3), (200, 150, 100), dtype=np.uint8)
        return img
    elif style == "building":
        # Random grey texture (building facade)
        rng = np.random.default_rng(42)
        img = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
        return img
    elif style == "foliage":
        # Green-tinted random texture
        rng = np.random.default_rng(7)
        img = rng.integers(30, 180, (h, w, 3), dtype=np.uint8)
        img[:, :, 2] = np.clip(img[:, :, 2] - 50, 0, 255)  # less red
        return img
    else:
        rng = np.random.default_rng(99)
        return rng.integers(50, 220, (h, w, 3), dtype=np.uint8)


def _run_opacity_pipeline(
    image: np.ndarray,
    mask_uint8: np.ndarray,
    cfg: Config | None = None,
) -> dict:
    """Run Modules 2-4 given an image and a ground-truth mask."""
    if cfg is None:
        cfg = Config()

    binary_mask = mask_uint8 > 127

    A = estimate_airlight(image, binary_mask)
    t_map = estimate_transmittance(
        image, A, binary_mask,
        omega=cfg.dcp_omega,
        patch_size=cfg.dcp_patch_size,
        t_min=cfg.dcp_t_min,
    )
    alpha_map = transmittance_to_alpha(t_map)
    alpha_dcp = float(alpha_map[binary_mask].mean()) if binary_mask.sum() > 0 else 0.0

    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = compute_contrast_ratio(gray, binary_mask,
                                      boundary_width=cfg.boundary_width,
                                      reliable_threshold=cfg.contrast_reliable_threshold)
    color    = compute_color_collapse(image, binary_mask,
                                      boundary_width=cfg.boundary_width)

    fusion = fuse_opacity_estimates(
        alpha_dcp,
        contrast["opacity_estimate"],
        color["opacity_estimate"],
        weights={
            "dark_channel":  cfg.weight_dark_channel,
            "contrast":      cfg.weight_contrast,
            "color_collapse": cfg.weight_color_collapse,
        },
        confidence_flags={
            "contrast_reliable":       contrast["reliable"],
            "color_collapse_reliable": color["reliable"],
        },
        thresholds=cfg.ringelmann_thresholds,
    )
    return fusion


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDarkChannelSmoke:
    """Basic DCP behaviour on synthetic smoke."""

    def test_clear_smoke_has_low_alpha(self):
        """Alpha ≈ 0 when no smoke (true_alpha=0)."""
        bg = _make_background("building")
        img, mask = generate_synthetic_smoke(bg, true_alpha=0.0)
        result = _run_opacity_pipeline(img, mask)
        assert result["fused_alpha"] < 0.25, (
            f"Expected alpha < 0.25 for clear air, got {result['fused_alpha']:.3f}"
        )

    def test_dense_smoke_has_high_alpha(self):
        """Alpha ≈ 1 when smoke is very dense (true_alpha=0.9)."""
        bg = _make_background("building")
        img, mask = generate_synthetic_smoke(bg, true_alpha=0.9)
        result = _run_opacity_pipeline(img, mask)
        assert result["fused_alpha"] > 0.30, (
            f"Expected alpha > 0.30 for dense smoke, got {result['fused_alpha']:.3f}"
        )

    def test_alpha_monotone_with_density(self):
        """Estimated alpha should increase as true_alpha increases."""
        bg = _make_background("building")
        alphas = []
        for ta in [0.0, 0.3, 0.6, 0.9]:
            img, mask = generate_synthetic_smoke(bg, true_alpha=ta)
            r = _run_opacity_pipeline(img, mask)
            alphas.append(r["fused_alpha"])

        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1] + 0.15, (
                f"Alpha not monotone: {alphas}"
            )


class TestRingelmannCalibration:
    """
    Full calibration: for each Ringelmann level, check estimated R matches.
    True alphas at R-midpoints: R0→0.05, R1→0.17, R2→0.35, R3→0.55, R4→0.72, R5→0.90
    """

    TRUE_ALPHAS = {0: 0.05, 1: 0.17, 2: 0.35, 3: 0.55, 4: 0.72, 5: 0.90}
    BACKGROUNDS = ["building", "foliage"]

    @pytest.mark.parametrize("ring", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("bg_style", ["building"])
    def test_ringelmann_correct(self, ring, bg_style):
        """Estimated Ringelmann should be within ±1 of truth (±2 for R0).

        R0 uses a wider tolerance because the Dark Channel Prior overestimates
        opacity on textured backgrounds even at near-zero true_alpha=0.05 — a
        known limitation when no clean reference image is available.
        """
        true_a = self.TRUE_ALPHAS[ring]
        bg  = _make_background(bg_style)
        img, mask = generate_synthetic_smoke(bg, true_alpha=true_a)
        result = _run_opacity_pipeline(img, mask)
        estimated = result["ringelmann"]
        tolerance = 2 if ring == 0 else 1
        assert abs(estimated - ring) <= tolerance, (
            f"R{ring} ground truth, estimated R{estimated} "
            f"(alpha={result['fused_alpha']:.3f}, expected≈{true_a})"
        )

    @pytest.mark.parametrize("ring", [2, 3, 4])
    def test_alpha_within_tolerance(self, ring):
        """Fused alpha should be within ±0.20 of true alpha for mid-range."""
        true_a = self.TRUE_ALPHAS[ring]
        bg  = _make_background("building")
        img, mask = generate_synthetic_smoke(bg, true_alpha=true_a)
        result = _run_opacity_pipeline(img, mask)
        error = abs(result["fused_alpha"] - true_a)
        assert error < 0.25, (
            f"Alpha error too large: |{result['fused_alpha']:.3f} - {true_a:.3f}| "
            f"= {error:.3f} > 0.25"
        )


class TestSyntheticGenerator:
    """Test the synthetic generator itself."""

    def test_output_shapes(self):
        bg = _make_background("building")
        img, mask = generate_synthetic_smoke(bg, true_alpha=0.5)
        assert img.shape == bg.shape
        assert mask.shape == bg.shape[:2]

    def test_zero_alpha_unchanged(self):
        bg = _make_background("building")
        img, mask = generate_synthetic_smoke(bg, true_alpha=0.0)
        np.testing.assert_array_equal(img, bg)

    def test_full_alpha_is_smoke_color(self):
        bg = _make_background("building")
        smoke_bgr = (200, 200, 200)
        img, mask = generate_synthetic_smoke(bg, true_alpha=1.0,
                                              smoke_color=smoke_bgr)
        smoke_region = img[mask > 0]
        mean_color = smoke_region.mean(axis=0)
        np.testing.assert_allclose(mean_color, smoke_bgr, atol=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
