"""
Synthetic smoke validation — ground truth alpha compositing.

Composites smoke at known true_alpha, runs the full pipeline
(skipping SAM3, using ground-truth mask + trimap), and checks that
ViTMatte recovers alpha within tolerance.

Run:
    conda run -n smokescreen python -m pytest vitmatte/tests/test_synthetic.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vitmatte.config import Config, HF_CACHE_DIR
from vitmatte.modules.matting import AlphaEstimator
from vitmatte.modules.trimap import generate_trimap
from vitmatte.modules.ringelmann import compute_ringelmann, alpha_to_ringelmann
from vitmatte.modules.validation import validate_alpha_map


# ── Synthetic smoke compositor ────────────────────────────────────────────────

def generate_synthetic_smoke(
    background: np.ndarray,
    alpha_target: float,
    smoke_color: tuple = (180, 180, 180),
    pattern: str = "gaussian_plume",
) -> dict:
    """
    Composite synthetic smoke over a background at a known opacity.

    Args:
        background:   (H, W, 3) clean BGR background.
        alpha_target: Peak alpha for the smoke region (0-1).
        smoke_color:  BGR smoke colour.
        pattern:      "uniform"        — constant alpha in central ellipse
                      "gaussian_plume" — 2D Gaussian blob

    Returns dict with:
        composite         : (H, W, 3) uint8 BGR smoky image
        alpha_gt          : (H, W) float32 ground truth alpha
        smoke_mask_gt     : (H, W) bool
        background        : original background
        peak_alpha        : alpha_target
        mean_alpha_in_mask: float
    """
    h, w   = background.shape[:2]
    alpha_map = np.zeros((h, w), dtype=np.float32)

    if pattern == "uniform":
        y0 = int(h * 0.25); y1 = int(h * 0.75)
        x0 = int(w * 0.25); x1 = int(w * 0.75)
        alpha_map[y0:y1, x0:x1] = alpha_target

    elif pattern == "gaussian_plume":
        cy, cx  = h // 2, w // 2
        sigma_y = h * 0.20
        sigma_x = w * 0.20
        ys, xs  = np.mgrid[0:h, 0:w]
        gauss   = np.exp(-((ys - cy)**2 / (2*sigma_y**2) +
                           (xs - cx)**2 / (2*sigma_x**2)))
        alpha_map = (gauss * alpha_target).astype(np.float32)

    smoke   = np.array(smoke_color, dtype=np.float32)
    bg      = background.astype(np.float32)
    alpha3  = alpha_map[:, :, np.newaxis]
    comp    = alpha3 * smoke + (1 - alpha3) * bg
    composite = np.clip(comp, 0, 255).astype(np.uint8)

    smoke_mask = alpha_map > 0.01

    return {
        "composite":          composite,
        "alpha_gt":           alpha_map,
        "smoke_mask_gt":      smoke_mask,
        "background":         background,
        "peak_alpha":         alpha_target,
        "mean_alpha_in_mask": float(alpha_map[smoke_mask].mean()) if smoke_mask.any() else 0.0,
    }


def _make_background(style: str, h: int = 256, w: int = 256) -> np.ndarray:
    rng = np.random.default_rng({"building": 42, "sky": 7,
                                  "foliage": 13, "water": 99}.get(style, 0))
    if style == "sky":
        img = np.full((h, w, 3), (200, 150, 80), dtype=np.uint8)  # blue-ish
        img += rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
    elif style == "building":
        img = rng.integers(60, 190, (h, w, 3), dtype=np.uint8)
    elif style == "foliage":
        img = rng.integers(30, 160, (h, w, 3), dtype=np.uint8)
        img[:, :, 2] = np.clip(img[:, :, 2].astype(int) - 40, 0, 255).astype(np.uint8)
    else:
        img = rng.integers(50, 220, (h, w, 3), dtype=np.uint8)
    return img


@pytest.fixture(scope="module")
def estimator():
    cfg = Config()
    return AlphaEstimator(
        model_name=cfg.matting_model,
        device=cfg.device,
        max_size=512,
        hf_cache_dir=HF_CACHE_DIR,
    )


# ── Synthetic generator tests ──────────────────────────────────────────────────

class TestSyntheticGenerator:
    def test_shapes(self):
        bg = _make_background("building")
        r  = generate_synthetic_smoke(bg, 0.5)
        assert r["composite"].shape == bg.shape
        assert r["alpha_gt"].shape  == bg.shape[:2]

    def test_zero_alpha_is_background(self):
        bg = _make_background("building")
        r  = generate_synthetic_smoke(bg, 0.0)
        np.testing.assert_array_equal(r["composite"], bg)

    def test_full_alpha_is_smoke_color(self):
        bg     = _make_background("building")
        smoke  = (200, 200, 200)
        r = generate_synthetic_smoke(bg, 1.0, smoke_color=smoke, pattern="uniform")
        region = r["composite"][r["smoke_mask_gt"]]
        np.testing.assert_allclose(region.mean(axis=0), smoke, atol=3.0)

    def test_alpha_gt_range(self):
        bg = _make_background("building")
        for a in [0.0, 0.3, 0.7, 1.0]:
            r = generate_synthetic_smoke(bg, a)
            assert r["alpha_gt"].min() >= 0.0
            assert r["alpha_gt"].max() <= 1.0 + 1e-5

    def test_gaussian_plume_has_gradient(self):
        bg = _make_background("building")
        r  = generate_synthetic_smoke(bg, 0.8, pattern="gaussian_plume")
        vals = r["alpha_gt"][r["smoke_mask_gt"]]
        assert vals.std() > 0.05, "Gaussian plume should have alpha gradation"


# ── Alpha estimation accuracy ──────────────────────────────────────────────────

class TestAlphaAccuracy:
    """
    Validate ViTMatte output on synthetic smoke images.

    **Known limitation**: ViTMatte was trained on solid objects (hair, portraits)
    where FG is fully opaque and alpha gradation only occurs at object boundaries.
    For semi-transparent smoke:
      - With FG (255) in trimap core: model forces alpha≈1 everywhere (over-estimate).
      - Without FG in trimap: model has no opacity anchor and underestimates.
    Absolute fractional alpha calibration from synthetic composites is therefore
    NOT tested here — instead we validate monotonicity, non-zero detection, and
    output shape/range, which are the properties we can reliably check.

    For absolute Ringelmann calibration on real smoke, use the smoke_opacity/
    DCP-based pipeline (validated separately in smoke_opacity/tests/test_synthetic.py).
    """

    def _all_unknown_trimap(self, mask, dilation=25):
        """Smoke-mode trimap: entire mask + dilation = unknown (128), rest = BG (0).
        No FG region, so ViTMatte estimates fractional alpha across the whole plume."""
        from vitmatte.utils.mask_ops import dilate_mask
        tm = np.zeros(mask.shape, dtype=np.uint8)
        tm[dilate_mask(mask.astype(np.uint8), dilation).astype(bool)] = 128
        return tm

    def test_alpha_nonzero_for_visible_smoke(self, estimator):
        """Dense smoke (alpha=0.8) should produce a non-zero mean alpha estimate."""
        bg   = np.full((256, 256, 3), [210, 190, 170], dtype=np.uint8)  # light bg
        syn  = generate_synthetic_smoke(bg, 0.8, smoke_color=(60, 60, 60),
                                         pattern="gaussian_plume")
        mask = syn["smoke_mask_gt"].astype(np.uint8)
        trimap = self._all_unknown_trimap(mask)
        result = estimator.estimate_alpha(syn["composite"], trimap)
        alpha_vals = result["alpha_map_masked"][mask.astype(bool)]
        assert alpha_vals.mean() > 0.02, (
            f"Dense smoke produced near-zero alpha: {alpha_vals.mean():.4f}"
        )

    def test_dense_smoke_higher_than_clear(self, estimator):
        """
        ViTMatte should produce higher representative alpha for clearly dense smoke
        (alpha=0.9, black on light background) vs completely clear air.
        Tests the extremes only — ViTMatte doesn't guarantee monotonicity between
        intermediate values on synthetic noise backgrounds.
        """
        bg    = np.full((256, 256, 3), [210, 190, 170], dtype=np.uint8)
        smoke = (50, 50, 50)  # dark smoke on light bg — maximum visual contrast

        # Dense smoke
        syn_dense = generate_synthetic_smoke(bg, 0.9, smoke_color=smoke,
                                              pattern="gaussian_plume")
        mask_d = syn_dense["smoke_mask_gt"].astype(np.uint8)
        tm_d   = self._all_unknown_trimap(mask_d)
        r_d    = estimator.estimate_alpha(syn_dense["composite"], tm_d)
        ring_d = compute_ringelmann(r_d["alpha_map_masked"],
                                    mask_d.astype(bool), method="density_weighted")

        # No smoke (clear background, same mask region)
        syn_clear = generate_synthetic_smoke(bg, 0.0, smoke_color=smoke,
                                              pattern="gaussian_plume")
        mask_c = syn_clear["smoke_mask_gt"].astype(np.uint8)
        if not mask_c.any():
            # alpha=0 mask might be empty; use same mask
            mask_c = mask_d
        tm_c   = self._all_unknown_trimap(mask_c)
        r_c    = estimator.estimate_alpha(syn_clear["composite"], tm_c)
        ring_c = compute_ringelmann(r_c["alpha_map_masked"],
                                    mask_c.astype(bool), method="density_weighted")

        assert ring_d["representative_alpha"] >= ring_c["representative_alpha"], (
            f"Dense smoke ({ring_d['representative_alpha']:.3f}) should give "
            f">= alpha than clear ({ring_c['representative_alpha']:.3f})"
        )

    @pytest.mark.parametrize("smoke_color,label", [
        ((60, 60, 60),   "black"),
        ((200, 200, 200), "white"),
    ])
    def test_high_contrast_smoke_detected(self, estimator, smoke_color, label):
        """High-contrast smoke (black on light / white on dark) should yield alpha > 0."""
        if label == "black":
            bg = np.full((256, 256, 3), [200, 190, 180], dtype=np.uint8)
        else:
            bg = np.full((256, 256, 3), [40, 50, 60], dtype=np.uint8)

        syn    = generate_synthetic_smoke(bg, 0.7, smoke_color=smoke_color,
                                           pattern="gaussian_plume")
        mask   = syn["smoke_mask_gt"].astype(np.uint8)
        trimap = self._all_unknown_trimap(mask)
        result = estimator.estimate_alpha(syn["composite"], trimap)
        ring_r = compute_ringelmann(result["alpha_map_masked"],
                                    mask.astype(bool), method="density_weighted")
        assert ring_r["representative_alpha"] > 0.01, (
            f"{label} smoke on contrasting bg produced near-zero alpha: "
            f"{ring_r['representative_alpha']:.3f}"
        )


# ── Trimap sensitivity ─────────────────────────────────────────────────────────

class TestTrimapSensitivity:
    def test_wider_trimap_covers_more_unknown_pixels(self):
        """
        Wider dilation creates more unknown-zone pixels — verifies the trimap
        geometry is correct before passing to ViTMatte.
        """
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 1   # central square

        tm_narrow = generate_trimap(mask, erosion_size=3, dilation_size=5)
        tm_wide   = generate_trimap(mask, erosion_size=3, dilation_size=20)

        n_unknown_narrow = int((tm_narrow == 128).sum())
        n_unknown_wide   = int((tm_wide   == 128).sum())
        assert n_unknown_wide > n_unknown_narrow, (
            f"Wide trimap has fewer unknown pixels ({n_unknown_wide}) "
            f"than narrow ({n_unknown_narrow})"
        )

    def test_alpha_in_unknown_zone_has_gradation(self, estimator):
        """
        ViTMatte should produce varying alpha values in the unknown zone —
        not all-zero or all-one. This validates the model is using visual cues.
        """
        bg  = np.full((256, 256, 3), [200, 190, 175], dtype=np.uint8)
        syn = generate_synthetic_smoke(bg, 0.7, smoke_color=(50, 50, 50),
                                        pattern="gaussian_plume")
        mask   = syn["smoke_mask_gt"].astype(np.uint8)
        from vitmatte.utils.mask_ops import dilate_mask
        tm = np.zeros(mask.shape, dtype=np.uint8)
        tm[dilate_mask(mask, 20).astype(bool)] = 128
        result = estimator.estimate_alpha(syn["composite"], tm)
        unknown_alpha = result["alpha_map"][tm == 128]
        assert len(unknown_alpha) > 100, "Unknown zone is too small"
        assert unknown_alpha.std() > 0.005, (
            f"Alpha in unknown zone has no gradation (std={unknown_alpha.std():.4f})"
        )


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_thin_smoke_no_crash(self, estimator):
        """
        Very thin smoke (alpha=0.05) — pipeline should not crash and should
        return valid arrays.  Note: ViTMatte cannot reliably detect near-invisible
        smoke without a clean reference background; accuracy is not tested here.
        """
        bg   = _make_background("building")
        syn  = generate_synthetic_smoke(bg, 0.05, pattern="uniform")
        mask = syn["smoke_mask_gt"].astype(np.uint8)
        trimap = generate_trimap(mask, erosion_size=5, dilation_size=15)
        result = estimator.estimate_alpha(syn["composite"], trimap)
        assert result["alpha_map"] is not None
        assert result["alpha_map"].shape == syn["composite"].shape[:2]
        assert 0.0 <= result["alpha_map"].min()
        assert result["alpha_map"].max() <= 1.0 + 1e-4

    def test_compressed_image(self, estimator):
        """JPEG-compressed input should not crash the pipeline."""
        import cv2, io, numpy as np
        bg   = _make_background("building")
        syn  = generate_synthetic_smoke(bg, 0.6, pattern="gaussian_plume")
        # Simulate JPEG compression at quality=20
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        _, buf = cv2.imencode(".jpg", syn["composite"], encode_param)
        compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        mask = syn["smoke_mask_gt"].astype(np.uint8)
        trimap = generate_trimap(mask, erosion_size=5, dilation_size=15)
        result = estimator.estimate_alpha(compressed, trimap)
        assert result["alpha_map"].shape == compressed.shape[:2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
