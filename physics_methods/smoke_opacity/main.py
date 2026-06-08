#!/usr/bin/env python3
"""
Smoke Opacity Estimation — CLI entrypoint.

Usage examples:
    python -m smoke_opacity.main --input path/to/image.jpg
    python -m smoke_opacity.main --input path/to/video.mp4 --frame 120
    python -m smoke_opacity.main --input path/to/image.jpg --prompt-box 100,50,400,300
    python -m smoke_opacity.main --input path/to/folder/ --batch
    python -m smoke_opacity.main --input img.jpg --no-sam   # HSV fallback only
    python -m smoke_opacity.main --input img.jpg --manual-mask mask.png

Outputs (in outputs/smoke_opacity/):
    - Console: Ringelmann number, fused alpha, per-method breakdown
    - <stem>_overlay.jpg : annotated single-image overlay
    - <stem>_panel.png   : 2×2 debug panel
    - <stem>_result.json : machine-readable results
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure package root is on path when invoked as a script
_PKG_ROOT = Path(__file__).parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from smoke_opacity.config import Config
from smoke_opacity.utils.image_io import load_image, save_image, extract_frame, collect_images
from smoke_opacity.modules.segmentation import SmokeSegmenter
from smoke_opacity.modules.atmospheric import estimate_airlight
from smoke_opacity.modules.dark_channel import estimate_transmittance, transmittance_to_alpha
from smoke_opacity.modules.contrast import compute_contrast_ratio
from smoke_opacity.modules.color_collapse import compute_color_collapse
from smoke_opacity.modules.fusion import fuse_opacity_estimates
from smoke_opacity.modules.visualization import annotate_overlay, create_output_panel


def run_pipeline(
    image_bgr: np.ndarray,
    config: Config,
    segmenter: SmokeSegmenter,
    manual_mask: np.ndarray | None = None,
    prompt: dict | None = None,
) -> dict:
    """
    Execute the full opacity estimation pipeline on a single BGR image.

    Returns a result dict containing all estimates and the visualisation arrays.
    """
    # ── Module 1: Segmentation ────────────────────────────────────────────
    if manual_mask is not None:
        h, w = image_bgr.shape[:2]
        binary = (manual_mask > 127).astype(bool)
        from smoke_opacity.utils.mask_ops import compute_boundary_band
        boundary = compute_boundary_band(binary.astype(np.uint8), config.boundary_width)
        seg_result = {
            "binary_mask":     binary,
            "soft_mask":       binary.astype(np.float32),
            "bbox":            (0, 0, w, h),
            "mask_area_ratio": binary.sum() / (h * w),
            "method":          "manual",
            "boundary_band":   boundary,
        }
    else:
        seg_result = segmenter.segment(image_bgr, prompt=prompt)

    binary_mask = seg_result["binary_mask"]

    if binary_mask.sum() == 0:
        print("  ⚠ No smoke region found — returning R0.")
        h, w = image_bgr.shape[:2]
        empty_alpha = np.zeros((h, w), dtype=np.float32)
        fusion = {
            "fused_alpha": 0.0, "ringelmann": 0, "confidence": "low",
            "per_method": {"dark_channel": 0.0, "contrast": 0.0, "color_collapse": 0.0},
            "weights_used": {},
        }
        fusion["segmentation_method"] = seg_result["method"]
        return {
            "fusion": fusion,
            "alpha_map": empty_alpha,
            "transmittance_map": np.ones((h, w), dtype=np.float32),
            "seg_result": seg_result,
            "airlight": np.array([0.9, 0.9, 0.9]),
        }

    # ── Module 2: Atmospheric light ───────────────────────────────────────
    A = estimate_airlight(
        image_bgr, binary_mask,
        patch_size=config.dcp_patch_size,
    )

    # ── Module 3a: Dark channel prior ─────────────────────────────────────
    t_map = estimate_transmittance(
        image_bgr, A, binary_mask,
        omega=config.dcp_omega,
        patch_size=config.dcp_patch_size,
        t_min=config.dcp_t_min,
        guided_radius=config.guided_filter_radius,
        guided_eps=config.guided_filter_eps,
    )
    alpha_map = transmittance_to_alpha(t_map)

    # Mean alpha inside smoke mask
    alpha_dcp = float(alpha_map[binary_mask].mean()) if binary_mask.sum() > 0 else 0.0

    # ── Module 3b: Contrast attenuation ──────────────────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    contrast_result = compute_contrast_ratio(
        gray, binary_mask,
        boundary_width=config.boundary_width,
        reliable_threshold=config.contrast_reliable_threshold,
    )

    # ── Module 3c: Color collapse ─────────────────────────────────────────
    color_result = compute_color_collapse(
        image_bgr, binary_mask,
        boundary_width=config.boundary_width,
    )

    # ── Module 4: Fusion ─────────────────────────────────────────────────
    weights = {
        "dark_channel":  config.weight_dark_channel,
        "contrast":      config.weight_contrast,
        "color_collapse": config.weight_color_collapse,
    }
    confidence_flags = {
        "contrast_reliable":       contrast_result["reliable"],
        "color_collapse_reliable": color_result["reliable"],
    }
    fusion = fuse_opacity_estimates(
        alpha_dcp,
        contrast_result["opacity_estimate"],
        color_result["opacity_estimate"],
        weights=weights,
        confidence_flags=confidence_flags,
        thresholds=config.ringelmann_thresholds,
    )
    fusion["segmentation_method"] = seg_result["method"]

    return {
        "fusion": fusion,
        "alpha_map": alpha_map,
        "transmittance_map": t_map,
        "seg_result": seg_result,
        "airlight": A,
        "contrast_result": contrast_result,
        "color_result": color_result,
    }


def process_single(
    image_bgr: np.ndarray,
    image_stem: str,
    config: Config,
    segmenter: SmokeSegmenter,
    out_dir: Path,
    manual_mask: np.ndarray | None = None,
    prompt: dict | None = None,
) -> dict:
    """Run pipeline + save outputs for one image. Returns fusion result dict."""
    pipeline_out = run_pipeline(image_bgr, config, segmenter, manual_mask, prompt)

    fusion        = pipeline_out["fusion"]
    alpha_map     = pipeline_out["alpha_map"]
    t_map         = pipeline_out["transmittance_map"]
    seg           = pipeline_out["seg_result"]

    # ── Console output ────────────────────────────────────────────────────
    ring = fusion["ringelmann"]
    print(f"\n{'='*50}")
    print(f"  Ringelmann:   {ring}")
    print(f"  Fused alpha:  {fusion['fused_alpha']:.3f}")
    print(f"  Confidence:   {fusion['confidence']}")
    print(f"  Segmentation: {fusion['segmentation_method']} "
          f"(area={seg['mask_area_ratio']*100:.1f}%)")
    print(f"  Per-method α:")
    for k, v in fusion["per_method"].items():
        print(f"    {k:<18}: {v:.3f}")
    print(f"{'='*50}\n")

    # ── Save visualisations ───────────────────────────────────────────────
    if config.save_overlay:
        overlay = annotate_overlay(image_bgr, seg["binary_mask"], alpha_map, fusion)
        save_image(overlay, out_dir / f"{image_stem}_overlay.jpg")

    if config.save_panel:
        create_output_panel(
            image_bgr, seg["binary_mask"], alpha_map, fusion,
            save_path=out_dir / f"{image_stem}_panel.png",
        )

    # ── Save JSON ─────────────────────────────────────────────────────────
    if config.save_json:
        json_data = {
            "ringelmann":   fusion["ringelmann"],
            "fused_alpha":  fusion["fused_alpha"],
            "confidence":   fusion["confidence"],
            "per_method":   fusion["per_method"],
            "segmentation": {
                "method":     fusion["segmentation_method"],
                "area_ratio": seg["mask_area_ratio"],
                "bbox":       list(seg["bbox"]),
            },
        }
        json_path = out_dir / f"{image_stem}_result.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return fusion


def build_segmenter(config: Config, use_sam: bool) -> SmokeSegmenter:
    return SmokeSegmenter(
        device=config.device,
        sam_checkpoint=config.sam_checkpoint,
        conf_threshold=config.sam_conf_threshold,
        smoke_max_saturation=config.smoke_max_saturation,
        smoke_min_value=config.smoke_min_value,
        morph_close_ksize=config.morph_close_ksize,
        boundary_width=config.boundary_width,
        use_sam=use_sam,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate smoke opacity and Ringelmann number from an image or video."
    )
    parser.add_argument("--input",  required=True, help="Image, video, or folder path.")
    parser.add_argument("--frame",  type=int, default=0,
                        help="Video frame index (default 0).")
    parser.add_argument("--prompt-box", type=str, default=None,
                        help="SAM prompt box as 'x1,y1,x2,y2'.")
    parser.add_argument("--prompt-text", type=str, default=None,
                        help="SAM text prompt override (default: 'smoke').")
    parser.add_argument("--manual-mask", type=str, default=None,
                        help="Path to a binary mask PNG (skip segmentation).")
    parser.add_argument("--batch", action="store_true",
                        help="Process all images in the input folder.")
    parser.add_argument("--no-sam", action="store_true",
                        help="Disable SAM3; use HSV fallback segmentation.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/smoke_opacity/).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device ('cuda' or 'cpu').")
    parser.add_argument("--no-panel", action="store_true",
                        help="Skip saving the debug panel figure.")
    args = parser.parse_args()

    # Config
    cfg = Config()
    cfg.device = args.device
    if args.no_panel:
        cfg.save_panel = False

    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAM prompt
    prompt = None
    if args.prompt_box:
        coords = [int(x) for x in args.prompt_box.split(",")]
        prompt = {"box": tuple(coords)}
    elif args.prompt_text:
        prompt = {"text": args.prompt_text}

    # Manual mask
    manual_mask = None
    if args.manual_mask:
        manual_mask = cv2.imread(args.manual_mask, cv2.IMREAD_GRAYSCALE)
        if manual_mask is None:
            print(f"ERROR: cannot read mask '{args.manual_mask}'", file=sys.stderr)
            sys.exit(1)

    # Segmenter (loaded once, reused across batch)
    print("Loading segmenter...")
    segmenter = build_segmenter(cfg, use_sam=not args.no_sam)

    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        # ── Batch mode ────────────────────────────────────────────────────
        images = collect_images(input_path)
        if not images:
            print(f"No images found in {input_path}")
            sys.exit(1)
        print(f"Processing {len(images)} images from {input_path}…")
        for img_path in images:
            print(f"\n→ {img_path.name}")
            try:
                img = load_image(img_path)
            except Exception as e:
                print(f"  SKIP (cannot load): {e}")
                continue
            process_single(img, img_path.stem, cfg, segmenter, out_dir, manual_mask, prompt)

    elif input_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        # ── Video mode ────────────────────────────────────────────────────
        print(f"Extracting frame {args.frame} from {input_path.name}…")
        img = extract_frame(input_path, args.frame)
        stem = f"{input_path.stem}_f{args.frame:06d}"
        process_single(img, stem, cfg, segmenter, out_dir, manual_mask, prompt)

    else:
        # ── Single image ──────────────────────────────────────────────────
        img = load_image(input_path)
        process_single(img, input_path.stem, cfg, segmenter, out_dir, manual_mask, prompt)


if __name__ == "__main__":
    main()
