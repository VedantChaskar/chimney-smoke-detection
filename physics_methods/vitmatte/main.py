#!/usr/bin/env python3
"""
VitMatte Smoke Opacity Pipeline — CLI entrypoint.

Usage:
    python -m vitmatte.main --input path/to/image.jpg
    python -m vitmatte.main --input image.jpg --box 100,50,400,300
    python -m vitmatte.main --input image.jpg --point 250,175
    python -m vitmatte.main --input folder/ --batch
    python -m vitmatte.main --input image.jpg --no-sam   # HSV segmentation
    python -m vitmatte.main --input image.jpg --manual-mask mask.png
    python -m vitmatte.main --input image.jpg --ringelmann-method peak_region

Outputs (in outputs/vitmatte/):
    <stem>_panel.png      — 2×3 debug panel
    <stem>_overlay.jpg    — annotated single-image overlay
    <stem>_alpha.png      — raw 16-bit alpha map
    <stem>_result.json    — all numeric results
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_PKG_ROOT = Path(__file__).parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from vitmatte.config import Config, HF_CACHE_DIR
from vitmatte.utils.image_io import (
    load_image, save_image, save_alpha_png, extract_frame, collect_images
)
from vitmatte.modules.segmentation import SmokeSegmenter
from vitmatte.modules.trimap import generate_trimap, adaptive_trimap
from vitmatte.modules.matting import AlphaEstimator
from vitmatte.modules.validation import validate_alpha_map
from vitmatte.modules.ringelmann import compute_ringelmann
from vitmatte.modules.visualization import (
    annotate_overlay, create_output_panel
)


def run_pipeline(
    image_bgr: np.ndarray,
    config: Config,
    segmenter: SmokeSegmenter,
    alpha_estimator: AlphaEstimator,
    manual_mask: np.ndarray | None = None,
    prompt: dict | None = None,
) -> dict:
    """Execute the full pipeline on one BGR image. Returns complete result dict."""

    # ── Module 1: Segmentation ─────────────────────────────────────────────
    if manual_mask is not None:
        binary = (manual_mask > 127).astype(bool)
        soft   = binary.astype(np.float32)
        seg    = {
            "binary_mask":     binary,
            "soft_mask":       soft,
            "bbox":            (0, 0, image_bgr.shape[1], image_bgr.shape[0]),
            "mask_area_ratio": float(binary.sum()) / binary.size,
            "confidence":      1.0,
            "method":          "manual",
        }
    else:
        seg = segmenter.segment(image_bgr, prompt=prompt)

    binary_mask = seg["binary_mask"]
    soft_mask   = seg["soft_mask"]

    if not binary_mask.any():
        print("  ⚠ No smoke detected — returning R0.")
        h, w = image_bgr.shape[:2]
        return _empty_result(h, w, seg["method"])

    # ── Module 2: Trimap ───────────────────────────────────────────────────
    if config.trimap_adaptive and manual_mask is None:
        trimap = adaptive_trimap(
            binary_mask, soft_mask, image_bgr,
            soft_mask_low=config.trimap_soft_mask_low,
            soft_mask_high=config.trimap_soft_mask_high,
        )
    else:
        trimap = generate_trimap(
            binary_mask, soft_mask,
            erosion_size=config.trimap_erosion_size,
            dilation_size=config.trimap_dilation_size,
            soft_mask_low=config.trimap_soft_mask_low,
            soft_mask_high=config.trimap_soft_mask_high,
        )

    # ── Module 3: Alpha Matting ────────────────────────────────────────────
    matting_out = alpha_estimator.estimate_alpha(image_bgr, trimap)
    alpha_map   = matting_out["alpha_map_masked"]

    # ── Module 4: Validation ───────────────────────────────────────────────
    validation = validate_alpha_map(image_bgr, alpha_map, binary_mask, trimap)

    # ── Module 5: Ringelmann ───────────────────────────────────────────────
    ring_result = compute_ringelmann(
        alpha_map, binary_mask, method=config.ringelmann_method
    )
    ring_result["segmentation_method"] = seg["method"]
    ring_result["quality"]             = validation["overall_quality"]

    return {
        "ring_result":  ring_result,
        "validation":   validation,
        "alpha_map":    alpha_map,
        "trimap":       trimap,
        "seg":          seg,
    }


def process_single(
    image_bgr: np.ndarray,
    stem: str,
    config: Config,
    segmenter: SmokeSegmenter,
    alpha_estimator: AlphaEstimator,
    out_dir: Path,
    manual_mask: np.ndarray | None = None,
    prompt: dict | None = None,
) -> dict:
    """Run pipeline + save all outputs. Returns ring_result dict."""
    out = run_pipeline(image_bgr, config, segmenter, alpha_estimator,
                       manual_mask, prompt)
    rr   = out["ring_result"]
    val  = out["validation"]
    seg  = out["seg"]
    alpha_map = out["alpha_map"]
    trimap    = out["trimap"]

    # ── Console ────────────────────────────────────────────────────────────
    ring = rr["ringelmann"]
    stats = rr["spatial_stats"]
    print(f"\n{'='*55}")
    print(f"  Ringelmann   : {ring}  (continuous {rr['ringelmann_float']:.2f})")
    print(f"  Rep. alpha   : {rr['representative_alpha']:.3f}  [{rr['method_used']}]")
    print(f"  Quality      : {val['overall_quality']}  (score={val['confidence_score']:.2f})")
    print(f"  Segmentation : {seg['method']}  (area={seg['mask_area_ratio']*100:.1f}%)")
    print(f"  Spatial α    : mean={stats['mean_alpha']:.3f}  "
          f"p75={stats['p75_alpha']:.3f}  p90={stats['p90_alpha']:.3f}")
    for w in val.get("warnings", []):
        print(f"  ⚠ {w}")
    print(f"{'='*55}\n")

    # ── Save outputs ────────────────────────────────────────────────────────
    if config.save_overlay:
        overlay = annotate_overlay(image_bgr, seg["binary_mask"], alpha_map, rr)
        save_image(overlay, out_dir / f"{stem}_overlay.jpg")

    if config.save_panel:
        create_output_panel(
            image_bgr, seg["binary_mask"], trimap, alpha_map,
            rr, val, save_path=out_dir / f"{stem}_panel.png",
        )

    if config.save_alpha:
        save_alpha_png(alpha_map, out_dir / f"{stem}_alpha.png")

    if config.save_json:
        json_data = {
            "ringelmann":           rr["ringelmann"],
            "ringelmann_float":     rr["ringelmann_float"],
            "representative_alpha": rr["representative_alpha"],
            "method":               rr["method_used"],
            "quality":              val["overall_quality"],
            "confidence_score":     val["confidence_score"],
            "spatial_stats":        rr["spatial_stats"],
            "segmentation":         {
                "method":     seg["method"],
                "area_ratio": seg["mask_area_ratio"],
                "bbox":       list(seg["bbox"]),
            },
            "warnings": val.get("warnings", []),
        }
        jp = out_dir / f"{stem}_result.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w") as f:
            json.dump(json_data, f, indent=2)

    return rr


def _empty_result(h: int, w: int, seg_method: str) -> dict:
    from vitmatte.modules.ringelmann import _empty_result as _er
    er = _er("density_weighted")
    er["segmentation_method"] = seg_method
    er["quality"] = "low"
    return {
        "ring_result": er,
        "validation":  {"overall_quality": "low", "confidence_score": 0.0,
                        "checks": {}, "warnings": ["No smoke detected."]},
        "alpha_map":   np.zeros((h, w), dtype=np.float32),
        "trimap":      np.zeros((h, w), dtype=np.uint8),
        "seg":         {"binary_mask": np.zeros((h, w), dtype=bool),
                        "soft_mask":   np.zeros((h, w), dtype=np.float32),
                        "bbox": (0, 0, 0, 0), "mask_area_ratio": 0.0,
                        "confidence": 0.0, "method": seg_method},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate smoke opacity via ViTMatte alpha matting."
    )
    parser.add_argument("--input",   required=True, help="Image, video, or folder.")
    parser.add_argument("--frame",   type=int, default=0, help="Video frame index.")
    parser.add_argument("--box",     type=str, default=None,
                        help="SAM3 box prompt: 'x1,y1,x2,y2'.")
    parser.add_argument("--point",   type=str, default=None,
                        help="SAM3 point prompt: 'x,y'.")
    parser.add_argument("--text",    type=str, default=None,
                        help="SAM3 text prompt override (default: 'smoke').")
    parser.add_argument("--manual-mask", type=str, default=None,
                        help="Path to binary mask PNG (skip segmentation).")
    parser.add_argument("--batch",   action="store_true",
                        help="Process all images in input folder.")
    parser.add_argument("--no-sam",  action="store_true",
                        help="Use HSV fallback instead of SAM3.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--ringelmann-method", type=str, default="density_weighted",
                        choices=["mean", "p75", "density_weighted", "peak_region"])
    parser.add_argument("--model",   type=str,
                        default="hustvl/vitmatte-small-composition-1k",
                        help="ViTMatte HuggingFace model ID.")
    parser.add_argument("--no-panel", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.device             = args.device
    cfg.ringelmann_method  = args.ringelmann_method
    cfg.matting_model      = args.model
    if args.no_panel:
        cfg.save_panel = False

    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prompt
    prompt = None
    if args.box:
        coords = [int(x) for x in args.box.split(",")]
        prompt = {"box": tuple(coords)}
    elif args.point:
        x, y = [int(v) for v in args.point.split(",")]
        prompt = {"point": (x, y)}
    elif args.text:
        prompt = {"text": args.text}

    # Manual mask
    manual_mask = None
    if args.manual_mask:
        manual_mask = cv2.imread(args.manual_mask, cv2.IMREAD_GRAYSCALE)
        if manual_mask is None:
            print(f"ERROR: cannot read mask '{args.manual_mask}'", file=sys.stderr)
            sys.exit(1)

    # Load models once
    print("Loading segmenter …")
    segmenter = SmokeSegmenter(
        device=cfg.device,
        sam_checkpoint=cfg.sam_checkpoint,
        conf_threshold=cfg.sam_conf_threshold,
        smoke_max_saturation=cfg.smoke_max_saturation,
        smoke_min_value=cfg.smoke_min_value,
        morph_close_ksize=cfg.morph_close_ksize,
        use_sam=not args.no_sam,
    )

    print("Loading ViTMatte …")
    alpha_estimator = AlphaEstimator(
        model_name=cfg.matting_model,
        device=cfg.device,
        max_size=cfg.matting_max_size,
        bilateral_d=cfg.matting_bilateral_d,
        bilateral_sigma=cfg.matting_bilateral_sigma,
        hf_cache_dir=HF_CACHE_DIR,
    )

    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        images = collect_images(input_path)
        if not images:
            print(f"No images found in {input_path}")
            sys.exit(1)
        print(f"\nProcessing {len(images)} images from {input_path} …")
        for img_path in images:
            print(f"\n→ {img_path.name}")
            try:
                img = load_image(img_path)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue
            process_single(img, img_path.stem, cfg, segmenter,
                           alpha_estimator, out_dir, manual_mask, prompt)

    elif input_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        img  = extract_frame(input_path, args.frame)
        stem = f"{input_path.stem}_f{args.frame:06d}"
        process_single(img, stem, cfg, segmenter,
                       alpha_estimator, out_dir, manual_mask, prompt)

    else:
        img = load_image(input_path)
        process_single(img, input_path.stem, cfg, segmenter,
                       alpha_estimator, out_dir, manual_mask, prompt)


if __name__ == "__main__":
    main()
