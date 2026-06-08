#!/usr/bin/env python3
# run.py — Temporal Reference: Stationary Camera Temporal Opacity Detection
"""
Usage
-----
    # Run from temporal_reference/ directory:
    python run.py

    # Custom alpha calibration constant:
    python run.py --alpha 94

    # Force HSV fallback (no SAM3 / no GPU):
    python run.py --no-sam

    # Point to a different input folder:
    python run.py --input /path/to/folder

Expects folder layout:
    Method_1_test/
    ├── ground_truth.jpg   ← no-smoke reference (required, exact name)
    ├── smoke_001.jpg
    └── ...

Outputs:
    outputs/annotated/<name>_annotated.jpg   ← one per smoke image
    outputs/results.csv                       ← summary table
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2

# Ensure temporal_reference/ is on path so relative imports in src/ work
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.sam3_loader    import SAM3Segmenter
from src.ground_truth   import load_ground_truth
from src.alignment      import align_to_ground_truth
from src.smoke_segmenter import segment_smoke_image
from src.opacity        import compute_opacity
from src.annotator      import save_annotated_image
from src.logger         import init_csv, log_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Method 1 — Temporal Opacity Detection")
    parser.add_argument("--alpha",  type=float, default=cfg.ALPHA_DEFAULT,
                        help=f"Opacity calibration constant (default: {cfg.ALPHA_DEFAULT})")
    parser.add_argument("--no-sam", action="store_true",
                        help="Skip SAM3 and use HSV brightness fallback")
    parser.add_argument("--input",  default=cfg.INPUT_FOLDER,
                        help=f"Input folder (default: {cfg.INPUT_FOLDER})")
    args = parser.parse_args()

    # ── Resolve paths relative to script location ─────────────────────────────
    script_dir  = Path(__file__).parent
    input_dir   = Path(args.input) if os.path.isabs(args.input) else script_dir / args.input
    gt_path     = input_dir / cfg.GROUND_TRUTH_FILENAME

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth image not found: {gt_path}")
        sys.exit(1)

    # ── Load SAM3 (once) ──────────────────────────────────────────────────────
    print("[Setup] Loading SAM3 segmenter...")
    segmenter = SAM3Segmenter() if not args.no_sam else _make_fallback_segmenter()

    # ── Process ground truth (once) ───────────────────────────────────────────
    print(f"\n[Setup] Processing ground truth: {gt_path}")
    ground_truth = load_ground_truth(str(gt_path), segmenter)

    # ── Find smoke images (case-insensitive extension matching) ──────────────
    exts = {"jpg", "jpeg", "png", "JPG", "JPEG", "PNG"}
    all_images = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lstrip(".").upper() in {"JPG", "JPEG", "PNG"}
    ]
    smoke_images = sorted(
        p for p in all_images
        if p.name.lower() != cfg.GROUND_TRUTH_FILENAME.lower()
    )

    if not smoke_images:
        print(f"[ERROR] No smoke images found in {input_dir}")
        sys.exit(1)

    print(f"\n[Run] {len(smoke_images)} smoke image(s) found")
    print(f"[Run] Alpha = {args.alpha}")

    # ── Init CSV ──────────────────────────────────────────────────────────────
    # Outputs go next to run.py
    os.chdir(script_dir)
    init_csv()

    # ── Process each smoke image ──────────────────────────────────────────────
    for idx, image_path in enumerate(smoke_images):
        fname = image_path.name
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(smoke_images)}] {fname}")

        smoke_image = cv2.imread(str(image_path))
        if smoke_image is None:
            print(f"[ERROR] Could not read {image_path} — skipping")
            continue

        # 1. Align to ground truth
        aligned, align_ok, num_matches = align_to_ground_truth(
            smoke_image, ground_truth.image_bgr
        )

        # 2. Segment smoke image
        classified_masks, best_regions = segment_smoke_image(aligned, segmenter, ground_truth)

        # 3. Compute opacity (temporal ground truth comparison)
        result = compute_opacity(aligned, best_regions, ground_truth, alpha=args.alpha)

        print(f"[Result] smoke={result['smoke_type']} | "
              f"opacity={result['opacity_pct']} | "
              f"ringelman={result['ringelman']} | "
              f"formula={result['formula_used']}")
        if result["reason"] != "OK":
            print(f"[Result] note: {result['reason']}")

        # 4. Save annotated image
        stem        = image_path.stem
        output_path = os.path.join(cfg.ANNOTATED_DIR, f"{stem}_annotated.jpg")
        save_annotated_image(aligned, best_regions, result, output_path)

        # 5. Log to CSV
        log_result(str(image_path), result, align_ok, num_matches)

    print(f"\n{'='*60}")
    print(f"[Done] Results  → {cfg.RESULTS_CSV}")
    print(f"[Done] Images   → {cfg.ANNOTATED_DIR}/")


def _make_fallback_segmenter() -> SAM3Segmenter:
    """Return a SAM3Segmenter forced into HSV fallback mode (no SAM3 loaded)."""
    seg            = SAM3Segmenter.__new__(SAM3Segmenter)
    seg.device     = cfg.SAM3_DEVICE
    seg._model     = None
    seg._processor = None
    return seg


if __name__ == "__main__":
    main()
