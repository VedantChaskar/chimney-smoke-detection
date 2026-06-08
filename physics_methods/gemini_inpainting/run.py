#!/usr/bin/env python3
# run.py — Gemini Inpainting: Synthetic Ground Truth via Smoke Inpainting
"""
Usage
-----
    python run.py                              # run on default input folder
    python run.py --input /path/to/folder      # custom input folder
    python run.py --input /path/to/image.jpg   # single image accepted directly
    python run.py --alpha 94        # custom calibration constant
    python run.py --no-sam          # force HSV fallback (no SAM3 / no GPU)

Requires
--------
    GEMINI_API_KEY in gemini_inpainting/.env  (or as environment variable)
    CUDA GPU recommended for SAM3   (falls back to HSV without GPU)

Input folder layout
-------------------
    input_folder/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...

No ground truth image needed — Gemini generates it synthetically.

Outputs
-------
    outputs/masks/        — binary pixel masks for each image
    outputs/inpainted/    — Gemini inpainted results
    outputs/annotated/    — 2-panel comparison images
    outputs/results.csv   — per-image results table
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure gemini_inpainting/ is on path
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.sam3_loader     import SAM3Segmenter
from src.segmenter       import segment_image
from src.smoke_validator import validate_smoke
from src.mask_generator  import generate_pixel_mask
from src.inpainter       import inpaint_smoke
from src.opacity         import compute_opacity
from src.annotator       import save_annotated
from src.logger          import init_csv, log

_EMPTY = {"opacity_pct": None, "ringelman": None, "formula_used": "none", "reason": "skipped"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Method 2 — Synthetic Ground Truth")
    parser.add_argument("--alpha",   type=float, default=cfg.ALPHA_DEFAULT)
    parser.add_argument("--no-sam",         action="store_true",
                        help="Force HSV fallback; skip SAM3 model loading")
    parser.add_argument("--local-fallback", action="store_true",
                        help="Use OpenCV inpainting instead of Gemini (offline, less realistic)")
    parser.add_argument("--input",   default=cfg.INPUT_FOLDER)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_path = Path(args.input) if os.path.isabs(args.input) else script_dir / args.input

    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)

    _VALID_EXTS = {"JPG", "JPEG", "PNG"}

    if input_path.is_file():
        if input_path.suffix.lstrip(".").upper() not in _VALID_EXTS:
            print(f"[ERROR] Not a supported image format: {input_path}")
            sys.exit(1)
        image_paths = [input_path]
    else:
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lstrip(".").upper() in _VALID_EXTS
        )
        if not image_paths:
            print(f"[ERROR] No images found in {input_path}")
            sys.exit(1)

    # ── Load SAM3 ─────────────────────────────────────────────────────────────
    print("[Setup] Loading SAM3 segmenter...")
    segmenter = SAM3Segmenter() if not args.no_sam else _make_fallback_segmenter()

    print(f"[Run] {len(image_paths)} image(s) found  |  alpha={args.alpha}")
    if not cfg.GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY is not set — inpainting will fail gracefully")

    # ── Init output dirs + CSV ────────────────────────────────────────────────
    os.chdir(script_dir)
    for d in (cfg.MASKS_DIR, cfg.INPAINTED_DIR, cfg.ANNOTATED_DIR):
        os.makedirs(d, exist_ok=True)
    init_csv()

    # ── Process each image ────────────────────────────────────────────────────
    for idx, img_path in enumerate(image_paths):
        fname = img_path.name
        stem  = img_path.stem
        print(f"\n{'='*60}\n[{idx+1}/{len(image_paths)}] {fname}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[ERROR] Cannot load {img_path} — skipping")
            continue

        # Step 1: Segment
        _, best_regions = segment_image(image, segmenter)

        # Step 2: Validate smoke
        validation = validate_smoke(best_regions, image.shape)
        print(f"[Smoke] present={validation['smoke_present']} | "
              f"type={validation['smoke_type']} | "
              f"coverage={validation['coverage_ratio']:.3f} | "
              f"{validation['reason']}")

        if not validation["smoke_present"]:
            no_smoke = {"opacity_pct": 0.0, "ringelman": 0,
                        "formula_used": "none", "reason": validation["reason"]}
            log(str(img_path), validation, no_smoke, False)
            continue

        smoke_mask = validation["smoke_mask"]
        smoke_type = validation["smoke_type"]

        # Step 3: Generate pixel mask
        pixel_mask = generate_pixel_mask(smoke_mask, image.shape)
        cv2.imwrite(os.path.join(cfg.MASKS_DIR, f"{stem}_mask_pixel.png"), pixel_mask)

        # Step 4: Inpaint with pixel mask
        method = "OpenCV fallback" if args.local_fallback else "Gemini"
        print(f"[Inpaint] {method} inpainting (pixel mask)...")
        inpainted_pixel = inpaint_smoke(image, pixel_mask,
                                        use_local_fallback=args.local_fallback)
        inpaint_ok = inpainted_pixel is not None
        if inpainted_pixel is not None:
            cv2.imwrite(os.path.join(cfg.INPAINTED_DIR, f"{stem}_inpainted_pixel.jpg"),
                        inpainted_pixel)

        # Step 5: Opacity
        result = _EMPTY.copy()
        if inpainted_pixel is not None:
            result = compute_opacity(image, inpainted_pixel, smoke_mask,
                                     best_regions, smoke_type, alpha=args.alpha)

        print(f"[Result] Opacity={result['opacity_pct']}  Ringelman={result['ringelman']}")

        # Step 6: Annotate
        save_annotated(
            image,
            inpainted_pixel,
            smoke_mask, result,
            pixel_mask,
            os.path.join(cfg.ANNOTATED_DIR, f"{stem}_annotated.jpg"),
        )

        # Step 7: Log
        log(str(img_path), validation, result, inpaint_ok)

    print(f"\n{'='*60}")
    print(f"[Done] Results   → {cfg.RESULTS_CSV}")
    print(f"[Done] Annotated → {cfg.ANNOTATED_DIR}/")


def _make_fallback_segmenter() -> SAM3Segmenter:
    seg            = SAM3Segmenter.__new__(SAM3Segmenter)
    seg.device     = cfg.SAM3_DEVICE
    seg._model     = None
    seg._processor = None
    return seg


if __name__ == "__main__":
    main()
