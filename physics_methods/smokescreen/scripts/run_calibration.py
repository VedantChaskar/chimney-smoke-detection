#!/usr/bin/env python3
"""
scripts/run_calibration.py

Grid-search the best alpha calibration constant against a labelled test set.

Test-set format (CSV or JSON):
    CSV:  image_path,grade
    JSON: [{"image": "path", "grade": 3}, ...]

Examples
--------
python scripts/run_calibration.py --test-set data/test_set/labels.csv
python scripts/run_calibration.py --test-set data/test_set/labels.json --save
"""

import sys
import json
import csv
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import config as cfg
from src.segmentation.sam3_segmenter import load_segmenter
from src.segmentation.region_classifier import classify_regions
from src.calibration.alpha_tuner import tune_alpha, save_alpha


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate SmokeScreen alpha constant")
    p.add_argument("--test-set",  required=True, help="CSV or JSON with image/grade pairs")
    p.add_argument("--alpha-lo",  type=float, default=cfg.ALPHA_SEARCH_RANGE[0])
    p.add_argument("--alpha-hi",  type=float, default=cfg.ALPHA_SEARCH_RANGE[1])
    p.add_argument("--step",      type=int,   default=cfg.ALPHA_SEARCH_STEP)
    p.add_argument("--no-sam",    action="store_true")
    p.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--save",      action="store_true",
                   help="Save best alpha to data/best_alpha.txt")
    return p.parse_args()


def load_labels(path: Path) -> list:
    """Return list of (image_path, true_grade) tuples."""
    ext = path.suffix.lower()
    rows = []
    if ext == ".csv":
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append((row.get("image_path") or row.get("image"), int(row["grade"])))
    elif ext == ".json":
        data = json.loads(path.read_text())
        for d in data:
            rows.append((d.get("image_path") or d.get("image"), int(d["grade"])))
    else:
        raise ValueError(f"Unsupported label format: {ext}")
    return rows


def main():
    args = parse_args()

    labels = load_labels(Path(args.test_set))
    print(f"Loaded {len(labels)} labelled images.")

    seg = load_segmenter(
        checkpoint=cfg.SAM3_CHECKPOINT,
        device=args.device,
        use_sam=not args.no_sam,
    )

    # Pre-segment all images (segment once, search alpha multiple times)
    print("Segmenting images …")
    pairs = []
    for img_path, true_grade in labels:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping (not found): {img_path}")
            continue
        raw  = seg.segment_all_regions(img)
        regs = classify_regions(img, raw)
        pairs.append((img, regs, true_grade))

    if not pairs:
        print("No valid images. Exiting.")
        sys.exit(1)

    print(f"\nRunning grid search (alpha {args.alpha_lo}–{args.alpha_hi}, step {args.step}) …\n")
    best_alpha, results = tune_alpha(
        pairs=pairs,
        search_range=(args.alpha_lo, args.alpha_hi),
        step=args.step,
        verbose=True,
    )

    if args.save:
        save_alpha(best_alpha)


if __name__ == "__main__":
    main()
