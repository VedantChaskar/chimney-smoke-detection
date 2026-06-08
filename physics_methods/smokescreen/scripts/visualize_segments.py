#!/usr/bin/env python3
"""
scripts/visualize_segments.py

Debug script: run SAM3 segmentation + region classification on an image and
display / save a visualization of the detected regions.

Examples
--------
python scripts/visualize_segments.py --input assets/smoke1.jpg
python scripts/visualize_segments.py --input assets/ --output-dir outputs/segments/
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from src.segmentation.sam3_segmenter import load_segmenter
from src.segmentation.region_classifier import classify_regions, summarise_regions


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Overlay colours (BGR) for each region type
_COLOURS = {
    "smoke_dark":        (0,   0, 200),   # red
    "smoke_light":       (0, 180, 200),   # yellow
    "background_bright": (200, 180, 0),   # cyan
    "background_dark":   (60,  60,  60),  # dark grey
}


def parse_args():
    p = argparse.ArgumentParser(description="Visualize SAM3 region segmentation")
    p.add_argument("--input",      required=True)
    p.add_argument("--output-dir", default="outputs/segments")
    p.add_argument("--no-sam",     action="store_true")
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def visualize_one(image_bgr: np.ndarray, regions: dict, title: str = "") -> np.ndarray:
    """Build a 1×5 panel: original + one panel per region type."""
    h, w = image_bgr.shape[:2]
    panels = [image_bgr]

    for rtype, colour in _COLOURS.items():
        mask_dict = regions.get(rtype)
        panel = image_bgr.copy()
        if mask_dict is not None:
            seg = mask_dict["segmentation"]
            overlay = np.full_like(panel, colour, dtype=np.uint8)
            panel[seg] = (panel[seg] * 0.5 + overlay[seg] * 0.5).astype(np.uint8)
            b = mask_dict.get("brightness", 0)
            cv2.putText(panel, f"B={b:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(panel, "Not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        caption = rtype.replace("_", " ")
        cv2.putText(panel, caption, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        panels.append(panel)

    row = np.concatenate(panels, axis=1)
    return row


def main():
    args    = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    if in_path.is_file():
        images = [in_path]
    else:
        images = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print(f"No images at {in_path}")
        sys.exit(1)

    seg = load_segmenter(
        checkpoint=cfg.SAM3_CHECKPOINT,
        device=args.device,
        use_sam=not args.no_sam,
    )

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping: {img_path.name}")
            continue

        print(f"  Processing: {img_path.name} …")
        raw     = seg.segment_all_regions(img)
        regions = classify_regions(img, raw)
        summary = summarise_regions(regions)

        for k, v in summary.items():
            print(f"    {k:20s}: {v}")

        row = visualize_one(img, regions, title=img_path.stem)

        out_path = out_dir / (img_path.stem + "_segments.jpg")
        cv2.imwrite(str(out_path), row)
        print(f"    → saved {out_path}")


if __name__ == "__main__":
    main()
