#!/usr/bin/env python3
"""
scripts/run_inference.py

Run SmokeScreen inference on one image or a directory of images.

Examples
--------
# Single image
python scripts/run_inference.py --input assets/smoke1.jpg

# Batch (directory)
python scripts/run_inference.py --input assets/ --output-dir outputs/smokescreen/

# Force CPU / HSV fallback (no SAM3)
python scripts/run_inference.py --input assets/ --no-sam

# Custom alpha calibration constant
python scripts/run_inference.py --input assets/ --alpha 90
"""

import sys
import json
import argparse
from pathlib import Path

# ── Ensure project root on path ───────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import config as cfg
from src.pipeline import create_pipeline
from scripts._visualize import save_panel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="SmokeScreen brightness-ratio Ringelmann grader")
    p.add_argument("--input",      required=True, help="Image file or directory")
    p.add_argument("--output-dir", default="outputs/smokescreen",
                   help="Where to save annotated images and JSON results")
    p.add_argument("--alpha",      type=float, default=cfg.ALPHA_DEFAULT,
                   help="Calibration constant (default %(default)s)")
    p.add_argument("--no-sam",     action="store_true",
                   help="Disable SAM3; use HSV brightness fallback")
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--save-json",  action="store_true",
                   help="Save per-image JSON result files")
    return p.parse_args()


def collect_images(path: Path) -> list:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def print_result(r: dict, path: str = "") -> None:
    tag = Path(path).name if path else "result"
    print(f"\n{'─'*60}")
    print(f"  Image : {tag}")
    if "error" in r:
        print(f"  ERROR : {r['error']}")
        return
    print(f"  Grade : R{r['grade']}  ({r['label']})")
    print(f"  Opacity    : {r['opacity']:.1f}%")
    print(f"  Formula    : {r['formula']}  ({r['smoke_type']} smoke)")
    print(f"  Reliable   : {r['reliable']}")
    print(f"  L_smoke    : {r['L_smoke']}")
    print(f"  L_bright   : {r['L_bright']}")
    print(f"  L_dark     : {r['L_dark']}")
    print(f"  Elapsed    : {r['elapsed_s']}s")
    print(f"  Regions    : {json.dumps(r['regions_summary'], indent=4)}")


def main():
    args     = parse_args()
    in_path  = Path(args.input)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(in_path)
    if not images:
        print(f"No images found at {in_path}")
        sys.exit(1)

    print(f"Found {len(images)} image(s).  Building pipeline …")
    pipe = create_pipeline(
        use_sam=not args.no_sam,
        alpha=args.alpha,
        device=args.device,
    )

    all_results = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            r = {"image_path": str(img_path), "error": "Could not load"}
        else:
            r = pipe.run(img)
            r["image_path"] = str(img_path)

            # Save annotated panel
            out_img = out_dir / (img_path.stem + "_smokescreen.jpg")
            try:
                save_panel(img, r, out_img)
            except Exception as e:
                print(f"  Warning: could not save panel ({e})")

            # Optionally save JSON
            if args.save_json:
                out_json = out_dir / (img_path.stem + "_result.json")
                safe = {k: v for k, v in r.items()
                        if isinstance(v, (int, float, str, bool, dict, list, type(None)))}
                out_json.write_text(json.dumps(safe, indent=2))

        print_result(r, str(img_path))
        all_results.append(r)

    # Summary table
    print(f"\n{'═'*60}")
    print(f"  {'Image':<30}  {'Grade':>5}  {'Opacity':>8}  {'Formula':<10}")
    print(f"{'─'*60}")
    for r in all_results:
        name = Path(r["image_path"]).name[:30]
        if "error" in r:
            print(f"  {name:<30}  {'ERR':>5}")
        else:
            print(f"  {name:<30}  R{r['grade']:>4}  {r['opacity']:>7.1f}%  {r['formula']:<10}")
    print(f"{'═'*60}")
    print(f"  Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
