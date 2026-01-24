"""
Extract Chimney Crops Script

Extract chimney regions from images using a trained detector.

Usage:
    python scripts/extract_chimneys.py input_folder/ output_folder/
    python scripts/extract_chimneys.py Dataset/ chimney_crops/ --conf 0.5
"""

import argparse
import sys
from pathlib import Path
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.config import CHIMNEY_MODEL_YOLOV8, IMAGE_EXTENSIONS


def extract_chimneys(
    input_dir: Path,
    output_dir: Path,
    model_path: Path = None,
    conf_threshold: float = 0.5,
    extensions: list = None
):
    """
    Extract chimney regions from images

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for chimney crops
        model_path: Path to chimney detector model
        conf_threshold: Confidence threshold for detection
        extensions: List of image file extensions to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_path = model_path or CHIMNEY_MODEL_YOLOV8
    extensions = extensions or IMAGE_EXTENSIONS

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CHIMNEY EXTRACTION")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_path.name}")
    print(f"Confidence: {conf_threshold}")
    print("=" * 60)

    # Load chimney detector
    print("\nLoading chimney detector...")
    # Use absolute path to prevent Ultralytics from caching in CWD
    model = YOLO(str(model_path.resolve()))

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    extracted = 0
    skipped = 0

    for img_path in image_files:
        print(f"Processing: {img_path.name}")

        # Run detection
        results = model(str(img_path), conf=conf_threshold, verbose=False)

        if len(results[0].boxes) > 0:
            # Get best detection (highest confidence)
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            x1, y1, x2, y2 = map(int, boxes[best_idx].xyxy[0])
            conf = boxes.conf[best_idx].item()

            # Load and crop image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ✗ Failed to read image")
                skipped += 1
                continue

            chimney_crop = img[y1:y2, x1:x2]

            # Save crop
            output_path = output_dir / f'crop_{img_path.name}'
            cv2.imwrite(str(output_path), chimney_crop)

            print(f"  ✓ Extracted chimney (conf: {conf:.2f}) -> {output_path.name}")
            extracted += 1
        else:
            print(f"  ⚠ No chimney detected")
            skipped += 1

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Extracted: {extracted}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(image_files)}")
    print(f"\nCrops saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract chimney regions from images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory with images'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for chimney crops'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to chimney detector model'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold for detection'
    )

    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None

    extract_chimneys(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_path=model_path,
        conf_threshold=args.conf
    )
