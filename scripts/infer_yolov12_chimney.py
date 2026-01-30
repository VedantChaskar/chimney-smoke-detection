"""
YOLOv12 Inference Script for Chimney Smoke Detection

This script provides inference using YOLOv12 for chimney detection
with the trained checkpoint from experiments/chimney_detection_yolov12/exp1/weights/best.pt

Usage:
    # Single image
    python scripts/infer_yolov12_chimney.py path/to/image.jpg

    # With custom output path
    python scripts/infer_yolov12_chimney.py path/to/image.jpg --output result.jpg

    # Batch processing
    python scripts/infer_yolov12_chimney.py path/to/images/ --batch --output-dir results/

    # Detection only (no smoke classification)
    python scripts/infer_yolov12_chimney.py path/to/image.jpg --detection-only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import SmokeDetectionPipeline
from src.inference.chimney_detector import ChimneyDetector
from src.config import (
    OUTPUT_IMAGES_DIR,
    EXPERIMENTS_DIR,
    SMOKE_MODEL_MOBILENET,
    ensure_dir
)
from src.utils.tee_logger import TeeLogger

# YOLOv12 checkpoint path
YOLOV12_CHECKPOINT = EXPERIMENTS_DIR / "chimney_detection_yolov12" / "exp1" / "weights" / "best.pt"


def run_detection_only(image_path: Path, detector: ChimneyDetector, output_path: Path = None):
    """Run chimney detection only without smoke classification."""
    import cv2

    print(f"\nProcessing: {image_path.name}")
    print("Running YOLOv12 chimney detection...")

    detections = detector.detect(image_path, visualize_all=True)

    if not detections:
        print("  No chimneys detected")
        return []

    print(f"  Found {len(detections)} chimney(s)")

    # Annotate image if output path specified
    if output_path:
        image = cv2.imread(str(image_path))

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Add label
            label = f"Chimney {i}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ensure_dir(output_path.parent)
        cv2.imwrite(str(output_path), image)
        print(f"  Saved: {output_path}")

    return detections


def run_batch_detection(input_dir: Path, detector: ChimneyDetector, output_dir: Path = None):
    """Run detection on all images in a directory."""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []

    for ext in extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
        image_paths.extend(input_dir.glob(f'*{ext.upper()}'))

    print(f"\nBatch processing {len(image_paths)} images...")

    all_results = {}
    for img_path in image_paths:
        output_path = None
        if output_dir:
            output_path = output_dir / f"detected_{img_path.name}"

        try:
            results = run_detection_only(img_path, detector, output_path)
            all_results[img_path.name] = results
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            all_results[img_path.name] = []

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH DETECTION SUMMARY")
    print(f"{'='*60}")

    total_detections = 0
    for img_name, dets in all_results.items():
        count = len(dets)
        total_detections += count
        print(f"  {img_name}: {count} chimney(s)")

    print(f"\nTotal: {len(image_paths)} images, {total_detections} chimneys detected")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run inference using YOLOv12 for chimney smoke detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to input image or directory (with --batch)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to save annotated output image'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save batch results (used with --batch)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images in the input directory'
    )

    parser.add_argument(
        '--detection-only',
        action='store_true',
        help='Run chimney detection only (skip smoke classification)'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization'
    )

    parser.add_argument(
        '--no-fallback',
        action='store_true',
        help='Disable full-image fallback when no chimney detected'
    )

    parser.add_argument(
        '--chimney-threshold',
        type=float,
        default=0.15,
        help='Confidence threshold for chimney detection'
    )

    parser.add_argument(
        '--smoke-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for smoke classification'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Custom path to YOLOv12 checkpoint (default: exp1/weights/best.pt)'
    )

    args = parser.parse_args()

    # Validate thresholds
    if not 0.0 <= args.chimney_threshold <= 1.0:
        parser.error("Chimney threshold must be between 0.0 and 1.0")
    if not 0.0 <= args.smoke_threshold <= 1.0:
        parser.error("Smoke threshold must be between 0.0 and 1.0")

    # Get checkpoint path
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else YOLOV12_CHECKPOINT

    if not checkpoint_path.exists():
        print(f"Error: YOLOv12 checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    print("="*60)
    print("CHIMNEY SMOKE DETECTION - YOLOv12 INFERENCE")
    print("="*60)
    print(f"YOLOv12 Checkpoint: {checkpoint_path}")
    print(f"Input: {input_path}")
    print(f"Chimney Threshold: {args.chimney_threshold}")
    if not args.detection_only:
        print(f"Smoke Threshold: {args.smoke_threshold}")
    print(f"Mode: {'Detection Only' if args.detection_only else 'Full Pipeline'}")
    print("="*60)

    if args.detection_only:
        # Detection-only mode
        detector = ChimneyDetector(
            model_path=checkpoint_path,
            conf_threshold=args.chimney_threshold
        )

        if args.batch:
            output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_IMAGES_DIR / "yolov12_batch"
            ensure_dir(output_dir)
            results = run_batch_detection(input_path, detector, output_dir)
        else:
            output_path = Path(args.output) if args.output else OUTPUT_IMAGES_DIR / f"yolov12_{input_path.name}"
            results = run_detection_only(input_path, detector, output_path)

            print("\n" + "="*60)
            print("DETECTION RESULTS")
            print("="*60)
            for i, det in enumerate(results, 1):
                print(f"  Chimney {i}:")
                print(f"    BBox: {det['bbox']}")
                print(f"    Confidence: {det['confidence']:.3f}")
    else:
        # Full pipeline mode with YOLOv12
        pipeline = SmokeDetectionPipeline(
            chimney_model_path=checkpoint_path,
            smoke_model_path=SMOKE_MODEL_MOBILENET
        )

        # Update thresholds
        if args.chimney_threshold != 0.15:
            pipeline.chimney_detector.update_thresholds(conf_threshold=args.chimney_threshold)
        if args.smoke_threshold != 0.5:
            pipeline.smoke_classifier.update_threshold(args.smoke_threshold)

        if args.batch:
            output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_IMAGES_DIR / "yolov12_batch"
            results = pipeline.batch_process(input_path, output_dir)
        else:
            output_path = Path(args.output) if args.output else OUTPUT_IMAGES_DIR / f"yolov12_{input_path.name}"
            results = pipeline.process_image(
                image_path=input_path,
                output_path=output_path,
                visualize=not args.no_visualize,
                use_fallback=not args.no_fallback
            )

            # Print results
            if results:
                print("\n" + "="*60)
                print("FINAL RESULTS")
                print("="*60)

                for i, result in enumerate(results, 1):
                    if result.get('used_fallback'):
                        print(f"\nFull Image Analysis:")
                    else:
                        print(f"\nChimney {i}:")
                        if result['chimney_bbox']:
                            print(f"  Location: {result['chimney_bbox']}")
                            print(f"  Detection Confidence: {result['chimney_confidence']:.3f}")

                    smoke_result = result['smoke_result']
                    print(f"  Smoke Status: {smoke_result['prediction'].upper()}")
                    print(f"  Confidence: {smoke_result['confidence']:.1%}")
                    print(f"  Probabilities:")
                    print(f"    - No Smoke: {smoke_result['probabilities']['no_smoke']:.1%}")
                    print(f"    - Smoke: {smoke_result['probabilities']['smoke']:.1%}")

    print("\n" + "="*60)
    print("Inference completed successfully")
    print("="*60)


if __name__ == '__main__':
    with TeeLogger("inference", "infer_yolov12_chimney"):
        main()
