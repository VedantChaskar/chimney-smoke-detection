"""
Single Image Inference Script

Quick command-line tool for running smoke detection on a single image.

Usage:
    python scripts/run_inference.py path/to/image.jpg
    python scripts/run_inference.py path/to/image.jpg --output result.jpg
    python scripts/run_inference.py path/to/image.jpg --no-visualize
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import SmokeDetectionPipeline
from src.config import OUTPUT_IMAGES_DIR


def main():
    parser = argparse.ArgumentParser(
        description='Run smoke detection on a single image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'image',
        type=str,
        help='Path to input image'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to save annotated output image'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization (don\'t show image)'
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
        '--chimney-model',
        type=str,
        default='/nfs/hpc/share/chaskarv/chimney-smoke-detection/experiments/chimney_detection_yolov12/exp1/weights/best.pt',
        help='Path to chimney detection model checkpoint (YOLOv12)'
    )

    args = parser.parse_args()

    # Validate confidence thresholds
    if not 0.0 <= args.chimney_threshold <= 1.0:
        parser.error("Chimney threshold must be between 0.0 and 1.0")
    if not 0.0 <= args.smoke_threshold <= 1.0:
        parser.error("Smoke threshold must be between 0.0 and 1.0")

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Validate chimney model
    chimney_model_path = Path(args.chimney_model)
    if not chimney_model_path.exists():
        print(f"Error: Chimney model not found: {chimney_model_path}")
        sys.exit(1)

    # Set output path
    if args.output is None:
        output_path = OUTPUT_IMAGES_DIR / f"annotated_{image_path.name}"
    else:
        output_path = Path(args.output)

    print("="*60)
    print("SMOKE DETECTION - SINGLE IMAGE INFERENCE")
    print("="*60)
    print(f"Input: {image_path}")
    print(f"Output: {output_path}")
    print(f"Chimney Model: {args.chimney_model}")
    print(f"Chimney Threshold: {args.chimney_threshold}")
    print(f"Smoke Threshold: {args.smoke_threshold}")
    print("="*60)

    # Initialize pipeline with YOLOv12 chimney detector
    pipeline = SmokeDetectionPipeline(chimney_model_path=args.chimney_model)

    # Update thresholds if specified
    if args.chimney_threshold != 0.15:
        pipeline.chimney_detector.update_thresholds(conf_threshold=args.chimney_threshold)
    if args.smoke_threshold != 0.5:
        pipeline.smoke_classifier.update_threshold(args.smoke_threshold)

    # Run inference
    results = pipeline.process_image(
        image_path=image_path,
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
        print("✓ Inference completed successfully")
    else:
        print("\n✗ No results generated")
        sys.exit(1)


if __name__ == '__main__':
    main()
