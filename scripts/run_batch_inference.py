"""
Batch Inference Script

Process multiple images in a directory for smoke detection.

Usage:
    python scripts/run_batch_inference.py input_folder/
    python scripts/run_batch_inference.py input_folder/ output_folder/
    python scripts/run_batch_inference.py input_folder/ output_folder/ --chimney-threshold 0.2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import SmokeDetectionPipeline
from src.config import OUTPUT_RESULTS_DIR, ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run smoke detection on multiple images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing input images'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        nargs='?',
        default=None,
        help='Directory to save annotated images (optional)'
    )

    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png'],
        help='Image file extensions to process'
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

    args = parser.parse_args()

    # Validate confidence thresholds
    if not 0.0 <= args.chimney_threshold <= 1.0:
        parser.error("Chimney threshold must be between 0.0 and 1.0")
    if not 0.0 <= args.smoke_threshold <= 1.0:
        parser.error("Smoke threshold must be between 0.0 and 1.0")

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Set output directory
    if args.output_dir is None:
        output_dir = OUTPUT_RESULTS_DIR
    else:
        output_dir = Path(args.output_dir)

    print("="*60)
    print("SMOKE DETECTION - BATCH INFERENCE")
    print("="*60)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Extensions: {', '.join(args.extensions)}")
    print(f"Chimney Threshold: {args.chimney_threshold}")
    print(f"Smoke Threshold: {args.smoke_threshold}")
    print("="*60)

    # Initialize pipeline
    pipeline = SmokeDetectionPipeline()

    # Update thresholds if specified
    if args.chimney_threshold != 0.15:
        pipeline.chimney_detector.update_thresholds(conf_threshold=args.chimney_threshold)
    if args.smoke_threshold != 0.5:
        pipeline.smoke_classifier.update_threshold(args.smoke_threshold)

    # Run batch processing
    results = pipeline.batch_process(
        input_dir=input_dir,
        output_dir=output_dir,
        extensions=args.extensions
    )

    # Export results to CSV (optional)
    export_results_csv(results, output_dir)

    print("\n✓ Batch processing completed successfully")


def export_results_csv(results, output_dir):
    """Export results to CSV file"""
    import csv

    output_dir = Path(output_dir)
    ensure_dir(output_dir)  # Create directory if it doesn't exist

    csv_path = output_dir / "results.csv"

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'image_name',
                'chimney_detected',
                'chimney_bbox',
                'chimney_confidence',
                'smoke_prediction',
                'smoke_confidence',
                'smoke_probability',
                'no_smoke_probability'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for img_name, img_results in results.items():
                if not img_results:
                    writer.writerow({
                        'image_name': img_name,
                        'chimney_detected': 'FAILED',
                        'chimney_bbox': '',
                        'chimney_confidence': '',
                        'smoke_prediction': '',
                        'smoke_confidence': '',
                        'smoke_probability': '',
                        'no_smoke_probability': ''
                    })
                    continue

                for result in img_results:
                    smoke_result = result['smoke_result']

                    writer.writerow({
                        'image_name': img_name,
                        'chimney_detected': 'No' if result.get('used_fallback') else 'Yes',
                        'chimney_bbox': str(result.get('chimney_bbox', '')),
                        'chimney_confidence': f"{result.get('chimney_confidence', 0):.3f}" if result.get('chimney_confidence') else '',
                        'smoke_prediction': smoke_result['prediction'],
                        'smoke_confidence': f"{smoke_result['confidence']:.3f}",
                        'smoke_probability': f"{smoke_result['probabilities']['smoke']:.3f}",
                        'no_smoke_probability': f"{smoke_result['probabilities']['no_smoke']:.3f}"
                    })

        print(f"\n  💾 Results exported to: {csv_path}")

    except Exception as e:
        print(f"\n  ⚠ Could not export CSV: {e}")


if __name__ == '__main__':
    main()
