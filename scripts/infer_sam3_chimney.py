"""
SAM3 Inference Script for Chimney Smoke Detection

This script provides inference using SAM3 (Segment Anything Model 3) for chimney detection.
SAM3 is an open-vocabulary model that can detect objects based on text prompts.

Usage:
    # Single image
    python scripts/infer_sam3_chimney.py path/to/image.jpg

    # With custom output path
    python scripts/infer_sam3_chimney.py path/to/image.jpg --output result.jpg

    # Batch processing
    python scripts/infer_sam3_chimney.py path/to/images/ --batch --output-dir results/

    # Detection only (no smoke classification)
    python scripts/infer_sam3_chimney.py path/to/image.jpg --detection-only

    # Custom text prompt
    python scripts/infer_sam3_chimney.py path/to/image.jpg --text-prompt "chimney on roof"

    # With segmentation masks
    python scripts/infer_sam3_chimney.py path/to/image.jpg --save-masks
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    OUTPUT_IMAGES_DIR,
    SMOKE_MODEL_MOBILENET,
    ensure_dir
)
from src.utils.tee_logger import TeeLogger


def run_detection_only(image_path: Path, detector, output_path: Path = None, save_masks: bool = False):
    """Run chimney detection only without smoke classification."""
    import cv2
    import numpy as np

    print(f"\nProcessing: {image_path.name}")
    print("Running SAM3 chimney detection...")

    if save_masks:
        results = detector.detect_with_masks(image_path, visualize_all=True)
        detections = results['detections']
        masks = results['masks']
    else:
        detections = detector.detect(image_path, visualize_all=True)
        masks = None

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

            # Draw segmentation mask if available
            if masks and i <= len(masks) and masks[i-1] is not None:
                mask = masks[i-1].squeeze()
                if mask.shape[:2] == image.shape[:2]:
                    # Create colored overlay for mask
                    colored_mask = np.zeros_like(image)
                    colored_mask[:, :] = [0, 255, 0]  # Green
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    image = cv2.addWeighted(
                        image, 1.0,
                        colored_mask * mask_binary[:, :, np.newaxis], 0.3,
                        0
                    )

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


def run_batch_detection(input_dir: Path, detector, output_dir: Path = None, save_masks: bool = False):
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
            output_path = output_dir / f"sam3_{img_path.name}"

        try:
            results = run_detection_only(img_path, detector, output_path, save_masks)
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
        description='Run inference using SAM3 for chimney smoke detection',
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
        default=0.3,
        help='Confidence threshold for chimney detection'
    )

    parser.add_argument(
        '--smoke-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for smoke classification'
    )

    parser.add_argument(
        '--text-prompt',
        type=str,
        default='chimney',
        help='Text prompt for SAM3 detection'
    )

    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='Save segmentation masks with detections'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Custom path to SAM3 checkpoint (default: download from HuggingFace)'
    )

    parser.add_argument(
        '--no-hf',
        action='store_true',
        help='Disable loading checkpoint from HuggingFace (requires --checkpoint)'
    )

    args = parser.parse_args()

    # Validate thresholds
    if not 0.0 <= args.chimney_threshold <= 1.0:
        parser.error("Chimney threshold must be between 0.0 and 1.0")
    if not 0.0 <= args.smoke_threshold <= 1.0:
        parser.error("Smoke threshold must be between 0.0 and 1.0")

    if args.no_hf and not args.checkpoint:
        parser.error("--no-hf requires --checkpoint to specify a local checkpoint path")

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    print("="*60)
    print("CHIMNEY SMOKE DETECTION - SAM3 INFERENCE")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Text Prompt: '{args.text_prompt}'")
    print(f"Chimney Threshold: {args.chimney_threshold}")
    if not args.detection_only:
        print(f"Smoke Threshold: {args.smoke_threshold}")
    print(f"Mode: {'Detection Only' if args.detection_only else 'Full Pipeline'}")
    print(f"Save Masks: {args.save_masks}")
    print("="*60)

    # Import here to avoid slow import if just checking help
    from src.inference.sam3_detector import SAM3ChimneyDetector

    # Create SAM3 detector
    print("\nInitializing SAM3 model...")
    detector = SAM3ChimneyDetector(
        conf_threshold=args.chimney_threshold,
        text_prompt=args.text_prompt,
        load_from_hf=not args.no_hf,
        checkpoint_path=args.checkpoint
    )

    if args.detection_only:
        # Detection-only mode
        if args.batch:
            output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_IMAGES_DIR / "sam3_batch"
            ensure_dir(output_dir)
            results = run_batch_detection(input_path, detector, output_dir, args.save_masks)
        else:
            output_path = Path(args.output) if args.output else OUTPUT_IMAGES_DIR / f"sam3_{input_path.name}"
            results = run_detection_only(input_path, detector, output_path, args.save_masks)

            print("\n" + "="*60)
            print("DETECTION RESULTS")
            print("="*60)
            for i, det in enumerate(results, 1):
                print(f"  Chimney {i}:")
                print(f"    BBox: {det['bbox']}")
                print(f"    Confidence: {det['confidence']:.3f}")
    else:
        # Full pipeline mode with SAM3
        from src.inference.pipeline import SmokeDetectionPipeline
        from src.inference.smoke_classifier import SmokeClassifier

        # Create smoke classifier
        smoke_classifier = SmokeClassifier(
            model_path=SMOKE_MODEL_MOBILENET,
            conf_threshold=args.smoke_threshold
        )

        if args.batch:
            output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_IMAGES_DIR / "sam3_batch"
            ensure_dir(output_dir)

            # Manual batch processing since we're using custom detector
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_paths = []
            for ext in extensions:
                image_paths.extend(input_path.glob(f'*{ext}'))
                image_paths.extend(input_path.glob(f'*{ext.upper()}'))

            print(f"\nBatch processing {len(image_paths)} images...")
            all_results = {}

            for img_path in image_paths:
                try:
                    output_file = output_dir / f"sam3_{img_path.name}"
                    result = process_single_image(
                        img_path, detector, smoke_classifier, output_file,
                        not args.no_visualize, not args.no_fallback
                    )
                    all_results[img_path.name] = result
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")
                    all_results[img_path.name] = []

            # Print summary
            print(f"\n{'='*60}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            for img_name, results in all_results.items():
                if results:
                    smoke_count = sum(1 for r in results if r.get('smoke_result', {}).get('prediction') == 'smoke')
                    print(f"  {img_name}: {len(results)} detection(s), {smoke_count} with smoke")
                else:
                    print(f"  {img_name}: No results")

        else:
            output_path = Path(args.output) if args.output else OUTPUT_IMAGES_DIR / f"sam3_{input_path.name}"
            results = process_single_image(
                input_path, detector, smoke_classifier, output_path,
                not args.no_visualize, not args.no_fallback
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


def process_single_image(image_path, detector, smoke_classifier, output_path, visualize, use_fallback):
    """Process a single image with SAM3 detection and smoke classification."""
    import cv2
    from PIL import Image

    print(f"\nProcessing: {image_path.name}")

    # Run detection
    detections = detector.detect(image_path)

    results = []
    image = cv2.imread(str(image_path))
    pil_image = Image.open(image_path).convert('RGB')

    if detections:
        print(f"  Found {len(detections)} chimney(s)")

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']

            # Crop chimney region for smoke classification
            chimney_crop = pil_image.crop((x1, y1, x2, y2))

            # Classify smoke
            smoke_result = smoke_classifier.classify(chimney_crop)

            result = {
                'chimney_bbox': det['bbox'],
                'chimney_confidence': det['confidence'],
                'smoke_result': smoke_result,
                'used_fallback': False
            }
            results.append(result)

            # Annotate image
            if visualize:
                # Color based on smoke prediction
                if smoke_result['prediction'] == 'smoke':
                    color = (0, 0, 255)  # Red for smoke
                else:
                    color = (0, 255, 0)  # Green for no smoke

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                label = f"Chimney {i}: {smoke_result['prediction']} ({smoke_result['confidence']:.1%})"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    elif use_fallback:
        print("  No chimneys detected, using full image fallback")

        # Use full image for smoke classification
        smoke_result = smoke_classifier.classify(pil_image)

        result = {
            'chimney_bbox': None,
            'chimney_confidence': None,
            'smoke_result': smoke_result,
            'used_fallback': True
        }
        results.append(result)

        # Annotate image
        if visualize:
            label = f"Full Image: {smoke_result['prediction']} ({smoke_result['confidence']:.1%})"
            cv2.putText(image, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Save annotated image
    if visualize and output_path:
        ensure_dir(output_path.parent)
        cv2.imwrite(str(output_path), image)
        print(f"  Saved: {output_path}")

    return results


if __name__ == '__main__':
    with TeeLogger("inference", "infer_sam3_chimney"):
        main()
