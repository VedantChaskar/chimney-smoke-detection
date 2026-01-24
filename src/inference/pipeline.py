"""
Two-Stage Smoke Detection Pipeline

This module implements the complete inference pipeline combining:
    Stage 1: Chimney detection using YOLO
    Stage 2: Smoke classification using CNN

The pipeline can process single images or batches with configurable thresholds.
"""

import torch
from pathlib import Path
from typing import Union, List, Dict, Optional
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .chimney_detector import ChimneyDetector
from .smoke_classifier import SmokeClassifier
from ..config import InferencePipelineConfig, ensure_dir


class SmokeDetectionPipeline:
    """
    Complete two-stage pipeline for smoke detection

    Stage 1: Detect chimneys in images using YOLO
    Stage 2: Classify smoke presence in detected chimney regions
    """

    def __init__(
        self,
        config: InferencePipelineConfig = None,
        chimney_model_path: Union[str, Path] = None,
        smoke_model_path: Union[str, Path] = None,
        device: str = None
    ):
        """
        Initialize the smoke detection pipeline

        Args:
            config: Configuration object with pipeline settings
            chimney_model_path: Path to chimney detector model (overrides config)
            smoke_model_path: Path to smoke classifier model (overrides config)
            device: Device to run models on (overrides config)
        """
        self.config = config or InferencePipelineConfig

        # Initialize Stage 1: Chimney Detector
        chimney_path = chimney_model_path or self.config.CHIMNEY_MODEL
        self.chimney_detector = ChimneyDetector(
            model_path=chimney_path,
            conf_threshold=self.config.CHIMNEY_CONF_THRESHOLD,
            img_size=self.config.CHIMNEY_IMG_SIZE
        )

        # Initialize Stage 2: Smoke Classifier
        smoke_path = smoke_model_path or self.config.SMOKE_MODEL
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.smoke_classifier = SmokeClassifier(
            model_path=smoke_path,
            conf_threshold=self.config.SMOKE_CONF_THRESHOLD,
            device=device
        )

        print(f"✓ Pipeline initialized successfully")
        print(f"  Chimney Detector: {chimney_path}")
        print(f"  Smoke Classifier: {smoke_path}")
        print(f"  Device: {device}")

    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        visualize: bool = False,
        use_fallback: bool = True
    ) -> List[Dict]:
        """
        Process a single image through the two-stage pipeline

        Args:
            image_path: Path to input image
            output_path: Path to save annotated output (optional)
            visualize: Whether to display the result
            use_fallback: If True and no chimney detected, classify entire image

        Returns:
            List of dictionaries containing results for each detection
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\nProcessing: {image_path.name}")

        # Load image once with OpenCV (avoid duplicate I/O)
        image_cv = cv2.imread(str(image_path))
        if image_cv is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to PIL only when needed (lazy conversion for smoke classifier)
        image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        # Stage 1: Detect chimneys
        print("Stage 1: Detecting chimneys...")
        detections = self.chimney_detector.detect(image_path)

        results = []

        if not detections:
            print("  ⚠ No chimney detected")

            if use_fallback:
                print("  → Fallback: Classifying entire image")
                smoke_result = self.smoke_classifier.classify(image)

                print(f"  Result: {smoke_result['prediction']} "
                      f"(confidence: {smoke_result['confidence']:.3f})")

                results.append({
                    'chimney_bbox': None,
                    'chimney_confidence': None,
                    'smoke_result': smoke_result,
                    'used_fallback': True
                })

                # Annotate image
                color = (0, 255, 0) if smoke_result['prediction'] == 'no_smoke' else (0, 0, 255)
                label = f"Full Image: {smoke_result['prediction']} ({smoke_result['confidence']:.2f})"
                cv2.putText(image_cv, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            else:
                print("  ✗ No fallback enabled")
                return []

        else:
            print(f"  ✓ Found {len(detections)} chimney(s)")

            for i, detection in enumerate(detections, 1):
                bbox = detection['bbox']
                chimney_conf = detection['confidence']

                print(f"\n  Chimney {i}:")
                print(f"    BBox: {bbox}")
                print(f"    Confidence: {chimney_conf:.3f}")

                # Stage 2: Classify smoke
                print(f"    Stage 2: Classifying smoke...")
                smoke_result = self.smoke_classifier.classify(image, bbox)

                print(f"    Result: {smoke_result['prediction']} "
                      f"(confidence: {smoke_result['confidence']:.3f})")

                results.append({
                    'chimney_bbox': bbox,
                    'chimney_confidence': chimney_conf,
                    'smoke_result': smoke_result,
                    'used_fallback': False
                })

                # Annotate image
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if smoke_result['prediction'] == 'no_smoke' else (0, 0, 255)

                cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 3)

                label1 = f"Chimney: {chimney_conf:.2f}"
                label2 = f"{smoke_result['prediction']}: {smoke_result['confidence']:.2f}"

                cv2.putText(image_cv, label1, (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image_cv, label2, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save annotated image
        if output_path:
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            cv2.imwrite(str(output_path), image_cv)
            print(f"\n  💾 Saved: {output_path}")

        # Visualize
        if visualize:
            plt.figure(figsize=(14, 10))
            plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Chimney Detection + Smoke Classification', fontsize=14)
            plt.tight_layout()
            plt.show()

        return results

    def batch_process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = None,
        extensions: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process multiple images in a directory

        Args:
            input_dir: Directory containing images
            output_dir: Directory to save annotated results (optional)
            extensions: Image file extensions to process (default: ['.jpg', '.png'])

        Returns:
            Dictionary mapping image names to results
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        extensions = extensions or ['.jpg', '.jpeg', '.png']
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.glob(f'*{ext}'))
            image_paths.extend(input_dir.glob(f'*{ext.upper()}'))

        print(f"\n{'='*60}")
        print(f"Batch Processing: {len(image_paths)} images")
        print(f"{'='*60}")

        if output_dir:
            output_dir = Path(output_dir)
            ensure_dir(output_dir)

        all_results = {}

        for idx, img_path in enumerate(image_paths):
            output_path = None
            if output_dir:
                output_path = output_dir / f"annotated_{img_path.name}"

            try:
                results = self.process_image(
                    image_path=img_path,
                    output_path=output_path,
                    visualize=False
                )
                all_results[img_path.name] = results
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
                all_results[img_path.name] = []

            # Clear GPU cache periodically to prevent memory accumulation
            if idx % 50 == 0 and idx > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("\n" + "-"*60)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results: Dict[str, List[Dict]]):
        """Print batch processing summary"""
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")

        smoke_count = 0
        no_smoke_count = 0
        failed_count = 0

        for img_name, img_results in results.items():
            if not img_results:
                print(f"{img_name}: FAILED")
                failed_count += 1
                continue

            for r in img_results:
                status = "Full Image" if r.get('used_fallback') else "Chimney"
                smoke = r['smoke_result']['prediction']
                conf = r['smoke_result']['confidence']

                print(f"{img_name}: {status} -> {smoke.upper()} ({conf:.2f})")

                if smoke == 'smoke':
                    smoke_count += 1
                else:
                    no_smoke_count += 1

        print(f"\n{'='*60}")
        print(f"Total: {len(results)} images")
        print(f"Smoke: {smoke_count} | No Smoke: {no_smoke_count} | Failed: {failed_count}")
        print(f"{'='*60}")


if __name__ == '__main__':
    # Test the pipeline
    from ..config import PROJECT_ROOT

    pipeline = SmokeDetectionPipeline()

    # Test with a sample image
    test_image = PROJECT_ROOT / "assets" / "test_image.jpg"

    if test_image.exists():
        results = pipeline.process_image(
            image_path=test_image,
            output_path=PROJECT_ROOT / "outputs" / "images" / "pipeline_test.jpg",
            visualize=True
        )

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        for i, result in enumerate(results, 1):
            print(f"\nDetection {i}:")
            print(f"  Smoke: {result['smoke_result']['prediction'].upper()}")
            print(f"  Confidence: {result['smoke_result']['confidence']:.1%}")
    else:
        print(f"Test image not found: {test_image}")
