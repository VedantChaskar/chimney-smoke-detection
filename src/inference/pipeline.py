"""
Multi-Stage Smoke Detection Pipeline

This module implements the complete inference pipeline combining:
    Stage 1: Chimney detection using YOLO
    Stage 2: Smoke classification using CNN (binary: smoke/no smoke)
    Stage 3: Ringlemann scale classification (optional: R0-R5 density rating)

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
from .ringlemann_classifier import RinglemannClassifier
from ..config import InferencePipelineConfig, ensure_dir


class SmokeDetectionPipeline:
    """
    Complete multi-stage pipeline for smoke detection

    Stage 1: Detect chimneys in images using YOLO
    Stage 2: Classify smoke presence in detected chimney regions (binary)
    Stage 3: (Optional) Classify smoke density on Ringlemann scale (R0-R5)
    """

    def __init__(
        self,
        config: InferencePipelineConfig = None,
        chimney_model_path: Union[str, Path] = None,
        smoke_model_path: Union[str, Path] = None,
        ringlemann_model_path: Union[str, Path] = None,
        ringlemann_model_type: str = 'mobilenet',
        device: str = None
    ):
        """
        Initialize the smoke detection pipeline

        Args:
            config: Configuration object with pipeline settings
            chimney_model_path: Path to chimney detector model (overrides config)
            smoke_model_path: Path to smoke classifier model (overrides config)
            ringlemann_model_path: Path to Ringlemann classifier model (optional)
            ringlemann_model_type: Type of Ringlemann model ('mobilenet', 'resnet18', etc.)
            device: Device to run models on (overrides config)
        """
        self.config = config or InferencePipelineConfig
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Stage 1: Chimney Detector
        chimney_path = chimney_model_path or self.config.CHIMNEY_MODEL
        self.chimney_detector = ChimneyDetector(
            model_path=chimney_path,
            conf_threshold=self.config.CHIMNEY_CONF_THRESHOLD,
            img_size=self.config.CHIMNEY_IMG_SIZE
        )

        # Initialize Stage 2: Smoke Classifier (binary)
        smoke_path = smoke_model_path or self.config.SMOKE_MODEL
        self.smoke_classifier = SmokeClassifier(
            model_path=smoke_path,
            conf_threshold=self.config.SMOKE_CONF_THRESHOLD,
            device=self.device
        )

        # Initialize Stage 3: Ringlemann Classifier (optional)
        self.ringlemann_classifier = None
        if ringlemann_model_path is not None:
            ringlemann_path = Path(ringlemann_model_path)
            if ringlemann_path.exists():
                self.ringlemann_classifier = RinglemannClassifier(
                    model_path=ringlemann_path,
                    model_type=ringlemann_model_type,
                    device=self.device
                )
            else:
                print(f"  \u26a0 Ringlemann model not found: {ringlemann_path}")

        print(f"\u2713 Pipeline initialized successfully")
        print(f"  Chimney Detector: {chimney_path}")
        print(f"  Smoke Classifier: {smoke_path}")
        if self.ringlemann_classifier:
            print(f"  Ringlemann Classifier: {ringlemann_model_path} ({ringlemann_model_type})")
        else:
            print(f"  Ringlemann Classifier: Not configured (Stage 3 disabled)")
        print(f"  Device: {self.device}")

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
            print("  \u26a0 No chimney detected")

            if use_fallback:
                print("  \u2192 Fallback: Classifying entire image")
                smoke_result = self.smoke_classifier.classify(image)

                print(f"  Result: {smoke_result['prediction']} "
                      f"(confidence: {smoke_result['confidence']:.3f})")

                result_dict = {
                    'chimney_bbox': None,
                    'chimney_confidence': None,
                    'smoke_result': smoke_result,
                    'ringlemann_result': None,
                    'used_fallback': True
                }

                # Stage 3: Ringlemann classification if smoke detected
                if (smoke_result['prediction'] == 'smoke' and
                    self.ringlemann_classifier is not None):
                    print("  Stage 3: Classifying Ringlemann rating...")
                    ringlemann_result = self.ringlemann_classifier.classify(image)
                    result_dict['ringlemann_result'] = ringlemann_result
                    print(f"  Ringlemann: {ringlemann_result['ringlemann_class']} "
                          f"(confidence: {ringlemann_result['confidence']:.3f})")

                results.append(result_dict)

                # Annotate image
                color = (0, 255, 0) if smoke_result['prediction'] == 'no_smoke' else (0, 0, 255)
                label = f"Full Image: {smoke_result['prediction']} ({smoke_result['confidence']:.2f})"
                if result_dict['ringlemann_result']:
                    label += f" | {result_dict['ringlemann_result']['ringlemann_class']}"
                cv2.putText(image_cv, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            else:
                print("  \u2717 No fallback enabled")
                return []

        else:
            print(f"  \u2713 Found {len(detections)} chimney(s)")

            for i, detection in enumerate(detections, 1):
                bbox = detection['bbox']
                chimney_conf = detection['confidence']

                print(f"\n  Chimney {i}:")
                print(f"    BBox: {bbox}")
                print(f"    Confidence: {chimney_conf:.3f}")

                # Stage 2: Classify smoke (binary)
                print(f"    Stage 2: Classifying smoke...")
                smoke_result = self.smoke_classifier.classify(
                    image, bbox,
                    top_scale=self.config.CROP_PAD_TOP_SCALE,
                    bottom_scale=self.config.CROP_PAD_BOTTOM_SCALE,
                    side_scale=self.config.CROP_PAD_SIDE_SCALE
                )

                print(f"    Result: {smoke_result['prediction']} "
                      f"(confidence: {smoke_result['confidence']:.3f})")

                result_dict = {
                    'chimney_bbox': bbox,
                    'chimney_confidence': chimney_conf,
                    'smoke_result': smoke_result,
                    'ringlemann_result': None,
                    'used_fallback': False
                }

                # Stage 3: Ringlemann classification if smoke detected
                if (smoke_result['prediction'] == 'smoke' and
                    self.ringlemann_classifier is not None):
                    print(f"    Stage 3: Classifying Ringlemann rating...")
                    ringlemann_result = self.ringlemann_classifier.classify(
                        image, bbox,
                        top_scale=self.config.CROP_PAD_TOP_SCALE,
                        bottom_scale=self.config.CROP_PAD_BOTTOM_SCALE,
                        side_scale=self.config.CROP_PAD_SIDE_SCALE
                    )
                    result_dict['ringlemann_result'] = ringlemann_result
                    print(f"    Ringlemann: {ringlemann_result['ringlemann_class']} "
                          f"(confidence: {ringlemann_result['confidence']:.3f})")

                results.append(result_dict)

                # Annotate image
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if smoke_result['prediction'] == 'no_smoke' else (0, 0, 255)

                cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 3)

                label1 = f"Chimney: {chimney_conf:.2f}"
                label2 = f"{smoke_result['prediction']}: {smoke_result['confidence']:.2f}"

                # Add Ringlemann rating to label if available
                if result_dict['ringlemann_result']:
                    r_result = result_dict['ringlemann_result']
                    label2 += f" | {r_result['ringlemann_class']}: {r_result['confidence']:.2f}"

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
        ringlemann_counts = {f'R{i}': 0 for i in range(6)}

        for img_name, img_results in results.items():
            if not img_results:
                print(f"{img_name}: FAILED")
                failed_count += 1
                continue

            for r in img_results:
                status = "Full Image" if r.get('used_fallback') else "Chimney"
                smoke = r['smoke_result']['prediction']
                conf = r['smoke_result']['confidence']

                # Build result string
                result_str = f"{img_name}: {status} -> {smoke.upper()} ({conf:.2f})"

                # Add Ringlemann rating if available
                if r.get('ringlemann_result'):
                    r_class = r['ringlemann_result']['ringlemann_class']
                    r_conf = r['ringlemann_result']['confidence']
                    result_str += f" | {r_class} ({r_conf:.2f})"
                    ringlemann_counts[r_class] += 1

                print(result_str)

                if smoke == 'smoke':
                    smoke_count += 1
                else:
                    no_smoke_count += 1

        print(f"\n{'='*60}")
        print(f"Total: {len(results)} images")
        print(f"Smoke: {smoke_count} | No Smoke: {no_smoke_count} | Failed: {failed_count}")

        # Print Ringlemann distribution if any were classified
        if any(ringlemann_counts.values()):
            ringlemann_str = " | ".join(f"{k}: {v}" for k, v in ringlemann_counts.items() if v > 0)
            print(f"Ringlemann Distribution: {ringlemann_str}")

        print(f"{'='*60}")


if __name__ == '__main__':
    # Test the pipeline
    from ..config import PROJECT_ROOT, RINGLEMANN_MODEL_MOBILENET

    # Initialize pipeline with optional Ringlemann classifier
    ringlemann_path = RINGLEMANN_MODEL_MOBILENET if RINGLEMANN_MODEL_MOBILENET.exists() else None

    pipeline = SmokeDetectionPipeline(
        ringlemann_model_path=ringlemann_path,
        ringlemann_model_type='mobilenet'
    )

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

            if result.get('ringlemann_result'):
                r = result['ringlemann_result']
                print(f"  Ringlemann Rating: {r['ringlemann_class']}")
                print(f"  Ringlemann Confidence: {r['confidence']:.1%}")
    else:
        print(f"Test image not found: {test_image}")
