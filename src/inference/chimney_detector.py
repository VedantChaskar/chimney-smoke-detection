"""
Chimney Detection Module

This module provides a wrapper around YOLO models for detecting
chimneys in images. Part of Stage 1 of the smoke detection pipeline.
"""

from ultralytics import YOLO
from pathlib import Path
from typing import Union, List, Dict
import numpy as np


class ChimneyDetector:
    """
    YOLO-based chimney detector

    Wraps Ultralytics YOLO for chimney detection with configurable
    confidence and IOU thresholds.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        img_size: int = 800
    ):
        """
        Initialize the chimney detector

        Args:
            model_path: Path to trained YOLO model weights
            conf_threshold: Minimum confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            img_size: Image size for inference
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Use absolute path to prevent Ultralytics from caching in CWD
        self.model = YOLO(str(model_path.resolve()))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        print(f"  ✓ Chimney detector loaded: {model_path.name}")

    def detect(
        self,
        image_path: Union[str, Path],
        conf_threshold: float = None,
        visualize_all: bool = False
    ) -> List[Dict]:
        """
        Detect chimneys in an image

        Args:
            image_path: Path to input image
            conf_threshold: Override default confidence threshold
            visualize_all: Print all detections including low confidence

        Returns:
            List of detection dictionaries with 'bbox', 'confidence', 'class'
        """
        conf_threshold = conf_threshold or self.conf_threshold

        # Run YOLO detection
        results = self.model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False
        )

        detections = []
        all_detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                detection = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf),
                    'class': cls,
                    'class_name': 'chimney'
                }

                all_detections.append(detection)
                if conf >= conf_threshold:
                    detections.append(detection)

        if visualize_all and all_detections:
            print(f"    All YOLO detections:")
            for i, det in enumerate(all_detections, 1):
                print(f"      {i}. BBox={det['bbox']}, Conf={det['confidence']:.3f}")

        return detections

    def update_thresholds(self, conf_threshold: float = None, iou_threshold: float = None):
        """
        Update detection thresholds

        Args:
            conf_threshold: New confidence threshold
            iou_threshold: New IOU threshold
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': str(type(self.model)),
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'img_size': self.img_size
        }


if __name__ == '__main__':
    # Test the chimney detector
    from ..config import CHIMNEY_MODEL_YOLOV8, PROJECT_ROOT

    print("Testing Chimney Detector")
    print("=" * 50)

    try:
        detector = ChimneyDetector(
            model_path=CHIMNEY_MODEL_YOLOV8,
            conf_threshold=0.15
        )

        # Test with a sample image
        test_image = PROJECT_ROOT / "assets" / "test_image.jpg"

        if test_image.exists():
            print(f"\nTesting on: {test_image.name}")
            detections = detector.detect(test_image, visualize_all=True)

            print(f"\nResults:")
            print(f"  Detections: {len(detections)}")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. Confidence: {det['confidence']:.3f}")
                print(f"     BBox: {det['bbox']}")
        else:
            print(f"\nTest image not found: {test_image}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")

    print("\n" + "=" * 50)
