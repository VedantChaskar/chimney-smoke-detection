"""
SAM3 Chimney Detection Module

This module provides a wrapper around SAM3 (Segment Anything Model 3) for detecting
chimneys in images using text prompts. Part of Stage 1 of the smoke detection pipeline.

SAM3 is an open-vocabulary detector that can detect objects based on text descriptions,
making it flexible for chimney detection without requiring custom training.
"""

import sys
from pathlib import Path
from typing import Union, List, Dict, Optional
import numpy as np

import torch
from PIL import Image

# Add sam3 to path
SAM3_PATH = Path(__file__).parent.parent.parent / "sam3"
if str(SAM3_PATH) not in sys.path:
    sys.path.insert(0, str(SAM3_PATH))

# BPE vocabulary path (inside sam3 package)
BPE_PATH = SAM3_PATH / "assets" / "bpe_simple_vocab_16e6.txt.gz"


class SAM3ChimneyDetector:
    """
    SAM3-based chimney detector using text prompts

    Uses SAM3's open-vocabulary detection capabilities to detect chimneys
    in images based on text descriptions.
    """

    def __init__(
        self,
        conf_threshold: float = 0.3,
        text_prompt: str = "chimney",
        device: str = None,
        load_from_hf: bool = True,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize the SAM3 chimney detector

        Args:
            conf_threshold: Minimum confidence threshold for detections
            text_prompt: Text prompt for detection (default: "chimney")
            device: Device to run inference on ('cuda' or 'cpu')
            load_from_hf: Whether to load checkpoint from HuggingFace
            checkpoint_path: Optional path to local checkpoint
        """
        self.conf_threshold = conf_threshold
        self.text_prompt = text_prompt
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"  Loading SAM3 model on {self.device}...")

        # Import SAM3 modules
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise ImportError(
                f"SAM3 not found. Please install it with: "
                f"pip install -e {SAM3_PATH}\n"
                f"Error: {e}"
            )

        # Enable TF32 for Ampere GPUs
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Build the model
        self.model = build_sam3_image_model(
            bpe_path=str(BPE_PATH),
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_hf,
            enable_segmentation=True,
            enable_inst_interactivity=False
        )

        # Create the processor
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=conf_threshold
        )

        print(f"  ✓ SAM3 chimney detector loaded")
        print(f"    Text prompt: '{self.text_prompt}'")
        print(f"    Confidence threshold: {self.conf_threshold}")

    def detect(
        self,
        image_path: Union[str, Path],
        conf_threshold: float = None,
        text_prompt: str = None,
        visualize_all: bool = False
    ) -> List[Dict]:
        """
        Detect chimneys in an image using SAM3

        Args:
            image_path: Path to input image
            conf_threshold: Override default confidence threshold
            text_prompt: Override default text prompt
            visualize_all: Print all detections including low confidence

        Returns:
            List of detection dictionaries with 'bbox', 'confidence', 'class', 'mask'
        """
        conf_threshold = conf_threshold or self.conf_threshold
        text_prompt = text_prompt or self.text_prompt

        # Update processor confidence threshold if different
        if conf_threshold != self.processor.confidence_threshold:
            self.processor.confidence_threshold = conf_threshold

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Set image and run inference with text prompt
        with torch.inference_mode():
            # Use bfloat16 for efficiency on supported GPUs
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=(self.device == 'cuda')):
                inference_state = self.processor.set_image(image)
                inference_state = self.processor.set_text_prompt(
                    prompt=text_prompt,
                    state=inference_state
                )

        detections = []
        all_detections = []

        # Extract results
        if 'boxes' in inference_state and len(inference_state['boxes']) > 0:
            # Convert to float32 before numpy (bfloat16 not supported by numpy)
            boxes = inference_state['boxes'].float().cpu().numpy()
            scores = inference_state['scores'].float().cpu().numpy()
            masks = inference_state['masks'].float().cpu().numpy() if 'masks' in inference_state else None

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(scores[i])

                detection = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': 0,
                    'class_name': 'chimney'
                }

                # Include mask if available
                if masks is not None:
                    detection['mask'] = masks[i]

                all_detections.append(detection)
                if conf >= conf_threshold:
                    detections.append(detection)

        if visualize_all and all_detections:
            print(f"    All SAM3 detections:")
            for i, det in enumerate(all_detections, 1):
                status = "✓" if det['confidence'] >= conf_threshold else "✗"
                print(f"      {i}. [{status}] BBox={det['bbox']}, Conf={det['confidence']:.3f}")

        return detections

    def detect_with_masks(
        self,
        image_path: Union[str, Path],
        conf_threshold: float = None,
        text_prompt: str = None
    ) -> Dict:
        """
        Detect chimneys and return both bounding boxes and segmentation masks

        Args:
            image_path: Path to input image
            conf_threshold: Override default confidence threshold
            text_prompt: Override default text prompt

        Returns:
            Dictionary with 'detections', 'masks', and 'image_size'
        """
        detections = self.detect(image_path, conf_threshold, text_prompt)

        # Load image for size info
        image = Image.open(image_path)

        return {
            'detections': detections,
            'masks': [d.get('mask') for d in detections if 'mask' in d],
            'image_size': image.size  # (width, height)
        }

    def update_thresholds(self, conf_threshold: float = None):
        """
        Update detection threshold

        Args:
            conf_threshold: New confidence threshold
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            self.processor.confidence_threshold = conf_threshold

    def update_text_prompt(self, text_prompt: str):
        """
        Update the text prompt used for detection

        Args:
            text_prompt: New text prompt for detection
        """
        self.text_prompt = text_prompt

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'SAM3',
            'text_prompt': self.text_prompt,
            'conf_threshold': self.conf_threshold,
            'device': self.device,
            'resolution': 1008  # SAM3 uses 1008x1008 resolution
        }


if __name__ == '__main__':
    # Test the SAM3 chimney detector
    from ..config import PROJECT_ROOT

    print("Testing SAM3 Chimney Detector")
    print("=" * 50)

    try:
        detector = SAM3ChimneyDetector(
            conf_threshold=0.3,
            text_prompt="chimney"
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

    except Exception as e:
        print(f"\n✗ Error: {e}")

    print("\n" + "=" * 50)
