"""
Smoke Classification Module

This module provides smoke classification functionality for chimney regions.
Part of Stage 2 of the smoke detection pipeline.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, Optional

from ..models.smoke_cnn import SmokeCNN


class SmokeClassifier:
    """
    CNN-based smoke classifier

    Classifies image regions as containing smoke or no smoke.
    Supports both MobileNetV2 and custom CNN architectures.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str = 'mobilenet',
        conf_threshold: float = 0.5,
        device: Union[str, torch.device] = None
    ):
        """
        Initialize the smoke classifier

        Args:
            model_path: Path to trained model weights
            model_type: Type of model ('mobilenet' or 'cnn')
            conf_threshold: Minimum confidence threshold for predictions
            device: Device to run model on
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.class_names = ['no_smoke', 'smoke']

        print(f"  ✓ Smoke classifier loaded: {model_path.name} ({model_type})")

    def _load_model(self) -> nn.Module:
        """
        Load the model based on type

        Returns:
            Loaded PyTorch model
        """
        if self.model_type == 'mobilenet':
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, 2)

            # Load checkpoint (handle both old and new checkpoint formats)
            # Note: weights_only=True cannot load dict checkpoints in PyTorch 2.x
            # For security, we validate that checkpoint only contains expected keys
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Validate checkpoint structure for security
            if isinstance(checkpoint, dict):
                allowed_keys = {'model_state_dict', 'optimizer_state_dict', 'epoch', 'train_acc', 'val_acc'}
                if not set(checkpoint.keys()).issubset(allowed_keys):
                    unexpected_keys = set(checkpoint.keys()) - allowed_keys
                    raise ValueError(f"Suspicious checkpoint: unexpected keys {unexpected_keys}")

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Direct state dict (legacy format)
                model.load_state_dict(checkpoint)

        elif self.model_type == 'cnn':
            model = SmokeCNN(num_classes=2)

            # Load checkpoint with validation
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Validate checkpoint structure for security
            if isinstance(checkpoint, dict):
                allowed_keys = {'model_state_dict', 'optimizer_state_dict', 'epoch', 'train_acc', 'val_acc'}
                if not set(checkpoint.keys()).issubset(allowed_keys):
                    unexpected_keys = set(checkpoint.keys()) - allowed_keys
                    raise ValueError(f"Suspicious checkpoint: unexpected keys {unexpected_keys}")

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Direct state dict (legacy format)
                model.load_state_dict(checkpoint)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def classify(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        padding: int = 20
    ) -> Dict:
        """
        Classify smoke presence in an image or image region

        Args:
            image: PIL Image, numpy array, or path to image
            bbox: Optional (x1, y1, x2, y2) bounding box to crop
            padding: Padding to add around bbox in pixels

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        # Convert numpy to PIL if needed
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Crop to bbox if provided
        if bbox is not None:
            image = self._crop_with_padding(image, bbox, padding)

        # Transform and classify
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            'prediction': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                'no_smoke': probabilities[0][0].item(),
                'smoke': probabilities[0][1].item()
            },
            'meets_threshold': confidence.item() >= self.conf_threshold
        }

    def _crop_with_padding(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        padding: int
    ) -> Image.Image:
        """
        Crop image to bounding box with padding

        Args:
            image: PIL Image
            bbox: (x1, y1, x2, y2) bounding box
            padding: Padding in pixels

        Returns:
            Cropped PIL Image
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image.size

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)

        return image.crop((x1, y1, x2, y2))

    def update_threshold(self, conf_threshold: float):
        """
        Update confidence threshold

        Args:
            conf_threshold: New confidence threshold
        """
        self.conf_threshold = conf_threshold

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': self.model_type,
            'model_path': str(self.model_path),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),
            'conf_threshold': self.conf_threshold,
            'device': str(self.device),
            'is_training': self.model.training
        }


if __name__ == '__main__':
    # Test the smoke classifier
    from ..config import SMOKE_MODEL_MOBILENET, SMOKE_MODEL_CNN, PROJECT_ROOT

    print("Testing Smoke Classifier")
    print("=" * 50)

    # Test MobileNet
    try:
        print("\n1. Testing MobileNetV2:")
        classifier_mobilenet = SmokeClassifier(
            model_path=SMOKE_MODEL_MOBILENET,
            model_type='mobilenet'
        )

        info = classifier_mobilenet.get_model_info()
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")

        # Test with sample image
        test_image = PROJECT_ROOT / "assets" / "test_image.jpg"
        if test_image.exists():
            result = classifier_mobilenet.classify(test_image)
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.3f}")

    except FileNotFoundError as e:
        print(f"   ✗ {e}")

    # Test Custom CNN
    try:
        print("\n2. Testing Custom CNN:")
        classifier_cnn = SmokeClassifier(
            model_path=SMOKE_MODEL_CNN,
            model_type='cnn'
        )

        info = classifier_cnn.get_model_info()
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")

        # Test with sample image
        test_image = PROJECT_ROOT / "assets" / "test_image.jpg"
        if test_image.exists():
            result = classifier_cnn.classify(test_image)
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.3f}")

    except FileNotFoundError as e:
        print(f"   ✗ {e}")

    print("\n" + "=" * 50)
