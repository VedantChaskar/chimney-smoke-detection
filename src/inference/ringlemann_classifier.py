"""
Ringlemann Scale Classification Module

This module provides Ringlemann scale smoke density classification functionality.
Part of Stage 3 of the smoke detection pipeline.

Ringlemann Scale:
    - R0: No visible smoke (clear)
    - R1: Light grey smoke (20% opacity)
    - R2: Medium grey smoke (40% opacity)
    - R3: Dark grey smoke (60% opacity)
    - R4: Very dark smoke (80% opacity)
    - R5: Black smoke (100% opacity)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, List

from ..models.ringlemann_model import RinglemannCNN


class RinglemannClassifier:
    """
    CNN-based Ringlemann scale smoke density classifier

    Classifies smoke density on the Ringlemann scale (R0-R5).
    Supports MobileNetV2, ResNet, and custom CNN architectures.
    """

    # Ringlemann class names
    CLASS_NAMES = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']

    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str = 'mobilenet',
        device: Union[str, torch.device] = None
    ):
        """
        Initialize the Ringlemann classifier

        Args:
            model_path: Path to trained model weights (.pt file)
            model_type: Type of model ('mobilenet', 'resnet18', 'resnet34', 'custom_cnn')
            device: Device to run model on (auto-detects CUDA if None)
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.model_type = model_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Define image transform (same as validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.class_names = self.CLASS_NAMES.copy()

        print(f"  \u2713 Ringlemann classifier loaded: {model_path.name} ({model_type})")

    def _load_model(self) -> nn.Module:
        """
        Load the model based on type

        Returns:
            Loaded PyTorch model
        """
        num_classes = 6  # R0-R5

        if self.model_type == 'mobilenet':
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        elif self.model_type == 'resnet18':
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif self.model_type == 'resnet34':
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif self.model_type == 'custom_cnn':
            model = RinglemannCNN(num_classes=num_classes)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Validate checkpoint structure for security
        if isinstance(checkpoint, dict):
            allowed_keys = {
                'model_state_dict', 'optimizer_state_dict', 'epoch',
                'train_acc', 'val_acc', 'train_loss', 'val_loss', 'mae'
            }
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

        return model

    def classify(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        top_scale: float = 1.5,
        bottom_scale: float = 0.2,
        side_scale: float = 0.3
    ) -> Dict:
        """
        Classify smoke density on the Ringlemann scale

        Args:
            image: PIL Image, numpy array (BGR), or path to image
            bbox: Optional (x1, y1, x2, y2) bounding box to crop chimney region
            top_scale: Proportional padding above bbox (fraction of box height)
            bottom_scale: Proportional padding below bbox (fraction of box height)
            side_scale: Proportional padding on left/right (fraction of box width)

        Returns:
            Dictionary with:
                - ringlemann_rating: int (0-5)
                - confidence: float (max probability)
                - probabilities: dict with R0-R5 probabilities
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        # Convert numpy (BGR) to PIL (RGB) if needed
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Crop to bbox if provided
        if bbox is not None:
            image = self._crop_with_padding(image, bbox, top_scale, bottom_scale, side_scale)

        # Transform and classify
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Build probabilities dictionary
        prob_dict = {}
        for i, class_name in enumerate(self.class_names):
            prob_dict[class_name] = probabilities[0][i].item()

        return {
            'ringlemann_rating': predicted.item(),
            'ringlemann_class': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': prob_dict
        }

    def classify_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
        top_scale: float = 1.5,
        bottom_scale: float = 0.2,
        side_scale: float = 0.3
    ) -> List[Dict]:
        """
        Batch inference for efficiency

        Args:
            images: List of PIL Images, numpy arrays, or paths
            bboxes: Optional list of bounding boxes (one per image, can contain None)
            top_scale: Proportional padding above bbox (fraction of box height)
            bottom_scale: Proportional padding below bbox (fraction of box height)
            side_scale: Proportional padding on left/right (fraction of box width)

        Returns:
            List of result dictionaries (same format as classify())
        """
        if bboxes is None:
            bboxes = [None] * len(images)

        # Prepare all images
        processed_images = []
        for img, bbox in zip(images, bboxes):
            # Load image if path provided
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Crop to bbox if provided
            if bbox is not None:
                img = self._crop_with_padding(img, bbox, top_scale, bottom_scale, side_scale)

            # Transform
            img_tensor = self.transform(img)
            processed_images.append(img_tensor)

        # Stack into batch
        batch_tensor = torch.stack(processed_images).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

        # Build results
        results = []
        for i in range(len(images)):
            prob_dict = {}
            for j, class_name in enumerate(self.class_names):
                prob_dict[class_name] = probabilities[i][j].item()

            results.append({
                'ringlemann_rating': predictions[i].item(),
                'ringlemann_class': self.class_names[predictions[i].item()],
                'confidence': confidences[i].item(),
                'probabilities': prob_dict
            })

        return results

    def _crop_with_padding(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        top_scale: float,
        bottom_scale: float,
        side_scale: float
    ) -> Image.Image:
        """
        Crop image to bounding box with asymmetric proportional padding

        Args:
            image: PIL Image
            bbox: (x1, y1, x2, y2) bounding box
            top_scale: Proportional padding above bbox (fraction of box height)
            bottom_scale: Proportional padding below bbox (fraction of box height)
            side_scale: Proportional padding on left/right (fraction of box width)

        Returns:
            Cropped PIL Image
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image.size
        box_h = y2 - y1
        box_w = x2 - x1

        pad_top = int(box_h * top_scale)
        pad_bottom = int(box_h * bottom_scale)
        pad_sides = int(box_w * side_scale)

        x1 = max(0, x1 - pad_sides)
        y1 = max(0, y1 - pad_top)
        x2 = min(img_w, x2 + pad_sides)
        y2 = min(img_h, y2 + pad_bottom)

        return image.crop((x1, y1, x2, y2))

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
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'device': str(self.device),
            'is_training': self.model.training
        }


if __name__ == '__main__':
    # Test the Ringlemann classifier
    from ..config import RINGLEMANN_MODEL_MOBILENET, RINGLEMANN_MODEL_CNN, PROJECT_ROOT

    print("Testing Ringlemann Classifier")
    print("=" * 50)

    # Test MobileNet
    try:
        print("\n1. Testing MobileNetV2:")
        classifier_mobilenet = RinglemannClassifier(
            model_path=RINGLEMANN_MODEL_MOBILENET,
            model_type='mobilenet'
        )

        info = classifier_mobilenet.get_model_info()
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")
        print(f"   Classes: {info['class_names']}")

        # Test with sample image
        test_image = PROJECT_ROOT / "assets" / "test_image.jpg"
        if test_image.exists():
            result = classifier_mobilenet.classify(test_image)
            print(f"   Prediction: {result['ringlemann_class']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Probabilities: {result['probabilities']}")

    except FileNotFoundError as e:
        print(f"   \u2717 {e}")

    # Test Custom CNN
    try:
        print("\n2. Testing Custom CNN:")
        classifier_cnn = RinglemannClassifier(
            model_path=RINGLEMANN_MODEL_CNN,
            model_type='custom_cnn'
        )

        info = classifier_cnn.get_model_info()
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")

        # Test with sample image
        test_image = PROJECT_ROOT / "assets" / "test_image.jpg"
        if test_image.exists():
            result = classifier_cnn.classify(test_image)
            print(f"   Prediction: {result['ringlemann_class']}")
            print(f"   Confidence: {result['confidence']:.3f}")

    except FileNotFoundError as e:
        print(f"   \u2717 {e}")

    print("\n" + "=" * 50)
