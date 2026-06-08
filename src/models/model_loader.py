"""
Model Loading Utilities

This module provides utility functions for loading trained models for both
chimney detection, smoke classification, and Ringlemann scale classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
from pathlib import Path
from typing import Union

from .smoke_cnn import SmokeCNN
from .ringlemann_model import RinglemannCNN
from ..config import (
    CHIMNEY_MODEL_YOLOV8,
    CHIMNEY_MODEL_YOLOV12,
    SMOKE_MODEL_MOBILENET,
    SMOKE_MODEL_CNN,
    RINGLEMANN_MODEL_MOBILENET,
    RINGLEMANN_MODEL_RESNET18,
    RINGLEMANN_MODEL_CNN,
    DEVICE
)


def load_chimney_detector(
    model_path: Union[str, Path] = None,
    model_type: str = 'yolov8'
) -> YOLO:
    """
    Load a trained chimney detection model (YOLO)

    Args:
        model_path: Path to model checkpoint. If None, uses default from config
        model_type: Type of YOLO model ('yolov8' or 'yolov12')

    Returns:
        Loaded YOLO model ready for inference
    """
    if model_path is None:
        if model_type == 'yolov8':
            model_path = CHIMNEY_MODEL_YOLOV8
        elif model_type == 'yolov12':
            model_path = CHIMNEY_MODEL_YOLOV12
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Use absolute path to prevent Ultralytics from caching in CWD
    model = YOLO(str(model_path.resolve()))
    return model


def load_smoke_classifier_mobilenet(
    model_path: Union[str, Path] = None,
    device: str = None
) -> torch.nn.Module:
    """
    Load a trained smoke classifier (MobileNetV2 based)

    Args:
        model_path: Path to model checkpoint. If None, uses default from config
        device: Device to load model on. If None, uses default from config

    Returns:
        Loaded MobileNetV2 model ready for inference
    """
    if model_path is None:
        model_path = SMOKE_MODEL_MOBILENET

    if device is None:
        device = DEVICE

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create MobileNetV2 model
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

    # Load checkpoint (handle both old and new checkpoint formats)
    # Note: weights_only=True cannot load dict checkpoints in PyTorch 2.x
    # For security, we validate that checkpoint only contains expected keys
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

    model.to(device)
    model.eval()

    return model


def load_smoke_classifier_cnn(
    model_path: Union[str, Path] = None,
    device: str = None
) -> SmokeCNN:
    """
    Load a trained smoke classifier (Custom CNN)

    Args:
        model_path: Path to model checkpoint. If None, uses default from config
        device: Device to load model on. If None, uses default from config

    Returns:
        Loaded SmokeCNN model ready for inference
    """
    if model_path is None:
        model_path = SMOKE_MODEL_CNN

    if device is None:
        device = DEVICE

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create and load model
    model = SmokeCNN()

    # Load checkpoint (handle both old and new checkpoint formats)
    # Note: weights_only=True cannot load dict checkpoints in PyTorch 2.x
    # For security, we validate that checkpoint only contains expected keys
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

    model.to(device)
    model.eval()

    return model


def load_smoke_classifier(
    model_type: str = 'mobilenet',
    model_path: Union[str, Path] = None,
    device: str = None
) -> torch.nn.Module:
    """
    Load a smoke classifier model (convenience function)

    Args:
        model_type: Type of model ('mobilenet' or 'cnn')
        model_path: Path to model checkpoint. If None, uses default from config
        device: Device to load model on. If None, uses default from config

    Returns:
        Loaded model ready for inference
    """
    if model_type == 'mobilenet':
        return load_smoke_classifier_mobilenet(model_path, device)
    elif model_type == 'cnn':
        return load_smoke_classifier_cnn(model_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mobilenet' or 'cnn'")


def get_model_info(model: torch.nn.Module) -> dict:
    """
    Get information about a PyTorch model

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        'is_training': model.training
    }


if __name__ == "__main__":
    # Test loading models
    print("Testing Model Loading Utilities")
    print("=" * 50)

    try:
        print("\n1. Loading Chimney Detector (YOLOv8)...")
        chimney_model = load_chimney_detector(model_type='yolov8')
        print(f"   ✓ Loaded successfully: {chimney_model}")
    except FileNotFoundError as e:
        print(f"   ✗ {e}")

    try:
        print("\n2. Loading Smoke Classifier (MobileNet)...")
        smoke_mobilenet = load_smoke_classifier(model_type='mobilenet')
        info = get_model_info(smoke_mobilenet)
        print(f"   ✓ Loaded successfully")
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")
    except FileNotFoundError as e:
        print(f"   ✗ {e}")

    try:
        print("\n3. Loading Smoke Classifier (Custom CNN)...")
        smoke_cnn = load_smoke_classifier(model_type='cnn')
        info = get_model_info(smoke_cnn)
        print(f"   ✓ Loaded successfully")
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")
    except FileNotFoundError as e:
        print(f"   ✗ {e}")

    try:
        print("\n4. Loading Ringlemann Classifier (MobileNet)...")
        ringlemann_mobilenet = load_ringlemann_classifier(model_type='mobilenet')
        info = get_model_info(ringlemann_mobilenet)
        print(f"   ✓ Loaded successfully")
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Size: {info['model_size_mb']:.2f} MB")
    except FileNotFoundError as e:
        print(f"   ✗ {e}")

    print("\n" + "=" * 50)
    print("Model loading tests completed!")


# ============================================================================
# RINGLEMANN CLASSIFICATION MODEL FUNCTIONS
# ============================================================================

def build_ringlemann_model(
    model_type: str = 'mobilenet',
    pretrained: bool = True,
    num_classes: int = 6
) -> nn.Module:
    """
    Build a Ringlemann classification model

    Args:
        model_type: Type of model ('mobilenet', 'resnet18', 'resnet34', 'custom_cnn')
        pretrained: Whether to use pretrained weights for transfer learning models
        num_classes: Number of output classes (default: 6 for R0-R5)

    Returns:
        PyTorch model ready for training
    """
    if model_type == 'mobilenet':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    elif model_type == 'resnet18':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'resnet34':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'custom_cnn':
        model = RinglemannCNN(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Use 'mobilenet', 'resnet18', 'resnet34', or 'custom_cnn'"
        )

    return model


def load_ringlemann_classifier(
    model_type: str = 'mobilenet',
    model_path: Union[str, Path] = None,
    device: str = None
) -> nn.Module:
    """
    Load a trained Ringlemann classifier model

    Args:
        model_type: Type of model ('mobilenet', 'resnet18', 'resnet34', 'custom_cnn')
        model_path: Path to model checkpoint. If None, uses default from config
        device: Device to load model on. If None, uses default from config

    Returns:
        Loaded model ready for inference
    """
    if device is None:
        device = DEVICE

    # Set default model paths based on model type
    if model_path is None:
        if model_type == 'mobilenet':
            model_path = RINGLEMANN_MODEL_MOBILENET
        elif model_type in ('resnet18', 'resnet34'):
            model_path = RINGLEMANN_MODEL_RESNET18
        elif model_type == 'custom_cnn':
            model_path = RINGLEMANN_MODEL_CNN
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Build model architecture (without pretrained weights)
    model = build_ringlemann_model(model_type=model_type, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

    model.to(device)
    model.eval()

    return model
