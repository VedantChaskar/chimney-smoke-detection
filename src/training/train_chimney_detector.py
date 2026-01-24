"""
Chimney Detector Training Script

Train YOLOv8/YOLOv12 models for chimney detection.
Uses centralized configuration from src.config.

Usage:
    python -m src.training.train_chimney_detector
    python -m src.training.train_chimney_detector --model yolo12 --epochs 50
"""

import argparse
import platform
import torch
from ultralytics import YOLO
from pathlib import Path

from ..config import (
    CHIMNEY_DATASET_V4,
    YOLOV8_PRETRAINED,
    YOLO11_PRETRAINED,
    YOLO12_PRETRAINED,
    CHIMNEY_YOLOV8_EXPERIMENTS,
    CHIMNEY_YOLOV12_EXPERIMENTS,
    ChimneyDetectionConfig,
    DEVICE
)


def train_chimney_detector(
    model_type: str = 'yolov8',
    dataset_version: str = 'v4',
    epochs: int = None,
    img_size: int = None,
    batch_size: int = None,
    patience: int = None,
    experiment_name: str = 'exp1',
    resume: bool = False
):
    """
    Train a chimney detection model

    Args:
        model_type: Type of YOLO model ('yolov8', 'yolo11', or 'yolo12')
        dataset_version: Dataset version to use ('v1', 'v3', 'v4', 'v6')
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size
        patience: Early stopping patience
        experiment_name: Name for this training experiment
        resume: Whether to resume from last checkpoint
    """
    print("=" * 60)
    print("CHIMNEY DETECTOR TRAINING")
    print("=" * 60)

    # Use config defaults if not specified
    epochs = epochs or ChimneyDetectionConfig.EPOCHS
    img_size = img_size or ChimneyDetectionConfig.IMG_SIZE
    batch_size = batch_size or ChimneyDetectionConfig.BATCH_SIZE
    patience = patience or ChimneyDetectionConfig.PATIENCE

    # Check GPU availability
    print(f"\nDevice Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        device = 0  # Use GPU 0
    else:
        print("  No GPU detected, using CPU")
        device = 'cpu'

    # Select pretrained model
    if model_type == 'yolov8':
        pretrained_model = YOLOV8_PRETRAINED
        project_dir = CHIMNEY_YOLOV8_EXPERIMENTS
    elif model_type == 'yolo11':
        pretrained_model = YOLO11_PRETRAINED
        project_dir = CHIMNEY_YOLOV8_EXPERIMENTS  # Use same folder
    elif model_type == 'yolo12':
        pretrained_model = YOLO12_PRETRAINED
        project_dir = CHIMNEY_YOLOV12_EXPERIMENTS
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Select dataset
    from ..config import PROJECT_ROOT
    dataset_map = {
        'v1': PROJECT_ROOT / 'data' / 'chimney_detection' / 'v1',
        'v3': PROJECT_ROOT / 'data' / 'chimney_detection' / 'v3',
        'v4': CHIMNEY_DATASET_V4,
        'v6': PROJECT_ROOT / 'data' / 'chimney_detection' / 'v6_yolo12'
    }

    dataset_dir = dataset_map.get(dataset_version)
    if not dataset_dir or not dataset_dir.exists():
        raise ValueError(f"Dataset not found: {dataset_dir}")

    data_yaml = dataset_dir / 'data.yaml'

    print(f"\nTraining Configuration:")
    print(f"  Model: {model_type}")
    print(f"  Pretrained weights: {pretrained_model.name}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patience: {patience}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Project dir: {project_dir}")

    # Load model
    print(f"\nLoading {model_type} model...")
    if not pretrained_model.exists():
        print(f"  Warning: Pretrained model not found at {pretrained_model}")
        print(f"  YOLO will attempt to download it automatically")

    # Use absolute path to prevent Ultralytics from caching in current working directory
    pretrained_path_absolute = pretrained_model.resolve()
    model = YOLO(str(pretrained_path_absolute))

    # Train the model
    print(f"\n{'=' * 60}")
    print("STARTING TRAINING")
    print("=" * 60)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,
        save=True,
        project=str(project_dir),
        name=experiment_name,
        verbose=True,
        device=device,
        workers=0 if device == 'cpu' else 4,  # Windows compatibility
        resume=resume,

        # Good defaults for dataset
        cos_lr=True,
        close_mosaic=10,
    )

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    best_model_path = project_dir / experiment_name / 'weights' / 'best.pt'
    print(f"\n✓ Best model saved to: {best_model_path}")

    if hasattr(results, 'box'):
        if hasattr(results.box, 'map50'):
            print(f"✓ Best mAP50: {results.box.map50:.3f}")
        if hasattr(results.box, 'map'):
            print(f"✓ Best mAP50-95: {results.box.map:.3f}")

    # Provide platform-specific copy instructions
    print(f"\nTo copy best model to models directory:")
    if platform.system() == "Windows":
        print(f"  copy \"{best_model_path}\" \"models\\chimney_detection\\{model_type}_best.pt\"")
    else:
        print(f"  cp {best_model_path} models/chimney_detection/{model_type}_best.pt")

    print(f"\nOr use Python to copy automatically:")
    print(f"  python -c \"import shutil; shutil.copy('{best_model_path}', 'models/chimney_detection/{model_type}_best.pt')\"")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train chimney detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['yolov8', 'yolo11', 'yolo12'],
        default='yolov8',
        help='YOLO model type'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['v1', 'v3', 'v4', 'v6'],
        default='v4',
        help='Dataset version'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of epochs (default: {ChimneyDetectionConfig.EPOCHS})'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=None,
        help=f'Image size (default: {ChimneyDetectionConfig.IMG_SIZE})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {ChimneyDetectionConfig.BATCH_SIZE})'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help=f'Early stopping patience (default: {ChimneyDetectionConfig.PATIENCE})'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='exp1',
        help='Experiment name'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )

    args = parser.parse_args()

    train_chimney_detector(
        model_type=args.model,
        dataset_version=args.dataset,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        patience=args.patience,
        experiment_name=args.name,
        resume=args.resume
    )
