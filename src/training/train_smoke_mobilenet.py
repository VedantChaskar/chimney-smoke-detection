"""
MobileNetV2 Smoke Classifier Training

Train MobileNetV2 model for smoke classification.

Usage:
    python -m src.training.train_smoke_mobilenet
    python -m src.training.train_smoke_mobilenet --epochs 50 --batch-size 64
"""

import argparse
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from pathlib import Path

from ..utils.data_utils import load_smoke_classification_data
from ..config import (
    SMOKE_DATASET_V1,
    SMOKE_EXPERIMENTS,
    SMOKE_MODELS_DIR,
    SmokeClassificationConfig,
    DEVICE,
    ensure_dir,
    OUTPUT_PLOTS_DIR
)
from .training_utils import train_epoch, validate, save_checkpoint, EarlyStopping


def train_mobilenet_classifier(
    data_dir: Path = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    save_dir: Path = None,
    early_stopping_patience: int = 10
):
    """
    Train MobileNetV2 for smoke classification

    Args:
        data_dir: Path to dataset directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Directory to save model checkpoints
        early_stopping_patience: Patience for early stopping
    """
    print("=" * 60)
    print("MOBILENETV2 SMOKE CLASSIFIER TRAINING")
    print("=" * 60)

    # Use config defaults if not specified
    data_dir = data_dir or SMOKE_DATASET_V1
    epochs = epochs or SmokeClassificationConfig.EPOCHS
    batch_size = batch_size or SmokeClassificationConfig.BATCH_SIZE
    learning_rate = learning_rate or SmokeClassificationConfig.LEARNING_RATE
    save_dir = save_dir or (SMOKE_EXPERIMENTS / 'mobilenet_training')

    ensure_dir(save_dir)

    device = DEVICE
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nTraining Configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Save directory: {save_dir}")

    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = load_smoke_classification_data(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=True
    )

    # Create model
    print("\nInitializing MobileNetV2...")
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # Modify classifier for binary classification
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=SmokeClassificationConfig.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    early_stop = EarlyStopping(patience=early_stopping_patience)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch} - Train"
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch} - Valid"
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        save_checkpoint(
            model, optimizer, epoch, train_acc, val_acc,
            save_dir / 'latest_mobilenet.pt',
            is_best=is_best
        )

        if is_best:
            save_checkpoint(
                model, optimizer, epoch, train_acc, val_acc,
                save_dir / 'best_mobilenet.pt',
                is_best=True
            )

        # Early stopping
        if early_stop(val_acc):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break

    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Best model saved to: {save_dir / 'best_mobilenet.pt'}")

    # Test set evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(
        model, test_loader, criterion, device,
        desc="Test Set"
    )
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")

    # Plot training history
    plot_path = OUTPUT_PLOTS_DIR / 'mobilenet_training_history.png'
    plot_training_history(history, plot_path)
    print(f"\n✓ Training plot saved to: {plot_path}")

    # Copy best model to models directory
    best_model_dest = SMOKE_MODELS_DIR / 'mobilenet_best.pt'
    best_model_src = save_dir / 'best_mobilenet.pt'

    print(f"\nTo copy best model to models directory:")
    if platform.system() == "Windows":
        print(f"  copy \"{best_model_src}\" \"{best_model_dest}\"")
    else:
        print(f"  cp {best_model_src} {best_model_dest}")

    print(f"\nOr use Python to copy automatically:")
    print(f"  python -c \"import shutil; shutil.copy('{best_model_src}', '{best_model_dest}')\"")

    return history


def plot_training_history(history: dict, save_path: Path):
    """Plot training history"""
    ensure_dir(save_path.parent)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 smoke classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of epochs (default: {SmokeClassificationConfig.EPOCHS})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {SmokeClassificationConfig.BATCH_SIZE})'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help=f'Learning rate (default: {SmokeClassificationConfig.LEARNING_RATE})'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )

    args = parser.parse_args()

    train_mobilenet_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience
    )
