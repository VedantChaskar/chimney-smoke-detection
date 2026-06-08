#!/usr/bin/env python
"""
Ringlemann Scale Classification Training Script

Train models for Ringlemann scale (0-5) smoke density classification.

Usage:
    python scripts/train_ringlemann.py
    python scripts/train_ringlemann.py --model-type mobilenet --epochs 50 --batch-size 32
    python scripts/train_ringlemann.py --model-type resnet18 --use-class-weights
    python scripts/train_ringlemann.py --model-type custom_cnn --lr 0.0005

Supported model types:
    - mobilenet: MobileNetV2 with transfer learning (recommended)
    - resnet18: ResNet18 with transfer learning
    - resnet34: ResNet34 with transfer learning
    - custom_cnn: Custom CNN architecture (no pretrained weights)
"""

import argparse
import json
import platform
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.models.model_loader import build_ringlemann_model
from src.utils.data_utils import load_ringlemann_data, get_ringlemann_class_weights
from src.evaluation.ringlemann_metrics import (
    calculate_ringlemann_mae,
    calculate_tolerance_accuracy,
    plot_ringlemann_confusion_matrix
)
from src.config import (
    RINGLEMANN_DATASET_V1,
    RINGLEMANN_EXPERIMENTS,
    RINGLEMANN_MODELS_DIR,
    RinglemannClassificationConfig,
    DEVICE,
    ensure_dir,
    OUTPUT_PLOTS_DIR
)
from src.utils.tee_logger import TeeLogger
from src.training.training_utils import EarlyStopping


def train_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    """Train for one epoch with MAE tracking"""
    from tqdm import tqdm

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=desc)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    mae = calculate_ringlemann_mae(np.array(all_preds), np.array(all_labels))

    return avg_loss, accuracy, mae


def validate(model, loader, criterion, device, desc="Validation"):
    """Validate the model with comprehensive metrics"""
    from tqdm import tqdm

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.3f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae = calculate_ringlemann_mae(all_preds, all_labels)
    tol_1_acc = calculate_tolerance_accuracy(all_preds, all_labels, tolerance=1)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(6)))

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mae': mae,
        'tolerance_1_accuracy': tol_1_acc * 100,  # As percentage
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, train_acc, val_acc, mae, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'mae': mae
    }

    torch.save(checkpoint, save_path)

    if is_best:
        print(f"  \u2713 New best model! Val Acc: {val_acc:.2f}%, MAE: {mae:.4f}")


def plot_training_history(history: dict, save_path: Path):
    """Plot training history with loss, accuracy, and MAE"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot loss
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2 = axes[1]
    ax2.plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    ax2.plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot MAE
    ax3 = axes[2]
    ax3.plot(history['train_mae'], label='Train MAE', marker='o', markersize=3)
    ax3.plot(history['val_mae'], label='Val MAE', marker='s', markersize=3)
    ax3.axhline(y=0.5, color='g', linestyle='--', label='Good (0.5)')
    ax3.axhline(y=1.0, color='orange', linestyle='--', label='Acceptable (1.0)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE')
    ax3.set_title('Mean Absolute Error (Ringlemann Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_ringlemann_classifier(
    data_dir: Path = None,
    model_type: str = 'mobilenet',
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    experiment_name: str = None,
    use_class_weights: bool = False,
    patience: int = None
):
    """
    Train Ringlemann scale classification model

    Args:
        data_dir: Path to dataset directory
        model_type: Model architecture ('mobilenet', 'resnet18', 'resnet34', 'custom_cnn')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        experiment_name: Name for experiment directory
        use_class_weights: Whether to use weighted loss for class imbalance
        patience: Early stopping patience
    """
    print("=" * 60)
    print("RINGLEMANN SCALE CLASSIFIER TRAINING")
    print("=" * 60)

    # Use config defaults if not specified
    data_dir = data_dir or RINGLEMANN_DATASET_V1
    epochs = epochs or RinglemannClassificationConfig.EPOCHS
    batch_size = batch_size or RinglemannClassificationConfig.BATCH_SIZE
    learning_rate = learning_rate or RinglemannClassificationConfig.LEARNING_RATE
    patience = patience or RinglemannClassificationConfig.PATIENCE
    experiment_name = experiment_name or f"{model_type}_training"

    # Create experiment directory
    save_dir = RINGLEMANN_EXPERIMENTS / experiment_name
    ensure_dir(save_dir)

    device = DEVICE
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nTraining Configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Model type: {model_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Use class weights: {use_class_weights}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Save directory: {save_dir}")

    # Save config
    config = {
        'data_dir': str(data_dir),
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'use_class_weights': use_class_weights,
        'patience': patience
    }
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = load_ringlemann_data(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=True
    )

    # Create model
    print(f"\nInitializing {model_type} model...")
    model = build_ringlemann_model(model_type=model_type, pretrained=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss function
    if use_class_weights:
        print("\nCalculating class weights...")
        class_weights = get_ringlemann_class_weights(data_dir)
        class_weights = class_weights.to(device)
        print(f"  Class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=RinglemannClassificationConfig.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping (monitor validation loss)
    early_stop = EarlyStopping(patience=patience)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_mae': [],
        'val_loss': [],
        'val_acc': [],
        'val_mae': []
    }

    best_val_acc = 0.0
    best_mae = float('inf')

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch} - Train"
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch} - Valid"
        )

        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_mae = val_metrics['mae']
        val_tol1 = val_metrics['tolerance_1_accuracy']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mae'].append(val_mae)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MAE: {train_mae:.4f}")
        print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MAE: {val_mae:.4f}")
        print(f"  Valid - Tolerance +/-1 Acc: {val_tol1:.2f}%")

        # Learning rate scheduling (monitor validation MAE)
        scheduler.step(val_mae)

        # Save checkpoint
        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            best_val_acc = val_acc

        save_checkpoint(
            model, optimizer, epoch, train_acc, val_acc, val_mae,
            save_dir / 'latest_checkpoint.pt',
            is_best=False
        )

        if is_best:
            save_checkpoint(
                model, optimizer, epoch, train_acc, val_acc, val_mae,
                save_dir / 'best_checkpoint.pt',
                is_best=True
            )
            # Save confusion matrix for best model
            plot_ringlemann_confusion_matrix(
                val_metrics['confusion_matrix'],
                save_path=save_dir / 'best_confusion_matrix.png',
                title=f'Validation Confusion Matrix (Epoch {epoch})'
            )

        # Early stopping (monitor validation accuracy for improvement)
        if early_stop(val_acc):
            print(f"\n\u26a0 Early stopping triggered at epoch {epoch}")
            break

    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n\u2713 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"\u2713 Best MAE: {best_mae:.4f}")
    print(f"\u2713 Best model saved to: {save_dir / 'best_checkpoint.pt'}")

    # Test set evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Load best model for testing
    checkpoint = torch.load(save_dir / 'best_checkpoint.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(
        model, test_loader, criterion, device,
        desc="Test Set"
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Tolerance +/-1 Acc: {test_metrics['tolerance_1_accuracy']:.2f}%")

    # Performance assessment
    mae = test_metrics['mae']
    if mae < RinglemannClassificationConfig.GOOD_MAE_THRESHOLD:
        print(f"\n★ EXCELLENT: MAE < {RinglemannClassificationConfig.GOOD_MAE_THRESHOLD}")
    elif mae < RinglemannClassificationConfig.ACCEPTABLE_MAE_THRESHOLD:
        print(f"\n★ GOOD: MAE < {RinglemannClassificationConfig.ACCEPTABLE_MAE_THRESHOLD}")
    else:
        print(f"\n★ NEEDS IMPROVEMENT: MAE >= {RinglemannClassificationConfig.ACCEPTABLE_MAE_THRESHOLD}")

    # Save test confusion matrix
    plot_ringlemann_confusion_matrix(
        test_metrics['confusion_matrix'],
        save_path=save_dir / 'test_confusion_matrix.png',
        title='Test Set Confusion Matrix'
    )
    print(f"\n\u2713 Test confusion matrix saved")

    # Plot training history
    plot_path = save_dir / 'training_history.png'
    plot_training_history(history, plot_path)
    print(f"\u2713 Training plot saved to: {plot_path}")

    # Save final results
    results = {
        'best_val_accuracy': best_val_acc,
        'best_mae': best_mae,
        'test_accuracy': test_metrics['accuracy'],
        'test_mae': test_metrics['mae'],
        'test_tolerance_1_accuracy': test_metrics['tolerance_1_accuracy'],
        'total_epochs': len(history['train_loss'])
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Copy best model to models directory
    ensure_dir(RINGLEMANN_MODELS_DIR)
    best_model_dest = RINGLEMANN_MODELS_DIR / f'{model_type}_best.pt'

    print(f"\nTo copy best model to models directory:")
    if platform.system() == "Windows":
        print(f"  copy \"{save_dir / 'best_checkpoint.pt'}\" \"{best_model_dest}\"")
    else:
        print(f"  cp {save_dir / 'best_checkpoint.pt'} {best_model_dest}")

    return history, results


def main():
    parser = argparse.ArgumentParser(
        description='Train Ringlemann scale smoke density classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to Ringlemann dataset directory'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['mobilenet', 'resnet18', 'resnet34', 'custom_cnn'],
        default='mobilenet',
        help='Model architecture to use'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of epochs (default: {RinglemannClassificationConfig.EPOCHS})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {RinglemannClassificationConfig.BATCH_SIZE})'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help=f'Learning rate (default: {RinglemannClassificationConfig.LEARNING_RATE})'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for experiment directory'
    )

    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        help='Use weighted loss function for class imbalance'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help=f'Early stopping patience (default: {RinglemannClassificationConfig.PATIENCE})'
    )

    args = parser.parse_args()

    # Convert data_dir to Path if provided
    data_dir = Path(args.data_dir) if args.data_dir else None

    with TeeLogger("training", "train_ringlemann"):
        train_ringlemann_classifier(
            data_dir=data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            experiment_name=args.experiment_name,
            use_class_weights=args.use_class_weights,
            patience=args.patience
        )


if __name__ == '__main__':
    main()
