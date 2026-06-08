#!/usr/bin/env python
"""
Binary Smoke Classifier Training on Ringlemann Dataset

Trains a binary smoke/no-smoke classifier using the Ringlemann dataset:
  - Ringlemann 0  -> no_smoke (class 0)
  - Ringlemann 1-5 -> smoke   (class 1)

Usage:
    python scripts/train_binary_ringlemann.py
    python scripts/train_binary_ringlemann.py --model-type mobilenet --epochs 50
    python scripts/train_binary_ringlemann.py --model-type resnet18 --use-class-weights
    python scripts/train_binary_ringlemann.py --lr 0.0005 --experiment-name my_run
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.models.model_loader import build_ringlemann_model
from src.utils.data_utils import load_ringlemann_binary_data, get_ringlemann_binary_class_weights
from src.config import (
    RINGLEMANN_DATASET_V1,
    RINGLEMANN_EXPERIMENTS,
    RINGLEMANN_BINARY_MODELS_DIR,
    RinglemannClassificationConfig,
    DEVICE,
    ensure_dir,
)
from src.utils.tee_logger import TeeLogger
from src.training.training_utils import EarlyStopping

CLASS_NAMES = ['no_smoke', 'smoke']


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    from tqdm import tqdm

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, desc="Validation"):
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

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    ax2.plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_acc, val_acc, save_path, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, save_path)
    if is_best:
        print(f"  \u2713 New best model! Val Acc: {val_acc:.2f}%")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_binary_ringlemann_classifier(
    data_dir: Path = None,
    model_type: str = 'mobilenet',
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    experiment_name: str = None,
    use_class_weights: bool = False,
    patience: int = None,
):
    print("=" * 60)
    print("BINARY SMOKE CLASSIFIER TRAINING (Ringlemann Dataset)")
    print("  Ringlemann 0   -> no_smoke (class 0)")
    print("  Ringlemann 1-5 -> smoke    (class 1)")
    print("=" * 60)

    data_dir = data_dir or RINGLEMANN_DATASET_V1
    epochs = epochs or RinglemannClassificationConfig.EPOCHS
    batch_size = batch_size or RinglemannClassificationConfig.BATCH_SIZE
    learning_rate = learning_rate or RinglemannClassificationConfig.LEARNING_RATE
    patience = patience or RinglemannClassificationConfig.PATIENCE
    experiment_name = experiment_name or f"binary_{model_type}_training"

    save_dir = RINGLEMANN_EXPERIMENTS / experiment_name
    ensure_dir(save_dir)

    device = DEVICE
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nTraining Configuration:")
    print(f"  Dataset:           {data_dir}")
    print(f"  Model type:        {model_type}")
    print(f"  Epochs:            {epochs}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Learning rate:     {learning_rate}")
    print(f"  Use class weights: {use_class_weights}")
    print(f"  Patience:          {patience}")
    print(f"  Save directory:    {save_dir}")

    config = {
        'task': 'binary_smoke_classification',
        'data_dir': str(data_dir),
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'use_class_weights': use_class_weights,
        'patience': patience,
        'class_mapping': {'Ringlemann 0': 'no_smoke(0)', 'Ringlemann 1-5': 'smoke(1)'},
    }
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = load_ringlemann_binary_data(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=True,
    )

    # Build model (2 output classes)
    print(f"\nInitializing {model_type} model (num_classes=2)...")
    model = build_ringlemann_model(model_type=model_type, pretrained=True, num_classes=2)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss function
    if use_class_weights:
        print("\nCalculating binary class weights...")
        class_weights = get_ringlemann_binary_class_weights(data_dir).to(device)
        print(f"  no_smoke weight: {class_weights[0]:.4f}")
        print(f"  smoke    weight: {class_weights[1]:.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=RinglemannClassificationConfig.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stop = EarlyStopping(patience=patience)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [],   'val_acc': [],
    }

    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch} - Train"
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch} - Valid"
        )

        val_loss = val_metrics['loss']
        val_acc  = val_metrics['accuracy']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        save_checkpoint(model, optimizer, epoch, train_acc, val_acc,
                        save_dir / 'latest_checkpoint.pt')
        if is_best:
            save_checkpoint(model, optimizer, epoch, train_acc, val_acc,
                            save_dir / 'best_checkpoint.pt', is_best=True)
            plot_confusion_matrix(
                val_metrics['confusion_matrix'],
                save_path=save_dir / 'best_confusion_matrix.png',
                title=f'Validation Confusion Matrix (Epoch {epoch})'
            )

        if early_stop(val_acc):
            print(f"\n\u26a0 Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n\u2713 Best validation accuracy: {best_val_acc:.2f}%")

    # Test set evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(save_dir / 'best_checkpoint.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, criterion, device, desc="Test Set")

    print(f"\nTest Results:")
    print(f"  Loss:     {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        test_metrics['labels'], test_metrics['predictions'],
        target_names=CLASS_NAMES, digits=3
    ))

    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        save_path=save_dir / 'test_confusion_matrix.png',
        title='Test Set Confusion Matrix'
    )
    print(f"\n\u2713 Test confusion matrix saved")

    plot_path = save_dir / 'training_history.png'
    plot_training_history(history, plot_path)
    print(f"\u2713 Training plot saved: {plot_path}")

    results = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'total_epochs': len(history['train_loss']),
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Copy best model to models directory
    ensure_dir(RINGLEMANN_BINARY_MODELS_DIR)
    best_model_dest = RINGLEMANN_BINARY_MODELS_DIR / f'{model_type}_best.pt'
    print(f"\nTo copy best model to models directory:")
    print(f"  cp {save_dir / 'best_checkpoint.pt'} {best_model_dest}")

    return history, results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train binary smoke/no-smoke classifier on Ringlemann dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Ringlemann dataset directory')
    parser.add_argument('--model-type', type=str,
                        choices=['mobilenet', 'resnet18', 'resnet34', 'custom_cnn'],
                        default='mobilenet',
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of epochs (default: {RinglemannClassificationConfig.EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size (default: {RinglemannClassificationConfig.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Learning rate (default: {RinglemannClassificationConfig.LEARNING_RATE})')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for experiment directory')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use weighted loss for class imbalance')
    parser.add_argument('--patience', type=int, default=None,
                        help=f'Early stopping patience (default: {RinglemannClassificationConfig.PATIENCE})')

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None

    with TeeLogger("training", "train_binary_ringlemann"):
        train_binary_ringlemann_classifier(
            data_dir=data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            experiment_name=args.experiment_name,
            use_class_weights=args.use_class_weights,
            patience=args.patience,
        )


if __name__ == '__main__':
    main()
