"""
Training Utilities

Common functions used across different training scripts.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple


def train_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    desc: str = "Training"
) -> Tuple[float, float]:
    """
    Train for one epoch

    Args:
        model: PyTorch model
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        desc: Description for progress bar

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    desc: str = "Validation"
) -> Tuple[float, float]:
    """
    Validate the model

    Args:
        model: PyTorch model
        loader: Data loader
        criterion: Loss function
        device: Device to validate on
        desc: Description for progress bar

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.3f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    train_acc: float,
    val_acc: float,
    save_path: str,
    is_best: bool = False
):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_acc: Training accuracy
        val_acc: Validation accuracy
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc
    }

    torch.save(checkpoint, save_path)

    if is_best:
        print(f"  ✓ New best model! Validation accuracy: {val_acc:.2f}%")


class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc: float) -> bool:
        """
        Check if training should stop

        Args:
            val_acc: Current validation accuracy

        Returns:
            True if training should stop, False otherwise
        """
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == '__main__':
    # Test utilities
    print("Testing Training Utilities")
    print("=" * 50)

    # Create a dummy model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )

    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")

    # Test early stopping
    print(f"\nTesting Early Stopping (patience=3):")
    early_stop = EarlyStopping(patience=3)

    val_accs = [0.7, 0.72, 0.73, 0.72, 0.71, 0.70, 0.69]
    for epoch, acc in enumerate(val_accs, 1):
        should_stop = early_stop(acc)
        print(f"  Epoch {epoch}: acc={acc:.2f}, counter={early_stop.counter}, stop={should_stop}")
        if should_stop:
            print(f"  → Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 50)
