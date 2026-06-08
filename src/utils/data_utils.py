"""
Data Utilities

Functions for loading and processing datasets.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image


# Fixed class-to-index mapping for Ringlemann dataset.
# Ensures consistent label assignment across train/valid/test even when
# some classes are absent from a split (e.g. Ringlemann 1 missing in valid).
_RINGLEMANN_CLASS_TO_IDX = {
    'Ringlemann 0': 0,
    'Ringlemann 1': 1,
    'Ringlemann 2': 2,
    'Ringlemann 3': 3,
    'Ringlemann 4': 4,
    'Ringlemann 5': 5,
}

# Binary mapping: Ringlemann 0 = no_smoke (0), Ringlemann 1-5 = smoke (1)
_RINGLEMANN_BINARY_CLASS_TO_IDX = {
    'Ringlemann 0': 0,  # no smoke
    'Ringlemann 1': 1,  # smoke
    'Ringlemann 2': 1,  # smoke
    'Ringlemann 3': 1,  # smoke
    'Ringlemann 4': 1,  # smoke
    'Ringlemann 5': 1,  # smoke
}


class RinglemannBinaryImageFolder(datasets.ImageFolder):
    """ImageFolder that maps Ringlemann folders to binary labels.

    Ringlemann 0  -> 0 (no_smoke)
    Ringlemann 1-5 -> 1 (smoke)

    Consistent across splits regardless of which Ringlemann classes are present.
    """

    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        class_to_idx = {
            cls: _RINGLEMANN_BINARY_CLASS_TO_IDX[cls]
            for cls in classes
            if cls in _RINGLEMANN_BINARY_CLASS_TO_IDX
        }
        return classes, class_to_idx


class RinglemannImageFolder(datasets.ImageFolder):
    """ImageFolder with a fixed Ringlemann class-to-index mapping.

    Without this, ImageFolder re-indexes from 0 based on only the folders
    present in that split. For example, if Ringlemann 1 is absent from the
    validation split, ImageFolder maps Ringlemann 2 → 1, 3 → 2, etc.,
    which breaks MAE and confusion-matrix computation.
    """

    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        class_to_idx = {
            cls: _RINGLEMANN_CLASS_TO_IDX[cls]
            for cls in classes
            if cls in _RINGLEMANN_CLASS_TO_IDX
        }
        return classes, class_to_idx


def get_smoke_classification_transforms(
    img_size: int = 224,
    augment: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms for smoke classification

    Args:
        img_size: Image size for resizing
        augment: Whether to apply data augmentation for training

    Returns:
        Tuple of (train_transform, test_transform)
    """
    # Test/validation transform
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Training transform with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform


def load_smoke_classification_data(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load smoke classification datasets

    Args:
        data_dir: Path to dataset directory (should contain train/valid/test folders)
        batch_size: Batch size for data loaders
        img_size: Image size for resizing
        num_workers: Number of workers for data loading
        augment: Whether to apply data augmentation

    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    data_dir = Path(data_dir)

    # Only use pin_memory when GPU is available (prevents warnings on CPU-only systems)
    pin_memory = torch.cuda.is_available()

    train_transform, test_transform = get_smoke_classification_transforms(img_size, augment)

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=data_dir / 'train',
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        root=data_dir / 'valid',
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=data_dir / 'test',
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Dataset loaded from: {data_dir}")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Valid: {len(valid_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")

    return train_loader, valid_loader, test_loader


def get_dataset_statistics(data_dir: Union[str, Path]) -> dict:
    """
    Get statistics about a dataset

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with dataset statistics
    """
    data_dir = Path(data_dir)
    stats = {'total': 0, 'splits': {}}

    for split in ['train', 'valid', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            dataset = datasets.ImageFolder(split_dir)
            stats['splits'][split] = {
                'total': len(dataset),
                'classes': {}
            }

            for class_name in dataset.classes:
                class_idx = dataset.class_to_idx[class_name]
                class_count = sum(1 for _, label in dataset.samples if label == class_idx)
                stats['splits'][split]['classes'][class_name] = class_count

            stats['total'] += len(dataset)

    return stats


def calculate_class_weights(data_dir: Union[str, Path]) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets

    Args:
        data_dir: Path to training data directory

    Returns:
        Tensor of class weights
    """
    train_dir = Path(data_dir) / 'train'
    dataset = datasets.ImageFolder(train_dir)

    class_counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1

    # Calculate inverse frequency weights
    total = class_counts.sum()
    class_weights = total / (len(dataset.classes) * class_counts)

    return class_weights


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a list of paths
    """

    def __init__(
        self,
        image_paths: list,
        labels: Optional[list] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize custom dataset

        Args:
            image_paths: List of image file paths
            labels: List of labels (optional)
            transform: Transforms to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image


# ============================================================================
# RINGLEMANN CLASSIFICATION DATA LOADING
# ============================================================================

def get_ringlemann_classification_transforms(
    img_size: int = 224,
    augment: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms for Ringlemann classification

    Args:
        img_size: Image size for resizing
        augment: Whether to apply data augmentation for training

    Returns:
        Tuple of (train_transform, test_transform)
    """
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Test/validation transform
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Training transform with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform


def load_ringlemann_data(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Ringlemann classification datasets

    Expected directory structure:
        data_dir/
        ├── train/
        │   ├── R0/
        │   ├── R1/
        │   ├── R2/
        │   ├── R3/
        │   ├── R4/
        │   └── R5/
        ├── valid/
        │   └── [same structure]
        └── test/
            └── [same structure]

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for data loaders
        img_size: Image size for resizing
        num_workers: Number of workers for data loading
        augment: Whether to apply data augmentation

    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    data_dir = Path(data_dir)

    # Only use pin_memory when GPU is available
    pin_memory = torch.cuda.is_available()

    train_transform, test_transform = get_ringlemann_classification_transforms(img_size, augment)

    # Load datasets using RinglemannImageFolder so that class indices are
    # consistent across splits regardless of which classes are present.
    train_dataset = RinglemannImageFolder(
        root=data_dir / 'train',
        transform=train_transform
    )

    valid_dataset = RinglemannImageFolder(
        root=data_dir / 'valid',
        transform=test_transform
    )

    test_dataset = RinglemannImageFolder(
        root=data_dir / 'test',
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Ringlemann dataset loaded from: {data_dir}")
    print(f"  Train: {len(train_dataset)} images, classes: {train_dataset.class_to_idx}")
    print(f"  Valid: {len(valid_dataset)} images, classes: {valid_dataset.class_to_idx}")
    print(f"  Test:  {len(test_dataset)} images, classes: {test_dataset.class_to_idx}")

    return train_loader, valid_loader, test_loader


def get_ringlemann_class_weights(data_dir: Union[str, Path]) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced Ringlemann dataset

    Uses inverse frequency weighting:
        weights[i] = total_samples / (num_classes * class_i_count)

    Args:
        data_dir: Path to dataset directory (should contain train/ folder)

    Returns:
        Tensor of class weights for use in nn.CrossEntropyLoss(weight=...)
    """
    train_dir = Path(data_dir) / 'train'
    dataset = RinglemannImageFolder(train_dir)

    num_classes = len(_RINGLEMANN_CLASS_TO_IDX)
    class_counts = torch.zeros(num_classes)

    for _, label in dataset.samples:
        class_counts[label] += 1

    # Calculate inverse frequency weights
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)

    # Handle potential division by zero for empty classes
    class_weights = torch.where(
        class_counts > 0,
        class_weights,
        torch.ones_like(class_weights)
    )

    return class_weights


def load_ringlemann_binary_data(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Ringlemann dataset with binary labels (no_smoke vs smoke).

    Ringlemann 0  -> 0 (no_smoke)
    Ringlemann 1-5 -> 1 (smoke)

    Args:
        data_dir: Path to Ringlemann dataset directory
        batch_size: Batch size for data loaders
        img_size: Image size for resizing
        num_workers: Number of workers for data loading
        augment: Whether to apply data augmentation

    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    data_dir = Path(data_dir)
    pin_memory = torch.cuda.is_available()

    train_transform, test_transform = get_ringlemann_classification_transforms(img_size, augment)

    train_dataset = RinglemannBinaryImageFolder(
        root=data_dir / 'train',
        transform=train_transform
    )
    valid_dataset = RinglemannBinaryImageFolder(
        root=data_dir / 'valid',
        transform=test_transform
    )
    test_dataset = RinglemannBinaryImageFolder(
        root=data_dir / 'test',
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # Count binary class distribution from samples
    train_no_smoke = sum(1 for _, lbl in train_dataset.samples if lbl == 0)
    train_smoke = sum(1 for _, lbl in train_dataset.samples if lbl == 1)

    print(f"Ringlemann binary dataset loaded from: {data_dir}")
    print(f"  Train: {len(train_dataset)} images  (no_smoke={train_no_smoke}, smoke={train_smoke})")
    print(f"  Valid: {len(valid_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")

    return train_loader, valid_loader, test_loader


def get_ringlemann_binary_class_weights(data_dir: Union[str, Path]) -> torch.Tensor:
    """
    Calculate binary class weights for the Ringlemann dataset.

    Returns a 2-element tensor [weight_no_smoke, weight_smoke] using
    inverse-frequency weighting.

    Args:
        data_dir: Path to dataset directory (must contain train/ folder)

    Returns:
        Tensor of shape (2,) for use in nn.CrossEntropyLoss(weight=...)
    """
    train_dir = Path(data_dir) / 'train'
    dataset = RinglemannBinaryImageFolder(train_dir)

    class_counts = torch.zeros(2)
    for _, label in dataset.samples:
        class_counts[label] += 1

    total = class_counts.sum()
    class_weights = total / (2 * class_counts)
    class_weights = torch.where(
        class_counts > 0,
        class_weights,
        torch.ones_like(class_weights)
    )
    return class_weights


def get_ringlemann_dataset_statistics(data_dir: Union[str, Path]) -> dict:
    """
    Get comprehensive statistics about a Ringlemann dataset

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with dataset statistics including class distribution
    """
    data_dir = Path(data_dir)
    stats = {
        'total': 0,
        'splits': {},
        'class_names': ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']
    }

    for split in ['train', 'valid', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            dataset = datasets.ImageFolder(split_dir)
            stats['splits'][split] = {
                'total': len(dataset),
                'classes': {}
            }

            for class_name in dataset.classes:
                class_idx = dataset.class_to_idx[class_name]
                class_count = sum(1 for _, label in dataset.samples if label == class_idx)
                stats['splits'][split]['classes'][class_name] = class_count

            stats['total'] += len(dataset)

    # Calculate class imbalance metrics
    if 'train' in stats['splits']:
        train_classes = stats['splits']['train']['classes']
        counts = list(train_classes.values())
        if counts:
            stats['imbalance_ratio'] = max(counts) / max(min(counts), 1)
            stats['most_common_class'] = max(train_classes, key=train_classes.get)
            stats['least_common_class'] = min(train_classes, key=train_classes.get)

    return stats


if __name__ == '__main__':
    # Test data utilities
    from ..config import SMOKE_DATASET_V1, RINGLEMANN_DATASET_V1

    print("Testing Data Utilities")
    print("=" * 50)

    if SMOKE_DATASET_V1.exists():
        # Get dataset statistics
        print("\n1. Smoke Dataset Statistics:")
        stats = get_dataset_statistics(SMOKE_DATASET_V1)
        print(f"   Total images: {stats['total']}")
        for split, split_stats in stats['splits'].items():
            print(f"\n   {split.capitalize()}:")
            print(f"     Total: {split_stats['total']}")
            for class_name, count in split_stats['classes'].items():
                print(f"     - {class_name}: {count}")

        # Test transforms
        print("\n2. Testing Smoke Transforms:")
        train_transform, test_transform = get_smoke_classification_transforms()
        print(f"   ✓ Train transform: {len(train_transform.transforms)} operations")
        print(f"   ✓ Test transform: {len(test_transform.transforms)} operations")

        # Calculate class weights
        print("\n3. Smoke Class Weights:")
        try:
            weights = calculate_class_weights(SMOKE_DATASET_V1)
            print(f"   Weights: {weights}")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    else:
        print(f"\nSmoke dataset not found: {SMOKE_DATASET_V1}")

    print("\n" + "-" * 50)

    if RINGLEMANN_DATASET_V1.exists():
        # Get Ringlemann dataset statistics
        print("\n4. Ringlemann Dataset Statistics:")
        stats = get_ringlemann_dataset_statistics(RINGLEMANN_DATASET_V1)
        print(f"   Total images: {stats['total']}")
        for split, split_stats in stats['splits'].items():
            print(f"\n   {split.capitalize()}:")
            print(f"     Total: {split_stats['total']}")
            for class_name, count in split_stats['classes'].items():
                print(f"     - {class_name}: {count}")

        if 'imbalance_ratio' in stats:
            print(f"\n   Imbalance ratio: {stats['imbalance_ratio']:.2f}")
            print(f"   Most common: {stats['most_common_class']}")
            print(f"   Least common: {stats['least_common_class']}")

        # Test Ringlemann transforms
        print("\n5. Testing Ringlemann Transforms:")
        train_transform, test_transform = get_ringlemann_classification_transforms()
        print(f"   ✓ Train transform: {len(train_transform.transforms)} operations")
        print(f"   ✓ Test transform: {len(test_transform.transforms)} operations")

        # Calculate Ringlemann class weights
        print("\n6. Ringlemann Class Weights:")
        try:
            weights = get_ringlemann_class_weights(RINGLEMANN_DATASET_V1)
            print(f"   Weights: {weights}")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    else:
        print(f"\nRinglemann dataset not found: {RINGLEMANN_DATASET_V1}")

    print("\n" + "=" * 50)
