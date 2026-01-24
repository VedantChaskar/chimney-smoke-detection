"""
Data Utilities

Functions for loading and processing datasets.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image


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


if __name__ == '__main__':
    # Test data utilities
    from ..config import SMOKE_DATASET_V1

    print("Testing Data Utilities")
    print("=" * 50)

    if SMOKE_DATASET_V1.exists():
        # Get dataset statistics
        print("\n1. Dataset Statistics:")
        stats = get_dataset_statistics(SMOKE_DATASET_V1)
        print(f"   Total images: {stats['total']}")
        for split, split_stats in stats['splits'].items():
            print(f"\n   {split.capitalize()}:")
            print(f"     Total: {split_stats['total']}")
            for class_name, count in split_stats['classes'].items():
                print(f"     - {class_name}: {count}")

        # Test transforms
        print("\n2. Testing Transforms:")
        train_transform, test_transform = get_smoke_classification_transforms()
        print(f"   ✓ Train transform: {len(train_transform.transforms)} operations")
        print(f"   ✓ Test transform: {len(test_transform.transforms)} operations")

        # Calculate class weights
        print("\n3. Class Weights:")
        try:
            weights = calculate_class_weights(SMOKE_DATASET_V1)
            print(f"   Weights: {weights}")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    else:
        print(f"\nDataset not found: {SMOKE_DATASET_V1}")

    print("\n" + "=" * 50)
