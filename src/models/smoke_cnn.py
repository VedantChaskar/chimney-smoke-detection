"""
Custom CNN Model for Smoke Detection

This module contains the SmokeCNN architecture, a custom convolutional neural
network designed for binary smoke classification (smoke/no-smoke) from chimney
images.

Architecture:
    - 4 Convolutional blocks with batch normalization and max pooling
    - 3 Fully connected layers with dropout for regularization
    - Input: 224x224x3 RGB images
    - Output: 2 classes (no_smoke, smoke)
"""

import torch
import torch.nn as nn
from typing import Tuple


class SmokeCNN(nn.Module):
    """
    Custom CNN for binary smoke detection

    Architecture:
        Conv Block 1: 3 -> 32 channels, 224 -> 112
        Conv Block 2: 32 -> 64 channels, 112 -> 56
        Conv Block 3: 64 -> 128 channels, 56 -> 28
        Conv Block 4: 128 -> 256 channels, 28 -> 14
        FC Layers: 256*14*14 -> 512 -> 128 -> 2

    Each convolutional block consists of:
        - Conv2d with 3x3 kernel
        - Batch Normalization
        - ReLU activation
        - MaxPool2d with 2x2 kernel

    Fully connected layers include dropout (0.5 and 0.3) for regularization.
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize the SmokeCNN model

        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
        """
        super(SmokeCNN, self).__init__()

        # Convolutional Block 1: 3 -> 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )

        # Convolutional Block 2: 32 -> 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )

        # Convolutional Block 3: 64 -> 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )

        # Convolutional Block 4: 128 -> 256 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # Binary classification: smoke/no-smoke
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Get the number of parameters in the model

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def load_smoke_cnn(model_path: str, device: str = 'cpu') -> SmokeCNN:
    """
    Load a trained SmokeCNN model from a checkpoint

    Args:
        model_path: Path to the model checkpoint (.pt file)
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        Loaded SmokeCNN model in evaluation mode
    """
    model = SmokeCNN()

    # Load checkpoint with validation
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


if __name__ == "__main__":
    # Test the model architecture
    model = SmokeCNN()
    total_params, trainable_params = model.get_num_parameters()

    print("SmokeCNN Model Architecture")
    print("=" * 50)
    print(model)
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024**2:.2f} MB (float32)")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
