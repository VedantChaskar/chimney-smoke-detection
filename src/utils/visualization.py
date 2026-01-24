"""
Visualization Utilities

This module provides functions for visualizing detection and classification results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    show_confidence: bool = True,
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes on an image

    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries with 'bbox', 'confidence', optional 'label'
        color: BGR color for boxes
        thickness: Line thickness
        show_confidence: Whether to show confidence scores
        show_labels: Whether to show labels

    Returns:
        Image with drawn bounding boxes
    """
    img_copy = image.copy()

    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label_parts = []
        if show_labels and 'label' in detection:
            label_parts.append(detection['label'])
        if show_confidence and 'confidence' in detection:
            label_parts.append(f"{detection['confidence']:.2f}")

        if label_parts:
            label = ': '.join(label_parts)
            cv2.putText(
                img_copy, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

    return img_copy


def visualize_detection_results(
    image_path: Union[str, Path],
    detections: List[Dict],
    output_path: Union[str, Path] = None,
    title: str = "Detection Results",
    figsize: Tuple[int, int] = (12, 12),
    show: bool = True
):
    """
    Visualize detection results with matplotlib

    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Path to save visualization (optional)
        title: Plot title
        figsize: Figure size (width, height)
        show: Whether to display the plot
    """
    # Read image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw boxes
    img_with_boxes = draw_bounding_boxes(img, detections)
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(img_with_boxes)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_smoke_detection(
    image_path: Union[str, Path],
    chimney_bbox: Optional[Tuple[int, int, int, int]],
    smoke_prediction: str,
    smoke_confidence: float,
    output_path: Union[str, Path] = None,
    show: bool = True
):
    """
    Visualize smoke detection results

    Args:
        image_path: Path to input image
        chimney_bbox: Chimney bounding box (x1, y1, x2, y2) or None
        smoke_prediction: 'smoke' or 'no_smoke'
        smoke_confidence: Confidence score
        output_path: Path to save visualization
        show: Whether to display the plot
    """
    # Read image
    img = cv2.imread(str(image_path))

    # Determine color based on prediction
    color = (0, 0, 255) if smoke_prediction == 'smoke' else (0, 255, 0)

    # Draw bounding box if available
    if chimney_bbox:
        x1, y1, x2, y2 = chimney_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{smoke_prediction}: {smoke_confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        # Full image classification
        label = f"Full Image: {smoke_prediction} ({smoke_confidence:.2f})"
        cv2.putText(img, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    plt.title(f"Smoke Detection: {smoke_prediction.upper()}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
        print(f"Saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: Dict,
    metrics: List[str] = None,
    output_path: Union[str, Path] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot training history metrics

    Args:
        history: Dictionary with metric names as keys and lists of values
        metrics: List of metric names to plot (if None, plots all)
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
    """
    if metrics is None:
        metrics = list(history.keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
        print(f"Saved plot to: {output_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False
):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
        print(f"Saved confusion matrix to: {output_path}")

    plt.show()


if __name__ == '__main__':
    # Test visualization functions
    from ..config import PROJECT_ROOT

    print("Testing Visualization Utilities")
    print("=" * 50)

    test_image = PROJECT_ROOT / "assets" / "test_image.jpg"

    if test_image.exists():
        # Test detection visualization
        dummy_detections = [
            {'bbox': (100, 100, 300, 400), 'confidence': 0.85, 'label': 'chimney'},
            {'bbox': (400, 150, 600, 450), 'confidence': 0.72, 'label': 'chimney'}
        ]

        print("\n1. Testing detection visualization...")
        visualize_detection_results(
            test_image,
            dummy_detections,
            title="Test Detection",
            show=False
        )
        print("   ✓ Detection visualization works")

        print("\n2. Testing smoke detection visualization...")
        visualize_smoke_detection(
            test_image,
            chimney_bbox=(100, 100, 300, 400),
            smoke_prediction='smoke',
            smoke_confidence=0.89,
            show=False
        )
        print("   ✓ Smoke detection visualization works")

    else:
        print(f"Test image not found: {test_image}")

    print("\n" + "=" * 50)
