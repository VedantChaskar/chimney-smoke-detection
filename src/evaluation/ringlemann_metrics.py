"""
Ringlemann Scale Evaluation Metrics

This module provides metrics specific to Ringlemann scale smoke density classification.
Per project specification: "Ringlemann scale prediction will be evaluated using
mean absolute error compared to ground truth ratings."

Ringlemann Scale:
    - R0: No visible smoke (clear)
    - R1: Light grey smoke (20% opacity)
    - R2: Medium grey smoke (40% opacity)
    - R3: Dark grey smoke (60% opacity)
    - R4: Very dark smoke (80% opacity)
    - R5: Black smoke (100% opacity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path


# Default Ringlemann class names
RINGLEMANN_CLASS_NAMES = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']


def calculate_ringlemann_mae(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Mean Absolute Error for Ringlemann predictions

    Per project spec: "Ringlemann scale prediction will be evaluated
    using mean absolute error compared to ground truth ratings"

    Args:
        predictions: numpy array of predicted classes (0-5)
        labels: numpy array of true classes (0-5)

    Returns:
        mae: float, average absolute difference between predicted and true ratings
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return np.mean(np.abs(predictions - labels))


def calculate_tolerance_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    tolerance: int = 1
) -> float:
    """
    Accuracy allowing for tolerance (e.g., +/-1 Ringlemann level)

    Useful metric since predicting R2 when true is R3 is less severe
    than predicting R0 when true is R5.

    Args:
        predictions: numpy array of predicted classes (0-5)
        labels: numpy array of true classes (0-5)
        tolerance: Maximum allowed difference for a prediction to be "correct"

    Returns:
        tolerance_accuracy: float, percentage of predictions within tolerance
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    within_tolerance = np.abs(predictions - labels) <= tolerance
    return np.mean(within_tolerance)


def calculate_weighted_mae(
    predictions: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Weighted MAE where errors at higher Ringlemann levels matter more

    Higher Ringlemann levels (R4-R5) are more critical for air quality compliance,
    so errors at these levels should be penalized more heavily.

    Args:
        predictions: numpy array of predicted classes (0-5)
        labels: numpy array of true classes (0-5)
        weights: Optional weights per class. If None, uses default weighting
                 that emphasizes higher Ringlemann levels.

    Returns:
        weighted_mae: float
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if weights is None:
        # Default weighting: higher levels are more important
        # R0: 1.0, R1: 1.2, R2: 1.4, R3: 1.6, R4: 1.8, R5: 2.0
        default_weights = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        sample_weights = default_weights[labels]
    else:
        sample_weights = weights[labels]

    absolute_errors = np.abs(predictions - labels)
    weighted_errors = absolute_errors * sample_weights

    return np.sum(weighted_errors) / np.sum(sample_weights)


def calculate_ringlemann_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate comprehensive metrics for Ringlemann classification

    Args:
        predictions: numpy array of predicted classes (0-5)
        labels: numpy array of true classes (0-5)
        class_names: List of class names (default: ['R0', 'R1', ..., 'R5'])

    Returns:
        Dictionary with all metrics
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if class_names is None:
        class_names = RINGLEMANN_CLASS_NAMES

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    mae = calculate_ringlemann_mae(predictions, labels)

    # Tolerance accuracies
    tolerance_1 = calculate_tolerance_accuracy(predictions, labels, tolerance=1)
    tolerance_2 = calculate_tolerance_accuracy(predictions, labels, tolerance=2)

    # Weighted MAE
    weighted_mae = calculate_weighted_mae(predictions, labels)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )

    # Macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)

    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        if class_mask.sum() > 0:
            per_class_accuracy[class_name] = (predictions[class_mask] == i).mean()
        else:
            per_class_accuracy[class_name] = 0.0

    metrics = {
        'accuracy': accuracy,
        'mae': mae,
        'weighted_mae': weighted_mae,
        'tolerance_1_accuracy': tolerance_1,
        'tolerance_2_accuracy': tolerance_2,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'per_class': {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i]),
                'accuracy': per_class_accuracy[class_names[i]]
            }
            for i in range(len(class_names))
        }
    }

    return metrics


def generate_ringlemann_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive text report for Ringlemann classification

    Args:
        predictions: numpy array of predicted classes (0-5)
        labels: numpy array of true classes (0-5)
        class_names: List of class names

    Returns:
        Formatted text report
    """
    if class_names is None:
        class_names = RINGLEMANN_CLASS_NAMES

    metrics = calculate_ringlemann_metrics(predictions, labels, class_names)

    lines = []
    lines.append("=" * 60)
    lines.append("RINGLEMANN SCALE CLASSIFICATION REPORT")
    lines.append("=" * 60)

    lines.append("\n--- Overall Metrics ---")
    lines.append(f"Accuracy:              {metrics['accuracy']:.2%}")
    lines.append(f"MAE:                   {metrics['mae']:.4f}")
    lines.append(f"Weighted MAE:          {metrics['weighted_mae']:.4f}")
    lines.append(f"Tolerance +/-1 Acc:    {metrics['tolerance_1_accuracy']:.2%}")
    lines.append(f"Tolerance +/-2 Acc:    {metrics['tolerance_2_accuracy']:.2%}")

    lines.append("\n--- Macro-Averaged Metrics ---")
    lines.append(f"Precision:             {metrics['precision_macro']:.2%}")
    lines.append(f"Recall:                {metrics['recall_macro']:.2%}")
    lines.append(f"F1-Score:              {metrics['f1_macro']:.4f}")

    lines.append("\n--- Per-Class Metrics ---")
    lines.append(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10} {'Accuracy':<10}")
    lines.append("-" * 64)

    for class_name in class_names:
        cls = metrics['per_class'][class_name]
        lines.append(
            f"{class_name:<8} {cls['precision']:<12.2%} {cls['recall']:<12.2%} "
            f"{cls['f1_score']:<12.4f} {cls['support']:<10} {cls['accuracy']:<10.2%}"
        )

    lines.append("\n--- Confusion Matrix ---")
    cm = metrics['confusion_matrix']
    # Header
    header = "True\\Pred " + " ".join(f"{name:>5}" for name in class_names)
    lines.append(header)
    lines.append("-" * len(header))

    for i, class_name in enumerate(class_names):
        row = f"{class_name:<10}" + " ".join(f"{cm[i, j]:>5}" for j in range(len(class_names)))
        lines.append(row)

    # Performance assessment
    lines.append("\n" + "=" * 60)
    lines.append("PERFORMANCE ASSESSMENT")
    lines.append("=" * 60)

    mae = metrics['mae']
    if mae < 0.5:
        lines.append(f"★ EXCELLENT: MAE = {mae:.3f} (target: < 0.5)")
    elif mae < 1.0:
        lines.append(f"★ GOOD: MAE = {mae:.3f} (target: < 1.0)")
    elif mae < 1.5:
        lines.append(f"★ FAIR: MAE = {mae:.3f} (needs improvement)")
    else:
        lines.append(f"★ NEEDS IMPROVEMENT: MAE = {mae:.3f}")

    lines.append("=" * 60)

    return "\n".join(lines)


def plot_ringlemann_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Ringlemann Classification Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False
) -> None:
    """
    Visualize 6x6 confusion matrix for Ringlemann classes

    Args:
        cm: Confusion matrix (6x6 numpy array)
        class_names: List of class names
        save_path: Path to save the figure (optional)
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    if class_names is None:
        class_names = RINGLEMANN_CLASS_NAMES

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")

    plt.close()


def plot_ringlemann_error_distribution(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Ringlemann Prediction Error Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot distribution of prediction errors (predicted - true)

    Args:
        predictions: numpy array of predicted classes (0-5)
        labels: numpy array of true classes (0-5)
        save_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    errors = predictions - labels

    plt.figure(figsize=figsize)

    # Error histogram
    bins = np.arange(-5.5, 6.5, 1)
    counts, _, patches = plt.hist(errors, bins=bins, edgecolor='black', alpha=0.7)

    # Color bars based on error magnitude
    for i, (patch, error) in enumerate(zip(patches, range(-5, 6))):
        if error == 0:
            patch.set_facecolor('green')
        elif abs(error) == 1:
            patch.set_facecolor('yellow')
        elif abs(error) == 2:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')

    plt.xlabel('Prediction Error (Predicted - True)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(-5, 6))
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics annotation
    mae = np.mean(np.abs(errors))
    plt.text(0.98, 0.95, f'MAE: {mae:.3f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution to: {save_path}")

    plt.close()


def analyze_prediction_errors(
    predictions: np.ndarray,
    labels: np.ndarray,
    image_paths: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze which images were misclassified and by how much

    Args:
        predictions: numpy array of predicted classes (0-5)
        labels: numpy array of true classes (0-5)
        image_paths: Optional list of image paths
        class_names: List of class names

    Returns:
        DataFrame with: image_path, true_class, pred_class, error_magnitude
        Sorted by error magnitude (worst first)
    """
    if class_names is None:
        class_names = RINGLEMANN_CLASS_NAMES

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    errors = np.abs(predictions - labels)

    data = {
        'index': list(range(len(predictions))),
        'true_label': labels,
        'true_class': [class_names[l] for l in labels],
        'pred_label': predictions,
        'pred_class': [class_names[p] for p in predictions],
        'error_magnitude': errors,
        'is_correct': predictions == labels
    }

    if image_paths is not None:
        data['image_path'] = image_paths

    df = pd.DataFrame(data)

    # Sort by error magnitude (descending) then by index
    df = df.sort_values(['error_magnitude', 'index'], ascending=[False, True])

    return df


def print_ringlemann_evaluation_summary(metrics: Dict, class_names: Optional[List[str]] = None):
    """
    Print a formatted evaluation summary

    Args:
        metrics: Dictionary with metrics from calculate_ringlemann_metrics()
        class_names: List of class names
    """
    if class_names is None:
        class_names = RINGLEMANN_CLASS_NAMES

    print("\n" + "=" * 60)
    print("RINGLEMANN CLASSIFICATION SUMMARY")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"Weighted MAE: {metrics['weighted_mae']:.4f}")

    print(f"\nTolerance Accuracies:")
    print(f"  Within +/-1 level: {metrics['tolerance_1_accuracy']:.2%}")
    print(f"  Within +/-2 levels: {metrics['tolerance_2_accuracy']:.2%}")

    print(f"\nMacro-Averaged Metrics:")
    print(f"  Precision: {metrics['precision_macro']:.2%}")
    print(f"  Recall: {metrics['recall_macro']:.2%}")
    print(f"  F1-Score: {metrics['f1_macro']:.4f}")

    print(f"\nPer-Class Accuracy:")
    for class_name in class_names:
        cls = metrics['per_class'][class_name]
        print(f"  {class_name}: {cls['accuracy']:.2%} ({cls['support']} samples)")

    # Performance assessment
    print("\n" + "=" * 60)
    mae = metrics['mae']
    if mae < 0.5:
        print("★ EXCELLENT: MAE < 0.5 (meets target)")
    elif mae < 1.0:
        print("★ GOOD: MAE < 1.0 (acceptable)")
    elif mae < 1.5:
        print("★ FAIR: MAE < 1.5 (room for improvement)")
    else:
        print("★ NEEDS IMPROVEMENT: MAE >= 1.5")
    print("=" * 60)


if __name__ == '__main__':
    # Test metrics module
    print("Testing Ringlemann Metrics Module")
    print("=" * 50)

    # Create dummy data
    np.random.seed(42)
    n_samples = 100

    # Generate ground truth labels (0-5)
    labels = np.random.randint(0, 6, n_samples)

    # Generate predictions with some noise
    predictions = labels.copy()
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=30, replace=False)
    predictions[noise_idx] = np.clip(predictions[noise_idx] + np.random.randint(-2, 3, size=30), 0, 5)

    print("\n1. Testing MAE:")
    mae = calculate_ringlemann_mae(predictions, labels)
    print(f"   MAE: {mae:.4f}")

    print("\n2. Testing Tolerance Accuracy:")
    tol_1 = calculate_tolerance_accuracy(predictions, labels, tolerance=1)
    tol_2 = calculate_tolerance_accuracy(predictions, labels, tolerance=2)
    print(f"   Within +/-1: {tol_1:.2%}")
    print(f"   Within +/-2: {tol_2:.2%}")

    print("\n3. Testing Weighted MAE:")
    wmae = calculate_weighted_mae(predictions, labels)
    print(f"   Weighted MAE: {wmae:.4f}")

    print("\n4. Testing Full Metrics:")
    metrics = calculate_ringlemann_metrics(predictions, labels)
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   F1 Macro: {metrics['f1_macro']:.4f}")

    print("\n5. Testing Report Generation:")
    report = generate_ringlemann_report(predictions, labels)
    print(report)

    print("\n6. Testing Error Analysis:")
    error_df = analyze_prediction_errors(predictions, labels)
    print(f"   Total samples: {len(error_df)}")
    print(f"   Correct predictions: {error_df['is_correct'].sum()}")
    print(f"   Worst errors (top 5):")
    print(error_df.head(5)[['true_class', 'pred_class', 'error_magnitude']])

    print("\n" + "=" * 50)
