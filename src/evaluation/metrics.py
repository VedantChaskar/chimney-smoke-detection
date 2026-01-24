"""
Evaluation Metrics

Functions for calculating and analyzing model performance metrics.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from typing import Dict, List, Tuple


def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate binary classification metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with metrics
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        raise ValueError("Binary metrics require exactly 2 classes")

    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'confusion_matrix': cm
    }


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict:
    """
    Calculate multiclass classification metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'confusion_matrix': cm
    }

    # Per-class metrics
    if class_names:
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None)

        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            metrics['per_class'][class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i],
                'support': int(support_per_class[i])
            }

    return metrics


def analyze_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray
) -> Dict:
    """
    Analyze prediction confidence distribution

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        probabilities: Prediction probabilities (N x C array)

    Returns:
        Dictionary with confidence analysis
    """
    # Get confidence for each prediction
    confidences = np.max(probabilities, axis=1)

    # Separate correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    confidence_correct = confidences[correct_mask]
    confidence_wrong = confidences[~correct_mask]

    analysis = {
        'overall': {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        },
        'correct_predictions': {
            'mean': float(np.mean(confidence_correct)) if len(confidence_correct) > 0 else 0,
            'std': float(np.std(confidence_correct)) if len(confidence_correct) > 0 else 0,
            'count': int(len(confidence_correct))
        },
        'incorrect_predictions': {
            'mean': float(np.mean(confidence_wrong)) if len(confidence_wrong) > 0 else 0,
            'std': float(np.std(confidence_wrong)) if len(confidence_wrong) > 0 else 0,
            'count': int(len(confidence_wrong))
        }
    }

    # Per-class average confidence
    n_classes = probabilities.shape[1]
    analysis['per_class_confidence'] = {}

    for i in range(n_classes):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_conf = probabilities[class_mask, i].mean()
            analysis['per_class_confidence'][i] = float(class_conf)
        else:
            analysis['per_class_confidence'][i] = 0.0

    return analysis


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    digits: int = 3
) -> str:
    """
    Generate sklearn classification report

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        digits: Number of digits for formatting

    Returns:
        Classification report as string
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=digits
    )


def print_evaluation_summary(metrics: Dict, class_names: List[str] = None):
    """
    Print a formatted evaluation summary

    Args:
        metrics: Dictionary with metrics
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")

    if 'precision' in metrics:
        # Binary classification
        print(f"\nBinary Classification Metrics:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.2%}")

        print(f"\nConfusion Matrix Breakdown:")
        print(f"  True Negatives: {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Positives: {metrics['true_positives']}")

        if metrics['false_positives'] > 0:
            print(f"\n  ⚠ {metrics['false_positives']} false positive(s)")
        if metrics['false_negatives'] > 0:
            print(f"  ⚠ {metrics['false_negatives']} false negative(s)")

    elif 'per_class' in metrics:
        # Multiclass classification
        print(f"\nMacro-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_macro']:.2%}")
        print(f"  Recall: {metrics['recall_macro']:.2%}")
        print(f"  F1-Score: {metrics['f1_macro']:.3f}")

        if class_names:
            print(f"\nPer-Class Metrics:")
            for class_name in class_names:
                if class_name in metrics['per_class']:
                    cls_metrics = metrics['per_class'][class_name]
                    print(f"\n  {class_name}:")
                    print(f"    Precision: {cls_metrics['precision']:.2%}")
                    print(f"    Recall: {cls_metrics['recall']:.2%}")
                    print(f"    F1-Score: {cls_metrics['f1_score']:.3f}")
                    print(f"    Support: {cls_metrics['support']}")

    # Performance assessment
    accuracy = metrics['accuracy']
    print("\n" + "=" * 60)
    if accuracy == 1.0:
        print("★ EXCELLENT: Perfect accuracy!")
        print("  Note: Verify test set represents real-world conditions")
    elif accuracy >= 0.95:
        print("★ VERY GOOD: Excellent performance!")
    elif accuracy >= 0.85:
        print("★ GOOD: Solid performance")
    elif accuracy >= 0.75:
        print("★ FAIR: Room for improvement")
    else:
        print("★ NEEDS IMPROVEMENT: Consider more training or tuning")
    print("=" * 60)


if __name__ == '__main__':
    # Test metrics functions
    print("Testing Metrics Module")
    print("=" * 50)

    # Dummy binary classification data
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    probs = np.random.rand(10, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize

    print("\n1. Binary Metrics:")
    binary_metrics = calculate_binary_metrics(y_true, y_pred)
    print(f"   Accuracy: {binary_metrics['accuracy']:.2%}")
    print(f"   Precision: {binary_metrics['precision']:.2%}")
    print(f"   Recall: {binary_metrics['recall']:.2%}")
    print(f"   F1-Score: {binary_metrics['f1_score']:.3f}")

    print("\n2. Confidence Analysis:")
    conf_analysis = analyze_confidence_distribution(y_true, y_pred, probs)
    print(f"   Overall Mean: {conf_analysis['overall']['mean']:.3f}")
    print(f"   Correct Mean: {conf_analysis['correct_predictions']['mean']:.3f}")

    print("\n3. Evaluation Summary:")
    print_evaluation_summary(binary_metrics, class_names=['no_smoke', 'smoke'])

    print("\n" + "=" * 50)
