"""
Smoke Classifier Evaluation Script

Comprehensive evaluation of trained smoke classification models on test data.
Generates detailed metrics, confusion matrix, and confidence analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..models.model_loader import load_smoke_classifier
from ..utils.data_utils import load_smoke_classification_data
from ..utils.visualization import plot_confusion_matrix
from .metrics import (
    calculate_binary_metrics,
    analyze_confidence_distribution,
    generate_classification_report,
    print_evaluation_summary
)
from ..config import (
    SMOKE_DATASET_V1,
    SMOKE_MODEL_MOBILENET,
    SMOKE_MODEL_CNN,
    OUTPUT_PLOTS_DIR,
    DEVICE,
    ensure_dir
)


def evaluate_model(
    model_path: Path,
    model_type: str = 'mobilenet',
    data_dir: Path = None,
    batch_size: int = 32,
    save_plots: bool = True
):
    """
    Evaluate a smoke classification model

    Args:
        model_path: Path to trained model
        model_type: Type of model ('mobilenet' or 'cnn')
        data_dir: Path to dataset directory
        batch_size: Batch size for evaluation
        save_plots: Whether to save visualization plots
    """
    print("=" * 60)
    print("SMOKE CLASSIFIER - DETAILED EVALUATION")
    print("=" * 60)

    # Set data directory
    if data_dir is None:
        data_dir = SMOKE_DATASET_V1

    # Load model
    print(f"\nLoading model from: {model_path.name}")
    print(f"Model type: {model_type}")

    model = load_smoke_classifier(model_type=model_type, model_path=model_path)
    model.eval()

    # Load test dataset
    print(f"\nLoading test data from: {data_dir}")
    _, _, test_loader = load_smoke_classification_data(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=False
    )

    class_names = test_loader.dataset.classes
    print(f"Classes: {class_names}")

    # Run inference
    print("\nRunning inference on test set...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = generate_classification_report(all_labels, all_preds, class_names)
    print(report)

    # Binary metrics
    metrics = calculate_binary_metrics(all_labels, all_preds)

    # Print summary
    print_evaluation_summary(metrics, class_names)

    # Confidence analysis
    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)
    conf_analysis = analyze_confidence_distribution(all_labels, all_preds, all_probs)

    print(f"\nOverall Confidence:")
    print(f"  Mean: {conf_analysis['overall']['mean']:.1%}")
    print(f"  Std: {conf_analysis['overall']['std']:.3f}")

    print(f"\nCorrect Predictions ({conf_analysis['correct_predictions']['count']}):")
    print(f"  Mean Confidence: {conf_analysis['correct_predictions']['mean']:.1%}")

    if conf_analysis['incorrect_predictions']['count'] > 0:
        print(f"\nIncorrect Predictions ({conf_analysis['incorrect_predictions']['count']}):")
        print(f"  Mean Confidence: {conf_analysis['incorrect_predictions']['mean']:.1%}")

    # Per-class analysis
    print("\n" + "=" * 60)
    print("PER-CLASS ANALYSIS")
    print("=" * 60)

    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if class_mask.sum() == 0:
            continue

        class_correct = (all_preds[class_mask] == i).sum()
        class_total = class_mask.sum()
        class_accuracy = class_correct / class_total

        class_confidences = all_probs[class_mask, i]
        avg_confidence = class_confidences.mean()
        min_confidence = class_confidences.min()
        max_confidence = class_confidences.max()

        print(f"\n{class_name}:")
        print(f"  Samples: {class_total}")
        print(f"  Correctly classified: {class_correct}/{class_total} ({class_accuracy:.1%})")
        print(f"  Avg confidence: {avg_confidence:.1%}")
        print(f"  Min confidence: {min_confidence:.1%}")
        print(f"  Max confidence: {max_confidence:.1%}")

    # Save plots if requested
    if save_plots:
        ensure_dir(OUTPUT_PLOTS_DIR)

        # Plot 1: Confusion Matrix
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        cm_path = OUTPUT_PLOTS_DIR / f"confusion_matrix_{model_type}.png"
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names,
            output_path=cm_path,
            title='Confusion Matrix'
        )
        print(f"\n✓ Saved confusion matrix: {cm_path.name}")

        # Plot 2: Confidence Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Correct vs Incorrect confidence
        ax1 = axes[0]
        correct_mask = (all_preds == all_labels)
        confidence_correct = np.max(all_probs[correct_mask], axis=1)
        confidence_wrong = np.max(all_probs[~correct_mask], axis=1) if (~correct_mask).sum() > 0 else []

        ax1.hist(confidence_correct, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
        if len(confidence_wrong) > 0:
            ax1.hist(confidence_wrong, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Average confidence per class
        ax2 = axes[1]
        avg_conf_per_class = []
        for i in range(len(class_names)):
            class_indices = all_labels == i
            if class_indices.sum() > 0:
                avg_conf = all_probs[class_indices, i].mean()
                avg_conf_per_class.append(avg_conf)
            else:
                avg_conf_per_class.append(0)

        bars = ax2.bar(class_names, avg_conf_per_class, color=['skyblue', 'salmon'],
                       edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Average Confidence', fontsize=12)
        ax2.set_title('Average Confidence per Class', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, avg_conf_per_class):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        conf_path = OUTPUT_PLOTS_DIR / f"confidence_analysis_{model_type}.png"
        plt.savefig(conf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confidence analysis: {conf_path.name}")

    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall: {metrics['recall']:.1%}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")

    if save_plots:
        print(f"\nGenerated files:")
        print(f"  - {cm_path.name}")
        print(f"  - {conf_path.name}")

    return metrics, all_preds, all_labels, all_probs


def compare_models():
    """
    Compare MobileNet and Custom CNN models side-by-side
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = {}

    # Evaluate MobileNet
    if SMOKE_MODEL_MOBILENET.exists():
        print("\n1. Evaluating MobileNetV2...")
        try:
            metrics_mn, _, _, _ = evaluate_model(
                model_path=SMOKE_MODEL_MOBILENET,
                model_type='mobilenet',
                save_plots=True
            )
            results['mobilenet'] = metrics_mn
        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Evaluate Custom CNN
    if SMOKE_MODEL_CNN.exists():
        print("\n2. Evaluating Custom CNN...")
        try:
            metrics_cnn, _, _, _ = evaluate_model(
                model_path=SMOKE_MODEL_CNN,
                model_type='cnn',
                save_plots=True
            )
            results['cnn'] = metrics_cnn
        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Print comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Metric':<20} {'MobileNetV2':<15} {'Custom CNN':<15}")
        print("-" * 50)

        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mn_val = results.get('mobilenet', {}).get(metric, 0)
            cnn_val = results.get('cnn', {}).get(metric, 0)

            if metric == 'f1_score':
                print(f"{metric.replace('_', ' ').title():<20} {mn_val:<15.3f} {cnn_val:<15.3f}")
            else:
                print(f"{metric.replace('_', ' ').title():<20} {mn_val:<15.1%} {cnn_val:<15.1%}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate smoke classification model')
    parser.add_argument(
        '--model',
        type=str,
        choices=['mobilenet', 'cnn', 'both'],
        default='both',
        help='Model to evaluate'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving plots'
    )

    args = parser.parse_args()

    if args.model == 'both':
        compare_models()
    else:
        model_path = SMOKE_MODEL_MOBILENET if args.model == 'mobilenet' else SMOKE_MODEL_CNN

        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            exit(1)

        data_dir = Path(args.data_dir) if args.data_dir else None

        evaluate_model(
            model_path=model_path,
            model_type=args.model,
            data_dir=data_dir,
            save_plots=not args.no_plots
        )
