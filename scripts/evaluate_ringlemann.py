#!/usr/bin/env python
"""
Ringlemann Scale Classification Evaluation Script

Comprehensive evaluation of trained Ringlemann classification models on test data.
Generates detailed metrics, confusion matrices, error analysis, and visualizations.

Usage:
    python scripts/evaluate_ringlemann.py --model-path models/ringlemann_classification/mobilenet_best.pt
    python scripts/evaluate_ringlemann.py --model-path path/to/model.pt --model-type resnet18
    python scripts/evaluate_ringlemann.py --model-type mobilenet --data-dir data/ringlemann_classification/v1
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.models.model_loader import load_ringlemann_classifier
from src.utils.data_utils import load_ringlemann_data
from src.evaluation.ringlemann_metrics import (
    calculate_ringlemann_metrics,
    generate_ringlemann_report,
    plot_ringlemann_confusion_matrix,
    plot_ringlemann_error_distribution,
    analyze_prediction_errors,
    print_ringlemann_evaluation_summary,
    RINGLEMANN_CLASS_NAMES
)
from src.config import (
    RINGLEMANN_DATASET_V1,
    RINGLEMANN_MODEL_MOBILENET,
    RINGLEMANN_MODEL_RESNET18,
    RINGLEMANN_MODEL_CNN,
    OUTPUT_PLOTS_DIR,
    DEVICE,
    ensure_dir
)


def evaluate_ringlemann_model(
    model_path: Path,
    model_type: str = 'mobilenet',
    data_dir: Path = None,
    batch_size: int = 32,
    save_plots: bool = True,
    output_dir: Path = None
):
    """
    Complete evaluation pipeline for Ringlemann classification model

    Args:
        model_path: Path to trained model
        model_type: Type of model ('mobilenet', 'resnet18', 'resnet34', 'custom_cnn')
        data_dir: Path to dataset directory
        batch_size: Batch size for evaluation
        save_plots: Whether to save visualization plots
        output_dir: Directory to save outputs
    """
    print("=" * 60)
    print("RINGLEMANN CLASSIFIER - DETAILED EVALUATION")
    print("=" * 60)

    # Set data directory
    if data_dir is None:
        data_dir = RINGLEMANN_DATASET_V1

    # Set output directory
    if output_dir is None:
        output_dir = OUTPUT_PLOTS_DIR / "ringlemann_evaluation"
    ensure_dir(output_dir)

    # Load model
    print(f"\nLoading model from: {model_path.name}")
    print(f"Model type: {model_type}")

    model = load_ringlemann_classifier(model_type=model_type, model_path=model_path)
    model.eval()

    # Load test dataset
    print(f"\nLoading test data from: {data_dir}")
    _, _, test_loader = load_ringlemann_data(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=False
    )

    class_names = RINGLEMANN_CLASS_NAMES
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

    # Calculate comprehensive metrics
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    metrics = calculate_ringlemann_metrics(all_preds, all_labels, class_names)

    # Print detailed report
    report = generate_ringlemann_report(all_preds, all_labels, class_names)
    print(report)

    # Confidence analysis
    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    # Overall confidence
    confidences = np.max(all_probs, axis=1)
    correct_mask = (all_preds == all_labels)

    print(f"\nOverall Confidence:")
    print(f"  Mean: {np.mean(confidences):.1%}")
    print(f"  Std: {np.std(confidences):.3f}")
    print(f"  Min: {np.min(confidences):.1%}")
    print(f"  Max: {np.max(confidences):.1%}")

    print(f"\nCorrect Predictions ({correct_mask.sum()}):")
    print(f"  Mean Confidence: {np.mean(confidences[correct_mask]):.1%}")

    if (~correct_mask).sum() > 0:
        print(f"\nIncorrect Predictions ({(~correct_mask).sum()}):")
        print(f"  Mean Confidence: {np.mean(confidences[~correct_mask]):.1%}")

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

    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    error_df = analyze_prediction_errors(all_preds, all_labels, class_names=class_names)
    errors_only = error_df[~error_df['is_correct']]

    print(f"\nTotal samples: {len(error_df)}")
    print(f"Correct predictions: {error_df['is_correct'].sum()}")
    print(f"Incorrect predictions: {len(errors_only)}")

    if len(errors_only) > 0:
        print(f"\nError magnitude distribution:")
        for mag in range(1, 6):
            count = (errors_only['error_magnitude'] == mag).sum()
            if count > 0:
                print(f"  Off by {mag}: {count} samples")

        print(f"\nWorst misclassifications (top 10):")
        print(errors_only.head(10)[['true_class', 'pred_class', 'error_magnitude']].to_string())

    # Save outputs
    if save_plots:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # 1. Confusion Matrix (raw counts)
        cm_path = output_dir / f"confusion_matrix_{model_type}.png"
        plot_ringlemann_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=class_names,
            save_path=cm_path,
            title=f'Ringlemann Confusion Matrix ({model_type})'
        )
        print(f"\n\u2713 Saved confusion matrix: {cm_path.name}")

        # 2. Normalized Confusion Matrix
        cm_norm_path = output_dir / f"confusion_matrix_normalized_{model_type}.png"
        plot_ringlemann_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=class_names,
            save_path=cm_norm_path,
            title=f'Normalized Ringlemann Confusion Matrix ({model_type})',
            normalize=True
        )
        print(f"\u2713 Saved normalized confusion matrix: {cm_norm_path.name}")

        # 3. Error Distribution
        error_path = output_dir / f"error_distribution_{model_type}.png"
        plot_ringlemann_error_distribution(
            all_preds, all_labels,
            save_path=error_path,
            title=f'Prediction Error Distribution ({model_type})'
        )
        print(f"\u2713 Saved error distribution: {error_path.name}")

        # 4. Confidence Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Correct vs Incorrect confidence
        ax1 = axes[0]
        confidence_correct = confidences[correct_mask]
        confidence_wrong = confidences[~correct_mask] if (~correct_mask).sum() > 0 else []

        ax1.hist(confidence_correct, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
        if len(confidence_wrong) > 0:
            ax1.hist(confidence_wrong, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Per-class confidence
        ax2 = axes[1]
        avg_conf_per_class = []
        for i in range(len(class_names)):
            class_mask = all_labels == i
            if class_mask.sum() > 0:
                avg_conf = all_probs[class_mask, i].mean()
                avg_conf_per_class.append(avg_conf)
            else:
                avg_conf_per_class.append(0)

        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(class_names)))
        bars = ax2.bar(class_names, avg_conf_per_class, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Average Confidence', fontsize=12)
        ax2.set_title('Average Confidence per Class', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, avg_conf_per_class):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        conf_path = output_dir / f"confidence_analysis_{model_type}.png"
        plt.savefig(conf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\u2713 Saved confidence analysis: {conf_path.name}")

        # 5. Per-class accuracy bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        per_class_acc = [metrics['per_class'][c]['accuracy'] * 100 for c in class_names]
        bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor='black', linewidth=1.5)

        ax.axhline(y=metrics['accuracy'] * 100, color='red', linestyle='--', linewidth=2, label='Overall Accuracy')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Ringlemann Class', fontsize=12)
        ax.set_title(f'Per-Class Accuracy ({model_type})', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, per_class_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        acc_path = output_dir / f"per_class_accuracy_{model_type}.png"
        plt.savefig(acc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\u2713 Saved per-class accuracy: {acc_path.name}")

        # Save error analysis CSV
        error_csv_path = output_dir / f"error_analysis_{model_type}.csv"
        error_df.to_csv(error_csv_path, index=False)
        print(f"\u2713 Saved error analysis: {error_csv_path.name}")

        # Save results JSON
        results_json = {
            'model_type': model_type,
            'model_path': str(model_path),
            'accuracy': float(metrics['accuracy']),
            'mae': float(metrics['mae']),
            'weighted_mae': float(metrics['weighted_mae']),
            'tolerance_1_accuracy': float(metrics['tolerance_1_accuracy']),
            'tolerance_2_accuracy': float(metrics['tolerance_2_accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'per_class': {
                k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                    for kk, vv in v.items()}
                for k, v in metrics['per_class'].items()
            }
        }
        results_path = output_dir / f"results_{model_type}.json"
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\u2713 Saved results: {results_path.name}")

    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Tolerance +/-1: {metrics['tolerance_1_accuracy']:.1%}")
    print(f"  F1-Score (Macro): {metrics['f1_macro']:.3f}")

    # Performance assessment
    mae = metrics['mae']
    if mae < 0.5:
        print(f"\n\u2605 EXCELLENT: MAE < 0.5 (meets target)")
    elif mae < 1.0:
        print(f"\n\u2605 GOOD: MAE < 1.0 (acceptable)")
    else:
        print(f"\n\u2605 NEEDS IMPROVEMENT: MAE >= 1.0")

    if save_plots:
        print(f"\nGenerated files saved to: {output_dir}")

    return metrics, all_preds, all_labels, all_probs


def compare_ringlemann_models(data_dir: Path = None):
    """
    Compare all available Ringlemann models side-by-side
    """
    print("\n" + "=" * 60)
    print("RINGLEMANN MODEL COMPARISON")
    print("=" * 60)

    if data_dir is None:
        data_dir = RINGLEMANN_DATASET_V1

    results = {}
    model_configs = [
        ('mobilenet', RINGLEMANN_MODEL_MOBILENET),
        ('resnet18', RINGLEMANN_MODEL_RESNET18),
        ('custom_cnn', RINGLEMANN_MODEL_CNN)
    ]

    for model_type, model_path in model_configs:
        if model_path.exists():
            print(f"\nEvaluating {model_type}...")
            try:
                metrics, _, _, _ = evaluate_ringlemann_model(
                    model_path=model_path,
                    model_type=model_type,
                    data_dir=data_dir,
                    save_plots=True
                )
                results[model_type] = metrics
            except Exception as e:
                print(f"   \u2717 Error: {e}")
        else:
            print(f"\n{model_type}: Model not found at {model_path}")

    # Print comparison table
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        headers = ['Metric'] + list(results.keys())
        print(f"\n{headers[0]:<25}", end='')
        for h in headers[1:]:
            print(f"{h:<15}", end='')
        print()
        print("-" * (25 + 15 * len(results)))

        metrics_to_compare = [
            ('Accuracy', 'accuracy', lambda x: f'{x:.1%}'),
            ('MAE', 'mae', lambda x: f'{x:.4f}'),
            ('Weighted MAE', 'weighted_mae', lambda x: f'{x:.4f}'),
            ('Tolerance +/-1', 'tolerance_1_accuracy', lambda x: f'{x:.1%}'),
            ('F1 (Macro)', 'f1_macro', lambda x: f'{x:.3f}'),
        ]

        for display_name, metric_key, formatter in metrics_to_compare:
            print(f"{display_name:<25}", end='')
            for model_type in results.keys():
                val = results[model_type].get(metric_key, 0)
                print(f"{formatter(val):<15}", end='')
            print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Ringlemann classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['mobilenet', 'resnet18', 'resnet34', 'custom_cnn', 'all'],
        default='mobilenet',
        help='Model type to evaluate (use "all" to compare all models)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save outputs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving plots'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.model_type == 'all':
        compare_ringlemann_models(data_dir=data_dir)
    else:
        # Determine model path
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            # Use default paths based on model type
            default_paths = {
                'mobilenet': RINGLEMANN_MODEL_MOBILENET,
                'resnet18': RINGLEMANN_MODEL_RESNET18,
                'resnet34': RINGLEMANN_MODEL_RESNET18,  # Fallback to resnet18
                'custom_cnn': RINGLEMANN_MODEL_CNN
            }
            model_path = default_paths[args.model_type]

        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            print("Train a model first using: python scripts/train_ringlemann.py")
            sys.exit(1)

        evaluate_ringlemann_model(
            model_path=model_path,
            model_type=args.model_type,
            data_dir=data_dir,
            batch_size=args.batch_size,
            save_plots=not args.no_plots,
            output_dir=output_dir
        )


if __name__ == '__main__':
    main()
