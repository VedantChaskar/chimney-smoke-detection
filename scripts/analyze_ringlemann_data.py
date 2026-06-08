#!/usr/bin/env python
"""
Ringlemann Dataset Analysis Script

Analyze dataset readiness and class distribution for Ringlemann classification.
Generates comprehensive reports on data quality and recommendations.

Usage:
    python scripts/analyze_ringlemann_data.py
    python scripts/analyze_ringlemann_data.py --data-dir data/ringlemann_classification/v1
    python scripts/analyze_ringlemann_data.py --show-samples
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

from src.utils.data_utils import get_ringlemann_dataset_statistics, get_ringlemann_class_weights
from src.config import (
    RINGLEMANN_DATASET_V1,
    OUTPUT_PLOTS_DIR,
    ensure_dir,
    RINGLEMANN_CLASSES
)


def analyze_ringlemann_dataset(
    data_dir: Path,
    output_dir: Path = None,
    show_samples: bool = False
):
    """
    Generate comprehensive dataset analysis report

    Args:
        data_dir: Path to Ringlemann dataset directory
        output_dir: Directory to save analysis outputs
        show_samples: Whether to display sample images
    """
    print("=" * 60)
    print("RINGLEMANN DATASET ANALYSIS")
    print("=" * 60)

    if not data_dir.exists():
        print(f"\nError: Dataset directory not found: {data_dir}")
        print("\nExpected directory structure:")
        print("  data_dir/")
        print("  ├── train/")
        print("  │   ├── R0/")
        print("  │   ├── R1/")
        print("  │   ├── R2/")
        print("  │   ├── R3/")
        print("  │   ├── R4/")
        print("  │   └── R5/")
        print("  ├── valid/")
        print("  │   └── [same structure]")
        print("  └── test/")
        print("      └── [same structure]")
        return None

    if output_dir is None:
        output_dir = OUTPUT_PLOTS_DIR / "ringlemann_data_analysis"
    ensure_dir(output_dir)

    print(f"\nDataset path: {data_dir}")

    # Get statistics
    stats = get_ringlemann_dataset_statistics(data_dir)

    print(f"\n--- Dataset Overview ---")
    print(f"Total images: {stats['total']}")

    # Check which splits exist
    splits_found = list(stats['splits'].keys())
    print(f"Splits found: {', '.join(splits_found)}")

    # Per-split analysis
    print(f"\n--- Per-Split Statistics ---")
    all_class_counts = {cls: 0 for cls in RINGLEMANN_CLASSES}

    for split in ['train', 'valid', 'test']:
        if split not in stats['splits']:
            print(f"\n{split.upper()}: NOT FOUND")
            continue

        split_stats = stats['splits'][split]
        print(f"\n{split.upper()}:")
        print(f"  Total: {split_stats['total']} images")

        if split_stats['classes']:
            for cls in RINGLEMANN_CLASSES:
                count = split_stats['classes'].get(cls, 0)
                pct = (count / split_stats['total'] * 100) if split_stats['total'] > 0 else 0
                print(f"    {cls}: {count:4d} ({pct:5.1f}%)")
                all_class_counts[cls] += count

    # Overall class distribution
    print(f"\n--- Overall Class Distribution ---")
    total = stats['total']
    for cls in RINGLEMANN_CLASSES:
        count = all_class_counts[cls]
        pct = (count / total * 100) if total > 0 else 0
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  {cls}: {count:4d} ({pct:5.1f}%) {bar}")

    # Imbalance analysis
    print(f"\n--- Class Imbalance Analysis ---")
    counts = list(all_class_counts.values())
    non_zero_counts = [c for c in counts if c > 0]

    if non_zero_counts:
        max_count = max(counts)
        min_count = min(non_zero_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        most_common = max(all_class_counts, key=all_class_counts.get)
        least_common = min((k for k, v in all_class_counts.items() if v > 0),
                          key=all_class_counts.get, default=None)

        print(f"  Most common class: {most_common} ({all_class_counts[most_common]} images)")
        if least_common:
            print(f"  Least common class: {least_common} ({all_class_counts[least_common]} images)")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

        # Class weights
        if (data_dir / 'train').exists():
            print(f"\n  Suggested class weights (for weighted loss):")
            try:
                weights = get_ringlemann_class_weights(data_dir)
                for i, cls in enumerate(RINGLEMANN_CLASSES):
                    print(f"    {cls}: {weights[i]:.4f}")
            except Exception as e:
                print(f"    Error calculating weights: {e}")
    else:
        print("  No images found in dataset!")

    # Missing classes
    missing_classes = [cls for cls, count in all_class_counts.items() if count == 0]
    if missing_classes:
        print(f"\n  ⚠ WARNING: Missing classes: {', '.join(missing_classes)}")

    # Underrepresented classes (less than 20 images)
    underrepresented = [cls for cls, count in all_class_counts.items() if 0 < count < 20]
    if underrepresented:
        print(f"  ⚠ WARNING: Underrepresented classes (< 20 images): {', '.join(underrepresented)}")

    # Recommendations
    print(f"\n--- Recommendations ---")

    min_recommended = 50  # Minimum images per class
    good_threshold = 200  # Good number of images per class

    recommendations = []

    for cls in RINGLEMANN_CLASSES:
        count = all_class_counts[cls]
        if count == 0:
            recommendations.append(f"  \u2717 {cls}: CRITICAL - No images! Collect at least {min_recommended}.")
        elif count < min_recommended:
            recommendations.append(f"  \u26a0 {cls}: Needs {min_recommended - count} more images (have {count}).")
        elif count < good_threshold:
            recommendations.append(f"  \u25cb {cls}: Could use {good_threshold - count} more images for optimal training.")
        else:
            recommendations.append(f"  \u2713 {cls}: Good coverage ({count} images).")

    for rec in recommendations:
        print(rec)

    if imbalance_ratio > 3:
        print(f"\n  \u26a0 High class imbalance detected ({imbalance_ratio:.1f}x).")
        print("    Consider using weighted loss function: --use-class-weights")

    # Data quality check
    print(f"\n--- Data Quality Check ---")
    quality_issues = []

    # Check image sizes (sample a few)
    if 'train' in stats['splits']:
        train_dir = data_dir / 'train'
        sample_sizes = []
        for cls_dir in train_dir.iterdir():
            if cls_dir.is_dir():
                for img_path in list(cls_dir.iterdir())[:5]:
                    try:
                        with Image.open(img_path) as img:
                            sample_sizes.append(img.size)
                    except Exception:
                        quality_issues.append(f"Could not read: {img_path.name}")

        if sample_sizes:
            widths = [s[0] for s in sample_sizes]
            heights = [s[1] for s in sample_sizes]
            print(f"  Sample image sizes:")
            print(f"    Width range: {min(widths)} - {max(widths)}")
            print(f"    Height range: {min(heights)} - {max(heights)}")

            if max(widths) < 224 or max(heights) < 224:
                quality_issues.append("Some images may be too small (< 224px)")

    if quality_issues:
        print(f"\n  Potential issues:")
        for issue in quality_issues:
            print(f"    \u26a0 {issue}")
    else:
        print(f"  \u2713 No obvious quality issues detected")

    # Generate visualizations
    print(f"\n--- Generating Visualizations ---")

    # 1. Class distribution bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(RINGLEMANN_CLASSES)))
    bars = ax1.bar(RINGLEMANN_CLASSES, [all_class_counts[c] for c in RINGLEMANN_CLASSES],
                   color=colors, edgecolor='black')
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_xlabel('Ringlemann Class', fontsize=12)
    ax1.set_title('Overall Class Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Per-split distribution
    ax2 = axes[1]
    x = np.arange(len(RINGLEMANN_CLASSES))
    width = 0.25
    split_colors = {'train': 'steelblue', 'valid': 'orange', 'test': 'green'}

    for i, split in enumerate(['train', 'valid', 'test']):
        if split in stats['splits']:
            counts = [stats['splits'][split]['classes'].get(c, 0) for c in RINGLEMANN_CLASSES]
            ax2.bar(x + i*width, counts, width, label=split.capitalize(),
                   color=split_colors[split], edgecolor='black')

    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_xlabel('Ringlemann Class', fontsize=12)
    ax2.set_title('Class Distribution by Split', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(RINGLEMANN_CLASSES)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    dist_path = output_dir / 'class_distribution.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 Saved: {dist_path.name}")

    # 2. Imbalance visualization
    if total > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages = [(all_class_counts[c] / total * 100) for c in RINGLEMANN_CLASSES]

        # Horizontal bar chart
        y_pos = np.arange(len(RINGLEMANN_CLASSES))
        bars = ax.barh(y_pos, percentages, color=colors, edgecolor='black')

        # Add ideal line (16.67% for 6 classes)
        ideal_pct = 100 / len(RINGLEMANN_CLASSES)
        ax.axvline(x=ideal_pct, color='red', linestyle='--', linewidth=2, label=f'Ideal ({ideal_pct:.1f}%)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(RINGLEMANN_CLASSES)
        ax.set_xlabel('Percentage of Dataset', fontsize=12)
        ax.set_title('Class Imbalance Analysis', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                   f'{pct:.1f}%', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        imb_path = output_dir / 'imbalance_analysis.png'
        plt.savefig(imb_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  \u2713 Saved: {imb_path.name}")

    # 3. Sample images grid (if requested)
    if show_samples and 'train' in stats['splits']:
        print(f"\n--- Sample Images ---")
        fig, axes = plt.subplots(len(RINGLEMANN_CLASSES), 3, figsize=(12, 4*len(RINGLEMANN_CLASSES)))

        train_dir = data_dir / 'train'

        for i, cls in enumerate(RINGLEMANN_CLASSES):
            cls_dir = train_dir / cls
            if cls_dir.exists():
                images = list(cls_dir.glob('*.[jJ][pP][gG]')) + list(cls_dir.glob('*.[pP][nN][gG]'))
                images = images[:3]  # Take up to 3 samples

                for j in range(3):
                    ax = axes[i, j] if len(RINGLEMANN_CLASSES) > 1 else axes[j]
                    if j < len(images):
                        try:
                            img = Image.open(images[j])
                            ax.imshow(img)
                            ax.set_title(f'{cls} - Sample {j+1}', fontsize=10)
                        except Exception:
                            ax.text(0.5, 0.5, 'Error loading', ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                    ax.axis('off')
            else:
                for j in range(3):
                    ax = axes[i, j] if len(RINGLEMANN_CLASSES) > 1 else axes[j]
                    ax.text(0.5, 0.5, f'{cls}: No data', ha='center', va='center')
                    ax.axis('off')

        plt.suptitle('Sample Images from Training Set', fontsize=14, fontweight='bold')
        plt.tight_layout()
        samples_path = output_dir / 'sample_images.png'
        plt.savefig(samples_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  \u2713 Saved: {samples_path.name}")

    # Summary
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nTotal images: {stats['total']}")

    readiness = "READY" if (stats['total'] >= 100 and not missing_classes) else "NEEDS WORK"
    print(f"Dataset status: {readiness}")

    print(f"\nOutputs saved to: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Ringlemann classification dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to Ringlemann dataset directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save analysis outputs'
    )

    parser.add_argument(
        '--show-samples',
        action='store_true',
        help='Generate sample images grid'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else RINGLEMANN_DATASET_V1
    output_dir = Path(args.output_dir) if args.output_dir else None

    analyze_ringlemann_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        show_samples=args.show_samples
    )


if __name__ == '__main__':
    main()
