"""
Remove Stray .pt Model Files from Project Root

This utility helps clean up YOLO model files (.pt) that may have been
cached to the project root directory by the Ultralytics library.

Usage:
    python scripts/cleanup_stray_models.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT


def cleanup_stray_models():
    """
    Remove .pt files from project root (keeping organized directories).

    Finds and optionally deletes .pt model files that exist directly in
    the project root, while preserving files in models/ and experiments/.
    """
    # Find .pt files only in the project root (not subdirectories)
    root_pt_files = list(PROJECT_ROOT.glob("*.pt"))

    if not root_pt_files:
        print("No stray .pt files found in project root.")
        print("\nProject structure is clean!")
        return

    print("=" * 60)
    print("STRAY MODEL FILES DETECTED")
    print("=" * 60)
    print(f"\nFound {len(root_pt_files)} stray .pt file(s) in project root:\n")

    total_size_mb = 0
    for file in root_pt_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size_mb += size_mb
        print(f"  - {file.name:30s} ({size_mb:6.2f} MB)")

    print(f"\n  Total size: {total_size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("These files are likely cached by Ultralytics YOLO library.")
    print("The authoritative copies are in models/pretrained/")
    print("=" * 60)

    # Ask for confirmation
    response = input("\nDelete these files? (y/N): ").strip().lower()

    if response == 'y':
        print("\nDeleting files...")
        for file in root_pt_files:
            try:
                file.unlink()
                print(f"  [DELETED] {file.name}")
            except Exception as e:
                print(f"  [ERROR] {file.name}: {e}")

        print(f"\nCleanup complete! Freed {total_size_mb:.2f} MB of space.")
    else:
        print("\nCleanup cancelled. No files were deleted.")


if __name__ == "__main__":
    cleanup_stray_models()
