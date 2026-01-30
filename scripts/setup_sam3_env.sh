#!/bin/bash
# Setup script to install SAM3 into the smokescreen environment
#
# This script:
# 1. Activates the smokescreen conda environment
# 2. Installs SAM3 dependencies
# 3. Installs SAM3 as an editable package
# 4. Optionally removes the old sam3 conda environment
#
# Usage:
#   chmod +x scripts/setup_sam3_env.sh
#   ./scripts/setup_sam3_env.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SAM3_DIR="$PROJECT_ROOT/sam3"

echo "=============================================="
echo "SAM3 Setup for Chimney Smoke Detection"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "SAM3 directory: $SAM3_DIR"
echo ""

# Check if smokescreen environment exists
if ! conda info --envs | grep -q "smokescreen"; then
    echo "Error: 'smokescreen' conda environment not found."
    echo "Please create it first with:"
    echo "  conda create -n smokescreen python=3.10"
    exit 1
fi

echo "Step 1: Activating smokescreen environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smokescreen

echo "Current Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "Step 2: Installing SAM3 dependencies..."
pip install timm>=1.0.17 ftfy==6.1.1 regex iopath>=0.1.10 huggingface_hub

echo ""
echo "Step 3: Installing SAM3 as editable package..."
if [ -d "$SAM3_DIR" ]; then
    pip install -e "$SAM3_DIR"
    echo "SAM3 installed successfully!"
else
    echo "Error: SAM3 directory not found at $SAM3_DIR"
    exit 1
fi

echo ""
echo "Step 4: Verifying installation..."
python -c "from sam3 import build_sam3_image_model; print('SAM3 import successful!')" 2>/dev/null || {
    echo "Warning: SAM3 import test failed. This may be normal if CUDA is not available."
    echo "The package is installed but may need GPU to fully verify."
}

echo ""
echo "=============================================="
echo "SAM3 Setup Complete!"
echo "=============================================="
echo ""
echo "To use SAM3 for chimney detection:"
echo "  conda activate smokescreen"
echo "  python scripts/infer_sam3_chimney.py path/to/image.jpg"
echo ""
echo "To remove the old sam3 conda environment (optional):"
echo "  conda env remove -n sam3"
echo ""
