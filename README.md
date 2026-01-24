# Chimney Smoke Detection System

A two-stage deep learning pipeline for detecting smoke emissions from chimney images using YOLO-based chimney detection followed by CNN-based smoke classification.

## Overview

This project implements an automated system for detecting smoke from industrial chimneys using computer vision and deep learning. The system uses a two-stage approach:

1. **Stage 1: Chimney Detection** - Uses YOLOv8/YOLOv12 to locate chimneys in images
2. **Stage 2: Smoke Classification** - Uses a CNN (MobileNetV2 or Custom CNN) to classify whether smoke is present

## Architecture

### Stage 1: Chimney Detection (YOLO)
- **Model**: YOLOv8n / YOLOv12n
- **Purpose**: Detect and localize chimney bounding boxes in images
- **Input**: Full images (800x800)
- **Output**: Bounding boxes with confidence scores
- **Best Model**: YOLOv8 trained on v4 dataset

### Stage 2: Smoke Classification
- **Model Options**:
  - MobileNetV2 (Fast, 27MB)
  - Custom CNN (More accurate, 314MB)
- **Purpose**: Classify cropped chimney regions as smoke/no-smoke
- **Input**: Cropped chimney regions (224x224)
- **Output**: Binary classification with confidence score

## Project Structure

```
chimney-smoke-detection/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git exclusions
│
├── src/                      # Source code (modular packages)
│   ├── config.py             # Centralized configuration
│   ├── models/               # Model architectures
│   │   ├── smoke_cnn.py      # Custom CNN architecture
│   │   └── model_loader.py   # Model loading utilities
│   ├── training/             # Training scripts
│   ├── inference/            # Inference pipeline
│   ├── evaluation/           # Evaluation scripts
│   └── utils/                # Utility functions
│
├── scripts/                  # Executable scripts
│   ├── run_inference.py      # Single image inference
│   ├── run_batch_inference.py # Batch processing
│   ├── extract_chimneys.py   # Extract chimney crops
│   └── test_gpu.py           # Test GPU availability
│
├── data/                     # Datasets (gitignored)
│   ├── chimney_detection/    # YOLO datasets (v1, v3, v4, v6)
│   ├── smoke_classification/ # Smoke classification dataset
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
│
├── models/                   # Trained weights (gitignored)
│   ├── pretrained/           # Base pretrained models
│   ├── chimney_detection/    # Trained chimney detectors
│   └── smoke_classification/ # Trained smoke classifiers
│
├── experiments/              # Training outputs (gitignored)
│   ├── chimney_detection_yolov8/
│   ├── chimney_detection_yolov12/
│   └── smoke_classification/
│
├── outputs/                  # Results and visualizations
│   ├── images/               # Annotated images
│   ├── plots/                # Training plots
│   └── results/              # Batch results
│
├── tests/                    # Test scripts
├── docs/                     # Documentation
└── assets/                   # Test images and demos
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/VedantChaskar/chimney-smoke-detection
cd chimney-smoke-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up directory structure:

The project expects the following directory structure. Create missing directories if needed:

```bash
# On Unix/Linux/macOS:
mkdir -p data/chimney_detection/v4/{train,valid,test}/{images,labels}
mkdir -p models/pretrained
mkdir -p models/chimney_detection
mkdir -p models/smoke_classification
mkdir -p outputs/{images,plots,results}
mkdir -p experiments

# On Windows (PowerShell):
New-Item -ItemType Directory -Force -Path data\chimney_detection\v4\train\images
New-Item -ItemType Directory -Force -Path data\chimney_detection\v4\train\labels
New-Item -ItemType Directory -Force -Path data\chimney_detection\v4\valid\images
New-Item -ItemType Directory -Force -Path data\chimney_detection\v4\valid\labels
New-Item -ItemType Directory -Force -Path data\chimney_detection\v4\test\images
New-Item -ItemType Directory -Force -Path data\chimney_detection\v4\test\labels
New-Item -ItemType Directory -Force -Path models\pretrained
New-Item -ItemType Directory -Force -Path models\chimney_detection
New-Item -ItemType Directory -Force -Path models\smoke_classification
New-Item -ItemType Directory -Force -Path outputs\images
New-Item -ItemType Directory -Force -Path outputs\plots
New-Item -ItemType Directory -Force -Path outputs\results
New-Item -ItemType Directory -Force -Path experiments
```

**Note**: The `.gitkeep` files already present will preserve these directories in git.

### Model Weights and Datasets

Due to their large size, model weights and datasets are not included in this repository. You can either:

**Option 1: Download Pre-trained Models**

Download your trained models and datasets, then place them in the appropriate directories:

- **Pretrained YOLO models**: Place in `models/pretrained/`
  - `yolov8n.pt`
  - `yolo11n.pt`
  - `yolo12n.pt`

- **Trained models**: Place in respective directories
  - Chimney detection: `models/chimney_detection/yolov8_best.pt`
  - Smoke classification: `models/smoke_classification/mobilenet_best.pt`

- **Datasets**: Place in `data/chimney_detection/v4/`
  - Training images in `data/chimney_detection/v4/train/images/`
  - Validation images in `data/chimney_detection/v4/valid/images/`
  - Test images in `data/chimney_detection/v4/test/images/`

Contact the project maintainer for access to trained weights.

**Option 2: Train from Scratch**
- Download datasets and place in `data/chimney_detection/v4/` as shown above
- Follow training instructions below

### YOLO Model Caching

When training YOLO models for the first time, Ultralytics will download pretrained weights to a cache directory:

- **Linux/macOS**: `~/.cache/ultralytics/`
- **Windows**: `%USERPROFILE%\.cache\ultralytics\`

To use a custom cache location, set the environment variable before training:

```bash
# Unix/Linux/macOS
export YOLO_HOME=/path/to/custom/cache

# Windows (PowerShell)
$env:YOLO_HOME="C:\path\to\custom\cache"
```

## Quick Start

### Test GPU Availability
```bash
python scripts/test_gpu.py
```

### Run Inference on a Single Image
```bash
python scripts/run_inference.py assets/test_image.jpg
```

### Batch Process Multiple Images
```bash
python scripts/run_batch_inference.py path/to/images/ outputs/results/
```

## Training

### Train Chimney Detector (YOLOv8)
```bash
python -m src.training.train_chimney_detector
```

Configuration options in `src/config.py`:
- `EPOCHS`: Number of training epochs (default: 100)
- `IMG_SIZE`: Image size for training (default: 800)
- `BATCH_SIZE`: Batch size (default: 16)

### Train Smoke Classifier

**Option 1: MobileNetV2 (Faster, Smaller)**
```bash
python -m src.training.train_smoke_mobilenet
```

**Option 2: Custom CNN (More Accurate, Larger)**
```bash
python -m src.training.train_smoke_cnn
```

Configuration options in `src/config.py`:
- `EPOCHS`: Number of training epochs (default: 30)
- `IMG_SIZE`: Image size (default: 224)
- `BATCH_SIZE`: Batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 0.001)

## Evaluation

Run detailed evaluation on test set:
```bash
python -m src.evaluation.evaluate
```

This will generate:
- Confusion matrix
- Precision, Recall, F1-Score metrics
- Confidence analysis plots
- Per-class performance metrics

## Results

### Chimney Detection (YOLOv8)
- **mAP50**: 90.5%
- **Precision**: 89.2%
- **Recall**: 91.8%

## Usage Examples

### Python API

```python
from src.inference.pipeline import SmokeDetectionPipeline
from src.config import InferencePipelineConfig

# Initialize pipeline
pipeline = SmokeDetectionPipeline(config=InferencePipelineConfig)

# Process single image
result = pipeline.process_image('path/to/image.jpg')

print(f"Smoke detected: {result['smoke_detected']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Chimney location: {result['chimney_bbox']}")
```

### Command Line

```bash
# Single image inference
python scripts/run_inference.py image.jpg --output result.jpg

# Batch processing
python scripts/run_batch_inference.py input_folder/ output_folder/

# Extract chimney crops from images
python scripts/extract_chimneys.py image.jpg --output chimney_crop.jpg
```

## Configuration

All configuration settings are centralized in `src/config.py`:

- **Paths**: Data directories, model paths, output locations
- **Training Hyperparameters**: Epochs, batch size, learning rate, etc.
- **Inference Settings**: Confidence thresholds, image sizes
- **Device Configuration**: GPU/CPU selection

To modify settings, edit `src/config.py` or override in your scripts.

## Development

### Adding a New Model

1. Create model class in `src/models/your_model.py`
2. Add model path to `src/config.py`
3. Create training script in `src/training/train_your_model.py`
4. Update model loader in `src/models/model_loader.py`

### Running Tests

```bash
# Test individual components
python tests/test_chimney_detection.py
python tests/test_smoke_classifier.py
python tests/test_pipeline_v2.py
```

## Acknowledgments

- **Ultralytics** - YOLOv8/YOLOv12 implementation
- **PyTorch Team** - Deep learning framework
- **Roboflow** - Dataset management and augmentation

## Contact

For questions or issues:
- **Author**: Vedant Chaskar
- **Email**: chaskarvedant1@gmail.com
- **GitHub**: [Open an issue](../../issues)

---

**Note**: For production deployment, additional considerations for robustness, scalability, and reliability should be addressed.
