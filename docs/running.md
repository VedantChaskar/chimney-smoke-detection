# Running the Project — Complete Guide

This document provides instructions for running all scripts in the chimney smoke detection project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Stage 1: Chimney Detection](#stage-1-chimney-detection)
4. [Stage 2: Smoke Classification (Binary)](#stage-2-smoke-classification-binary)
5. [Stage 3: Ringlemann Scale Classification](#stage-3-ringlemann-scale-classification)
6. [Full Pipeline Inference](#full-pipeline-inference)
7. [Alternative Opacity Pipelines](#alternative-opacity-pipelines)
8. [Utility Scripts](#utility-scripts)
9. [Dataset Paths](#dataset-paths)

---

## Prerequisites

```bash
# Activate conda environment
conda activate smokescreen

# Verify GPU availability
python scripts/util_check_gpu.py
```

---

## Project Structure

```
chimney-smoke-detection/
├── data/                          # Datasets (gitignored — download separately)
│   ├── chimney_detection/v4/      # YOLO format chimney dataset
│   ├── smoke_classification/v1/   # Binary smoke dataset (smoke/no_smoke)
│   └── Ringlemann Test.v1i.folder/ # 6-class Ringlemann dataset (R0-R5)
├── models/                        # Model weights (gitignored — train or download)
│   ├── chimney_detection/
│   ├── smoke_classification/
│   └── ringlemann_classification/
├── scripts/                       # Standalone scripts
├── src/                           # Core source modules
├── physics_methods/               # All 5 physics/signal-based opacity pipelines
│   ├── smokescreen/               # SmokeScreen pipeline (Chen brightness-ratio)
│   ├── smoke_opacity/             # Dark Channel Prior pipeline
│   ├── vitmatte/                  # ViTMatte alpha matting pipeline
│   ├── temporal_reference/        # Temporal Reference pipeline
│   └── gemini_inpainting/         # Gemini AI Inpainting pipeline
├── sam3/                          # Vendored SAM3 segmentation model
└── assets/                        # Sample test images
```

---

## Stage 1: Chimney Detection

### Train Chimney Detector (YOLO)

```bash
# Train YOLOv8 (default)
python -m src.training.train_chimney_detector

# Train YOLOv12
python -m src.training.train_chimney_detector --model yolo12 --epochs 100

# Train with custom parameters
python -m src.training.train_chimney_detector \
    --model yolo12 \
    --dataset v4 \
    --epochs 100 \
    --img-size 800 \
    --batch-size 16 \
    --patience 50 \
    --name my_experiment
```

### Run Chimney Detection Inference

```bash
# Single image with YOLOv12
python scripts/infer_yolov12_chimney.py path/to/image.jpg

# With custom output
python scripts/infer_yolov12_chimney.py image.jpg --output result.jpg

# Batch processing
python scripts/infer_yolov12_chimney.py images_folder/ --batch --output-dir results/

# Custom confidence threshold
python scripts/infer_yolov12_chimney.py image.jpg --chimney-threshold 0.25
```

### Extract Chimney Crops

```bash
python scripts/crop_chimney_regions.py input_folder/ output_folder/
python scripts/crop_chimney_regions.py input_folder/ output_folder/ --conf 0.5
```

---

## Stage 2: Smoke Classification (Binary)

### Train Smoke Classifier

```bash
# Train MobileNetV2 (recommended)
python -m src.training.train_smoke_mobilenet

# Train with custom parameters
python -m src.training.train_smoke_mobilenet \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --patience 10

# Train Custom CNN
python -m src.training.train_smoke_cnn --epochs 30 --batch-size 32
```

### Evaluate Smoke Classifier

```bash
python -m src.evaluation.evaluate --model mobilenet
python -m src.evaluation.evaluate --model cnn
python -m src.evaluation.evaluate --model both
```

---

## Stage 3: Ringlemann Scale Classification

### Analyze Ringlemann Dataset

```bash
python scripts/analyze_ringlemann_data.py
python scripts/analyze_ringlemann_data.py --data-dir "data/Ringlemann Test.v1i.folder"
```

### Train Ringlemann Classifier

```bash
# Train MobileNetV2 with class weights (recommended for imbalanced data)
python scripts/train_ringlemann.py --model-type mobilenet --use-class-weights --epochs 50

# Train ResNet18
python scripts/train_ringlemann.py --model-type resnet18 --epochs 50

# Train Custom CNN
python scripts/train_ringlemann.py --model-type custom_cnn

# Full custom training
python scripts/train_ringlemann.py \
    --data-dir "data/Ringlemann Test.v1i.folder" \
    --model-type mobilenet \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --use-class-weights \
    --patience 10
```

### Evaluate Ringlemann Classifier

```bash
python scripts/evaluate_ringlemann.py --model-type mobilenet
python scripts/evaluate_ringlemann.py --model-type mobilenet --data-dir "data/Ringlemann Test.v1i.folder"
```

---

## Full Pipeline Inference

### Single Image Inference (all 3 stages)

```bash
# Basic usage
python scripts/infer_smoke_pipeline.py assets/test_image.jpg

# With output
python scripts/infer_smoke_pipeline.py image.jpg --output result.jpg

# Custom thresholds
python scripts/infer_smoke_pipeline.py image.jpg \
    --chimney-threshold 0.2 \
    --smoke-threshold 0.6
```

### Batch Image Inference

```bash
python scripts/infer_batch_images.py input_folder/
python scripts/infer_batch_images.py input_folder/ output_folder/
```

### Python API

```python
from src.inference import SmokeDetectionPipeline

pipeline = SmokeDetectionPipeline(
    ringlemann_model_path="models/ringlemann_classification/mobilenet_best.pt",
    ringlemann_model_type="mobilenet"
)
results = pipeline.process_image("path/to/image.jpg", output_path="output.jpg")
for r in results:
    print(f"Smoke: {r['smoke_result']['prediction']}")
    if r['ringlemann_result']:
        print(f"Ringlemann: {r['ringlemann_result']['ringlemann_class']}")
```

---

## Alternative Opacity Pipelines

See `docs/pipelines.md` for a comparison of all 5 approaches.

### SmokeScreen (Chen brightness-ratio)
```bash
cd physics_methods/smokescreen
python scripts/run_inference.py --input ../../assets/ --output-dir ../../outputs/smokescreen/
```

### Dark Channel Prior
```bash
cd physics_methods
python -m smoke_opacity.main --input ../assets/test_image.jpg
```

### ViTMatte
```bash
cd physics_methods
python -m vitmatte.main --input ../assets/test_image.jpg --no-sam
python -m vitmatte.main --input ../assets/test_image.jpg  # with SAM3 segmentation
```

### Temporal Reference (requires ground_truth.jpg)
```bash
cd physics_methods/temporal_reference
python run.py --input ../../assets/temporal_reference_test/
```

### Gemini Inpainting (requires API key)
```bash
# Set up API key first:
cp physics_methods/gemini_inpainting/.env.example physics_methods/gemini_inpainting/.env
# Edit .env and add your key from https://aistudio.google.com/app/apikey

cd physics_methods/gemini_inpainting
python run.py --input path/to/image.jpg
```

---

## Utility Scripts

```bash
# Check GPU availability
python scripts/util_check_gpu.py

# Cleanup model cache
python scripts/util_cleanup_models.py

# SAM3 chimney detection
python scripts/infer_sam3_chimney.py path/to/image.jpg
```

---

## Dataset Paths

| Dataset | Path | Format |
|---------|------|--------|
| Chimney Detection | `data/chimney_detection/v4/` | YOLO |
| Smoke Classification | `data/smoke_classification/v1/` | ImageFolder |
| Ringlemann | `data/Ringlemann Test.v1i.folder/` | ImageFolder |

---

## Logs

All training and inference scripts save logs to `logs/` directory with timestamps.

## Running Tests

```bash
# All tests across all pipelines
conda run -n smokescreen python -m pytest -v

# Specific pipeline
conda run -n smokescreen python -m pytest physics_methods/smokescreen/tests/ -v
conda run -n smokescreen python -m pytest physics_methods/smoke_opacity/tests/ -v
conda run -n smokescreen python -m pytest physics_methods/vitmatte/tests/ -v
```
