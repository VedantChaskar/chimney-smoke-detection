# Chimney Smoke Detection System

An automated three-stage deep learning pipeline for detecting and grading smoke emissions from chimney images. The system locates chimneys (YOLOv12), classifies binary smoke presence (MobileNetV2), and grades opacity on the international Ringlemann scale R0–R5 (MobileNetV2). Five alternative opacity estimation pipelines are also provided for research comparison.

---

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1: Chimney Detection (YOLOv12)   │  mAP50: 93.2%
│  Locates chimney bounding boxes         │
└──────────────────────────┬──────────────┘
                           │ crop + pad
                           ▼
┌─────────────────────────────────────────┐
│  Stage 2: Smoke Classifier (MobileNetV2)│  Accuracy: 87.5%
│  Binary: smoke / no smoke               │
└──────────────────────────┬──────────────┘
                           │ if smoke
                           ▼
┌─────────────────────────────────────────┐
│  Stage 3: Ringlemann Grader (MobileNetV2│  Val MAE: 0.4
│  6-class: R0 (clear) → R5 (black)       │
└─────────────────────────────────────────┘

Alternative opacity pipelines (no training required):
  smokescreen/        — Chen brightness-ratio grading
  smoke_opacity/      — Dark Channel Prior transmittance
  vitmatte/           — ViT alpha matting
  temporal_reference/ — Stationary-camera temporal GT
  gemini_inpainting/  — Gemini AI synthetic GT inpainting
```

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (Tesla V100 tested; CPU inference is possible but slow)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

---

## Installation

```bash
git clone https://github.com/<your-username>/chimney-smoke-detection.git
cd chimney-smoke-detection

# Create and activate the conda environment
conda create -n smokescreen python=3.10 -y
conda activate smokescreen

# Install PyTorch with CUDA (adjust cu121 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt

# (Optional) Dev dependencies for testing and notebooks
pip install -r requirements-dev.txt
```

### SAM3 (vendored)

SAM3 (Segment Anything Model v3) is vendored in `sam3/` and is not a pip package. The opacity pipelines add it to `sys.path` automatically — no additional setup is needed. Model weights are downloaded from HuggingFace on first use.

### Gemini Inpainting

The `gemini_inpainting/` pipeline requires a free Gemini API key:

```bash
cp physics_methods/gemini_inpainting/.env.example physics_methods/gemini_inpainting/.env
# Edit .env and paste your key from https://aistudio.google.com/app/apikey
```

---

## Model Weights

Trained model weights are not included in the repository (too large for git). You have two options:

**Option A — Download pre-trained weights** (not yet hosted; train from scratch below)

**Option B — Train from scratch:**

```bash
conda activate smokescreen

# Stage 1: Chimney detector (~100 epochs, ~2h on V100)
python -m src.training.train_chimney_detector --model yolo12 --epochs 100

# Stage 2: Smoke classifier (~30 epochs, ~20min)
python -m src.training.train_smoke_mobilenet --epochs 30

# Stage 3: Ringlemann classifier (~50 epochs, ~10min)
python scripts/train_ringlemann.py --model-type mobilenet --use-class-weights --epochs 50
```

---

## Quick Start

```bash
conda activate smokescreen

# Check GPU availability
python scripts/util_check_gpu.py

# Run full 3-stage pipeline on a single image
python scripts/infer_smoke_pipeline.py assets/test_image.jpg

# Run with output saved
python scripts/infer_smoke_pipeline.py assets/test_image.jpg --output result.jpg

# Batch process a folder
python scripts/infer_batch_images.py assets/ outputs/
```

---

## Core Pipeline — Stage by Stage

### Stage 1: Chimney Detection (YOLOv12)

```bash
# Single image
python scripts/infer_yolov12_chimney.py assets/test_image.jpg

# Extract chimney crops for downstream stages
python scripts/crop_chimney_regions.py assets/ outputs/chimney_crops/
```

### Stage 2: Smoke Classification

```bash
python -m src.evaluation.evaluate --model mobilenet
```

### Stage 3: Ringlemann Scale Classification

```bash
# Evaluate on test set
python scripts/evaluate_ringlemann.py --model-type mobilenet

# Train with class weights (recommended — dataset is imbalanced)
python scripts/train_ringlemann.py --model-type mobilenet --use-class-weights --epochs 50
```

---

## Alternative Opacity Pipelines

Five independent pipelines estimate smoke opacity without requiring trained Ringlemann models. See [`docs/pipelines.md`](docs/pipelines.md) for a full comparison.

### SmokeScreen (Chen brightness-ratio)
```bash
cd physics_methods/smokescreen
python scripts/run_inference.py --input ../../assets/ --output-dir ../../outputs/smokescreen/
```

### Dark Channel Prior
```bash
cd physics_methods
python -m smoke_opacity.main --input ../assets/test_image.jpg
python -m smoke_opacity.main --input ../assets/test_image.jpg --no-sam  # HSV fallback
```

### ViTMatte (alpha matting)
```bash
cd physics_methods
python -m vitmatte.main --input ../assets/test_image.jpg
python -m vitmatte.main --input ../assets/test_image.jpg --no-sam
```

### Temporal Reference (requires a clean reference frame)
```bash
cd physics_methods/temporal_reference
python run.py --input ../../assets/temporal_reference_test/
```

### Gemini Inpainting (requires API key)
```bash
cd physics_methods/gemini_inpainting
python run.py --input ../../assets/test_image.jpg
```

---

## Testing

```bash
conda activate smokescreen

# Full test suite (162+ tests across all pipelines)
python -m pytest -v

# Individual pipelines
python -m pytest physics_methods/smokescreen/tests/ -v
python -m pytest physics_methods/smoke_opacity/tests/ -v
python -m pytest physics_methods/vitmatte/tests/ -v
python -m pytest physics_methods/temporal_reference/tests/ -v
python -m pytest physics_methods/gemini_inpainting/tests/ -v
```

---

## Project Structure

```
chimney-smoke-detection/
├── README.md
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Dev/testing dependencies
├── conftest.py                # Pytest namespace isolation across subprojects
├── pytest.ini
│
├── src/                       # Core 3-stage pipeline source
│   ├── config.py              # Centralized paths and constants
│   ├── models/                # Model architectures
│   ├── training/              # Training scripts (importable modules)
│   ├── inference/             # Pipeline: chimney → smoke → Ringlemann
│   ├── evaluation/            # Evaluation utilities
│   └── utils/                 # Shared data utilities
│
├── scripts/                   # Executable entry points
│   ├── infer_smoke_pipeline.py
│   ├── infer_yolov12_chimney.py
│   ├── infer_batch_images.py
│   ├── crop_chimney_regions.py
│   ├── train_ringlemann.py
│   ├── evaluate_ringlemann.py
│   └── util_check_gpu.py
│
├── physics_methods/           # All 5 physics/signal-based opacity pipelines
│   ├── smokescreen/           # Chen brightness-ratio opacity pipeline
│   ├── smoke_opacity/         # Dark Channel Prior opacity pipeline
│   ├── vitmatte/              # ViT alpha matting opacity pipeline
│   ├── temporal_reference/    # Temporal reference opacity pipeline
│   └── gemini_inpainting/     # Gemini AI inpainting pipeline
│       └── .env.example       # Copy to .env and add your API key
│
├── sam3/                      # Vendored SAM3 (Meta, Apache 2.0)
│
├── data/                      # Datasets (gitignored — see below)
│   ├── chimney_detection/     # YOLO format (v1, v3, v4, v6_yolo12)
│   ├── smoke_classification/  # Binary smoke dataset
│   └── Ringlemann Test.v1i.folder/  # 6-class R0–R5 dataset
│
├── models/                    # Trained weights (gitignored)
├── experiments/               # Training run outputs (gitignored)
├── outputs/                   # Inference results (gitignored)
├── logs/                      # Training logs (gitignored)
│
├── assets/                    # Sample test images
│   ├── test_image.jpg
│   ├── temporal_reference_test/   # Images for temporal pipeline
│   └── reference/                 # Clean reference frames
│
├── tests/                     # Core pipeline tests
└── docs/                      # Additional documentation
    ├── pipelines.md           # Pipeline comparison table
    ├── running.md             # Detailed run instructions
    └── technical-notes.md     # Dataset stats, calibration constants
```

---

## Datasets

| Stage | Dataset | Format | Size |
|-------|---------|--------|------|
| Chimney detection | `data/chimney_detection/v4/` | YOLO | ~1k images |
| Smoke classification | `data/smoke_classification/v1/` | ImageFolder | ~2k images |
| Ringlemann grading | `data/Ringlemann Test.v1i.folder/` | ImageFolder | 253 images (R0–R5) |

Datasets are not included in this repository. The Ringlemann dataset was sourced via [Roboflow](https://roboflow.com).

---

## SAM3 Attribution

`sam3/` contains a vendored copy of Meta's [Segment Anything Model 3](https://github.com/facebookresearch/segment-anything-3) (SAM3), licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). No modifications were made to the SAM3 source. It is imported as `import sam3` by all opacity estimation pipelines; model weights are downloaded automatically from HuggingFace on first use.

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8/v12 framework
- [PyTorch](https://pytorch.org) — Deep learning backbone
- [Roboflow](https://roboflow.com) — Dataset annotation and management
- [Meta SAM3](https://github.com/facebookresearch/segment-anything-3) — Vendored segmentation model
- [HuggingFace ViTMatte](https://huggingface.co/hustvl/vitmatte-small-composition-1k) — Alpha matting model
- [Google Gemini](https://ai.google.dev) — AI inpainting API
