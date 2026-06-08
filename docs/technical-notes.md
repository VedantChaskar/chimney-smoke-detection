# Technical Notes: Chimney Smoke Detection & Opacity Estimation

**Author**: Vedant Chaskar  
**Date compiled**: May 2026  
**Environment**: conda env `smokescreen`, Tesla V100-SXM3-32GB, PyTorch 2.6+

---

## 1. PROJECT OVERVIEW

### What the project does
An automated, multi-stage computer vision system that:
1. Detects chimneys in photographs using YOLO object detection
2. Classifies whether smoke is present (binary)
3. Estimates the **opacity / density** of the smoke on the **Ringlemann scale (R0–R5)**

The Ringlemann number is an internationally recognized, standardized metric for smoke opacity used by environmental regulators. R0 = clear/no smoke, R5 = fully opaque black smoke.

### Why it matters (problem statement)
Monitoring chimney smoke emissions is critical for environmental compliance. Traditional inspection methods are manual, expensive, and infrequent. An automated vision-based system can enable continuous, scalable monitoring from existing cameras.

---

## 2. SYSTEM ARCHITECTURE: THREE-STAGE PIPELINE

### High-level flow
```
Input Image
    → [Stage 1] YOLO Chimney Detector  → Bounding box(es)
    → [Stage 2] CNN Smoke Classifier   → Smoke / No Smoke (binary)
    → [Stage 3] Ringlemann Classifier  → R0–R5 smoke density rating
```

Implemented in `src/inference/pipeline.py` as `SmokeDetectionPipeline`.

### Stage 1: Chimney Detection
- **Model**: YOLOv8n / YOLOv12n (Ultralytics)
- **Input**: Full images at 800×800 px
- **Output**: Bounding boxes + confidence scores
- **Dataset**: Roboflow chimney detection dataset (CC BY 4.0)
  - v1 dataset: 90 images total (75 train, 10 valid, 5 test), 1 class: `chimney`
  - Later versions (v4, v6) used for improved training — v4 is default in config
- **Best trained model**: `models/chimney_detection/yolov8_best.pt`
- **Conf threshold**: 0.15 (low, to avoid missing chimneys)
- **Crop padding**: asymmetric — 1.5× box height above (smoke rises upward), 0.3× width on sides, 0.2× height below

### Stage 2: Smoke Classification (Binary)
- **Model options**: MobileNetV2 (27 MB) or custom SmokeCNN (314 MB)
- **Input**: Cropped chimney region (224×224 px), padded upward
- **Output**: `smoke` / `no_smoke` with confidence
- **Dataset**: 237 images total
  - Train: 99 Smoke, 108 No Smoke
  - Valid: 6 Smoke, 14 No Smoke
  - Test: 8 Smoke, 2 No Smoke
- **Preprocessing**: ImageNet normalization [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
- **Augmentation**: RandomHorizontalFlip, RandomRotation(15°), ColorJitter

### Stage 3: Ringlemann Scale Classification (CNN-based, supervised)
- **Classes**: 6 classes — R0 (clear) through R5 (black smoke, ~20% intervals)
  - R0: no visible smoke
  - R1: light grey (20% opacity)
  - R2: medium grey (40%)
  - R3: dark grey (60%)
  - R4: very dark (80%)
  - R5: black smoke (100%)
- **Model options**: MobileNetV2, ResNet18, ResNet34, custom CNN
- **Dataset**: Roboflow "Ringlemann Test v1" — 253 images total
  - Train (222): R0=78, R1=12, R2=21, R3=45, R4=27, R5=39 (severely imbalanced)
  - Valid (15): R0=3, R1=0 (missing!), R2=1, R3=1, R4=2, R5=8
  - Test (16): R0=3, R1=1, R2=0 (missing!), R3=3, R4=6, R5=3
- **Class imbalance ratio**: 6.5× (R0=78 vs R1=12)
- **Key engineering fix**: `RinglemannImageFolder` with hardcoded `_RINGLEMANN_CLASS_TO_IDX` dict, ensuring R0→0, R1→1, ..., R5→5 regardless of which folders are present
- **Loss**: CrossEntropyLoss with inverse-frequency class weights
- **Metrics tracked**: Accuracy, MAE (ordinal metric — R0→R5 are ordered), ±1 tolerance accuracy

---

## 3. ALTERNATIVE OPACITY ESTIMATION APPROACHES

Because the supervised Ringlemann classifier has a very small labeled dataset, five additional physics/signal-based pipelines were developed that require NO training labels:

---

### 3A. SmokeScreen: Brightness-Ratio (Chen et al. formula)
**Location**: `physics_methods/smokescreen/`  
**Reference**: Chen et al. brightness-ratio Ringlemann grading  

**Pipeline**:
1. Segmentation: SAM3 text prompts (`"smoke"`, `"smoke plume"`, etc.) or HSV fallback
2. Region classification: dark smoke vs light smoke
3. Formula application:
   - **Dark smoke (Formula 1)**: `opacity = alpha * (1 - L_smoke / L_background)`
   - **Light smoke (Formula 2)**: `opacity = ((L_smoke - L_dark) / (L_bright - L_dark)) * 100`
4. Map opacity % → Ringlemann grade using thresholds [10, 30, 50, 70, 90]%

**Key parameters**:
- `alpha` calibration constant (default=100), tunable via `data/best_alpha.txt`
- HSV smoke detection: max saturation=50, min value=40 (grey/white smoke)
- SAM3 confidence threshold=0.3

**Results on 13 test images** (all used HSV fallback segmentation):
```
test_image        → R2 (fused_alpha ≈ 0.27)
test_image2       → R1 (fused_alpha ≈ 0.22)
test_image3       → R3 (fused_alpha ≈ 0.57, high confidence)
test_image4       → R3 (fused_alpha ≈ 0.59, high confidence)
test_image5       → R1 (fused_alpha ≈ 0.13)
bing_brick_00001565  → R1 (fused_alpha ≈ 0.22)
bing_brick_00000580  → R3 (fused_alpha ≈ 0.58, low confidence)
bing_brick_00000582  → R1 (fused_alpha ≈ 0.21)
bing_brick_00000583  → R2 (fused_alpha ≈ 0.45, high confidence)
bing_brick_winter_00001588 → R1 (fused_alpha ≈ 0.21)
bing_cabin_00001901  → R2 (fused_alpha ≈ 0.27)
bing_cabin_00001902  → R2 (fused_alpha ≈ 0.35, high confidence)
bing_cabin_00001903  → R2 (fused_alpha ≈ 0.45)
```
Results concentrated in R1–R3, consistent with the test image set being moderately smoky.

**Tests**: 46/46 passing

---

### 3B. Smoke Opacity: Dark Channel Prior (DCP) Fusion
**Location**: `physics_methods/smoke_opacity/`

**Physics**: The Dark Channel Prior (He et al. 2013) exploits the haze/smoke formation model. Transmittance `t` = fraction of light that passes through smoke, so opacity = 1 - t.

**Pipeline**:
```
Image
  → [Module 1] Segmentation (SAM3 / HSV) → binary smoke mask
  → [Module 2] Atmospheric light estimation (A) → brightest patch in dark channel
  → [Module 3a] DCP transmittance map → guided-filter refined (manually implemented,
               cv2.ximgproc unavailable)
  → [Module 3b] Contrast attenuation ratio (boundary vs interior)
  → [Module 3c] Color collapse score (smoke desaturates color)
  → [Module 4] Weighted fusion → fused_alpha → Ringlemann
```

**Fusion weights**: DCP=0.60, Contrast=0.25, Color collapse=0.15

**Ringlemann thresholds** (alpha → grade):
- R0: [0, 0.10), R1: [0.10, 0.25), R2: [0.25, 0.45), R3: [0.45, 0.65), R4: [0.65, 0.80), R5: [0.80, 1.0]

**Known limitation**: DCP tends to overestimate opacity at R0 (very light/no smoke); ±2 tolerance in tests.

**Tests**: 54/54 passing  
**End-to-end validation**: true_alpha=0.55 → estimated R3 (fused_alpha=0.553, confidence=high) ✓

---

### 3C. ViTMatte: Alpha Matting Pipeline
**Location**: `physics_methods/vitmatte/`

**Concept**: Alpha matting treats smoke as a semi-transparent foreground layer. ViTMatte (vision transformer) estimates per-pixel alpha values with fine boundary detail.

**Pipeline**:
```
Image
  → [Module 1] Segmentation (SAM3 / HSV) → binary + soft mask
  → [Module 2] Trimap generation (erosion → FG, dilation → unknown, rest → BG)
               Adaptive trimap based on soft mask confidence
  → [Module 3] ViTMatte alpha estimation
               (model: hustvl/vitmatte-small-composition-1k from HuggingFace)
               Bilateral smoothing post-processing
  → [Module 4] Validation (brightness correlation, edge coherence)
  → [Module 5] Ringlemann computation
               Methods: mean, p75, density_weighted (default), peak_region
```

**Key parameters**:
- Trimap erosion=10px, dilation=20px
- Adaptive trimap: auto-compute based on soft mask [0.2, 0.8] range
- Bilateral filter: d=5, sigma=0.1
- Ringlemann boundaries: [0.10, 0.30, 0.50, 0.70, 0.90]

**Known limitation (critical)**: ViTMatte was trained on solid opaque objects (composition-1k dataset), not translucent smoke. It systematically overshoots alpha — p75 and p90 are both ≈1.0 for most smoke images. True fractional alpha calibration is not achieved. ViTMatte is reliable for boundary sharpness, not absolute opacity values. Most images are rated R4–R5 regardless of actual density.

**ViTMatte results on 13 images** (all HSV segmentation):
```
bing_brick_00001565  → R5 (alpha=0.968, quality=medium)
bing_brick_00000580  → R5 (alpha=0.990, quality=high)
bing_brick_00000582  → R3 (alpha=0.700, quality=high) ← exception, thinner smoke
bing_brick_00000583  → R5 (alpha=0.973, quality=medium)
bing_brick_winter    → R4 (alpha=0.838, quality=high)
bing_cabin_00001901  → R5 (alpha=0.978, quality=high)
bing_cabin_00001902  → R4 (alpha=0.887, quality=high)
bing_cabin_00001903  → R5 (alpha=0.963, quality=high)
test_image           → R4 (alpha=0.889)
test_image3          → R3 (alpha=0.694)
test_image4          → R5 (alpha=0.994)
test_image5          → R4 (alpha=0.862)
```
ViTMatte consistently overestimates — nearly all images classified R4–R5.

**Tests**: 62/62 passing

---

### 3D. Temporal Reference: Stationary Camera Ground Truth
**Location**: `physics_methods/temporal_reference/`

**Concept**: Requires a clean reference image (no smoke) from the same camera angle. Uses image alignment (ORB feature matching + homography) then compares smoke image against ground truth at per-pixel level.

**Pipeline**:
```
Ground Truth Image + Smoke Images
  → Feature-based alignment (ORB, RANSAC homography)
  → SAM3 segmentation → region classification (smoke_dark, smoke_light, background_bright, background_dark)
  → Temporal opacity: compare brightness at smoke pixels vs same location in GT
  → Ringlemann mapping
```

**Formulas** (same as SmokeScreen):
- Dark smoke: `opacity = alpha * (1 - L_smoke / L_ground_truth)`
- Light smoke: `opacity = ((L_smoke - L_dark) / (L_bright - L_dark)) * 100`

**Test dataset**: 11 smoke images of a single chimney (`IMG_0826.JPG` through `IMG_0900.JPG`) plus `ground_truth.jpg`. All were real photographs taken from a stationary position. Located in `assets/temporal_reference_test/`.

**Temporal Reference results** (all alignment succeeded, 1137–1967 feature matches):
| Image | Smoke Type | Opacity % | Ringlemann | Formula |
|-------|------------|-----------|------------|---------|
| IMG_0826 | light | 48.22% | R2 | formula_2_light |
| IMG_0827 | light | 100.0% | R5 | formula_2_light |
| IMG_0829 | light | 51.25% | R3 | formula_2_light |
| IMG_0869 | light | 32.64% | R2 | formula_2_light |
| IMG_0872 | light | 62.74% | R3 | formula_2_light |
| IMG_0874 | light | 96.42% | R5 | formula_2_light |
| IMG_0885 | light | 47.25% | R2 | formula_2_light |
| IMG_0890 | light | 39.52% | R2 | formula_2_light |
| IMG_0894 | light | 34.89% | R2 | formula_2_light |
| IMG_0899 | light | 51.75% | R3 | formula_2_light |
| IMG_0900 | light | 43.38% | R2 | formula_2_light |

All smoke classified as "light" — formula 2 used throughout. Results span R2–R5, with most (7/11) in R2–R3 range.

---

### 3E. Gemini Inpainting: Synthetic Ground Truth
**Location**: `physics_methods/gemini_inpainting/`

**Concept**: When no ground truth reference image exists, uses Google Gemini AI to synthetically generate a "no-smoke" version of the image by inpainting the smoke region. The inpainted image serves as the reference for opacity estimation.

**Pipeline**:
```
Smoke Image
  → SAM3 segmentation → binary smoke mask
  → Smoke validation (min area=1.5%, min confidence=0.15)
  → Mask generation: pixel mask (8px dilation) + bbox mask (5% expand)
  → [Method 2a] Gemini inpainting with pixel mask
  → [Method 2b] Gemini inpainting with bbox mask
  → Opacity computation (same formulas as Temporal Reference)
  → Ringlemann mapping
```

**Gemini API models tried**:
- `gemini-3.1-flash-image-preview` (newest)
- `gemini-2.5-flash-image`
- `gemini-3-pro-image-preview`
- Fallback: OpenCV `cv2.inpaint` (TELEA algorithm, offline)

**Results** (only 1 image fully tested due to API costs):
```
bing_brick_chimney_smoke_00001565.jpg:
  smoke_present=True, smoke_type=light, coverage=42.4%
  Method 2a (pixel mask): opacity=24.72%, Ringlemann=2
  Method 2b (bbox mask): opacity=0.0%, Ringlemann=0
  Both inpainting methods succeeded.
```
Method 2b giving 0% suggests the bbox mask may have erased too much — a known limitation when the background color matches smoke.

**Config notes**:
- Requires `GEMINI_API_KEY` in `gemini_inpainting/.env` (see `.env.example`)
- Ringlemann thresholds: [20, 40, 60, 80]% (5-grade scheme, slightly different from other methods)
- SAM3 conf_threshold=0.15 (more permissive than temporal_reference's 0.3)

---

## 4. DATASETS SUMMARY

| Dataset | Purpose | Total | Train | Valid | Test |
|---------|---------|-------|-------|-------|------|
| Chimney Detection v1 | YOLO chimney detector | 90 images | 75 | 10 | 5 |
| Smoke Classification v1 | Binary smoke/no-smoke | 237 images | 207 | 20 | 10 |
| Ringlemann Test v1 | 6-class R0–R5 density | 253 images | 222 | 15 | 16 |

**Ringlemann class distribution (train)**:
- R0: 78 (35.1%) — most common
- R1: 12 (5.4%) — rarest
- R2: 21 (9.5%)
- R3: 45 (20.3%)
- R4: 27 (12.2%)
- R5: 39 (17.6%)
- Imbalance ratio: 6.5× (R0 to R1)

**Critical split issues**: Valid split missing R1; Test split missing R2. This required the custom `RinglemannImageFolder` fix to prevent label-shift.

---

## 5. TRAINING RESULTS

### 5A. Chimney Detection (YOLOv8 / YOLOv12)

**Best YOLOv8 run (exp13, v4 dataset)**:
- mAP50: **99.5%**, Precision: 98.0%, Recall: 100%

**Best YOLOv12 run (exp1, v6 dataset, 102 epochs)**:
- mAP50: **93.2%**, Precision: 91.9%, Recall: 90.7%

**Best YOLOv12 run (exp18, ~100 epochs)**:
- mAP50: **94.9%**, Precision: 100%, Recall: 90.5%

Training config:
- Epochs: 100–150, image size: 800×800, batch: 16
- Patience: 20–100, cosine LR, AMP enabled
- Pretrained on COCO (yolo12n.pt)

### 5B. Binary Smoke Classification (MobileNetV2)

**Binary Ringlemann (Ringlemann 0 → no_smoke, 1–5 → smoke)**:
- Best val accuracy: **100%**
- Test accuracy: **87.5%**
- Total epochs: 18 (early stopped)
- (Trained on Ringlemann dataset with binary labels)

### 5C. Ringlemann 6-Class Classification (MobileNetV2)

**Run with class weights (mobilenet_training)**:
- Best val accuracy: 53.3%, Best val MAE: 0.80
- Test accuracy: **50.0%**, Test MAE: **0.75**
- ±1 tolerance accuracy: **81.25%** (correctly within 1 grade of truth)
- Total epochs: 50

**Evaluation on saved model** (models/ringlemann_classification/mobilenet_best.pt):
- Accuracy: **25%** (4/16 correct)
- MAE: **1.5**
- ±1 tolerance: 56.25%, ±2 tolerance: 75%
- Macro F1: 0.204

**Per-class performance on test set**:
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| R0 | 1.00 | 0.33 | 0.50 | 3 |
| R1 | 0.00 | 0.00 | 0.00 | 1 |
| R2 | 0.00 | 0.00 | 0.00 | 0 (missing!) |
| R3 | 0.17 | 0.33 | 0.22 | 3 |
| R4 | 0.00 | 0.00 | 0.00 | 6 |
| R5 | 0.40 | 0.67 | 0.50 | 3 |

R4 is completely missed (6 test samples, 0 correct) — likely confused with R3 and R5. Worst errors: R1→R5 (error=4), R0→R3 (error=3).

**Custom CNN (custom_cnn_training)**:
- Test accuracy: **6.25%**, Test MAE: **1.75**, ±1 tolerance: 37.5%
- Much worse than MobileNetV2 on this tiny dataset

---

## 6. KEY TECHNICAL CHALLENGES AND FIXES

### 6.1 Label consistency across splits
**Problem**: `torchvision.datasets.ImageFolder` re-indexes classes 0..N based on which folders exist in each split. If R1 is absent from validation, it maps R2→0, R3→1, etc., silently corrupting MAE and confusion matrices.

**Fix**: `RinglemannImageFolder` with hardcoded `_RINGLEMANN_CLASS_TO_IDX` dict, ensuring R0→0, R1→1, ..., R5→5 regardless of which folders are present.

### 6.2 PyTorch 2.6 compatibility
- `ReduceLROnPlateau(verbose=True)` → removed `verbose=True` (deprecated in 2.6)
- `torch.load()` → requires `weights_only=False` for numpy-containing checkpoints

### 6.3 cv2.ximgproc unavailability (guided filter)
For the Dark Channel Prior pipeline, the guided filter from cv2.ximgproc was unavailable on the HPC. Manually implemented using `cv2.boxFilter` per He et al. 2013 equations.

### 6.4 ViTMatte overshooting alpha on smoke
ViTMatte was trained on solid composition objects. Smoke is semi-transparent with no definite foreground. Without putting a foreground (FG=255) region in the trimap, it underestimates; with FG, it overestimates. True fractional alpha calibration not achievable with this model.

### 6.5 SAM3 integration
SAM3 (Meta's Segment Anything Model 3) used as text-prompted segmenter throughout. Vendored locally at `sam3/` (not available on PyPI). Downloads model weights from HuggingFace automatically. Multiple text prompts tried in sequence for smoke/sky/background regions.

---

## 7. RESULTS SUMMARY TABLE

### Chimney Detection
| Model | Dataset | mAP50 | Precision | Recall |
|-------|---------|-------|-----------|--------|
| YOLOv8 | v4 | **99.5%** | 98.0% | 100% |
| YOLOv12 | v6 | 93.2–94.9% | 91–100% | 85–91% |

### Binary Smoke Classification
| Model | Val Acc | Test Acc | Epochs |
|-------|---------|----------|--------|
| MobileNetV2 (binary Ringlemann) | 100% | **87.5%** | 18 |

### 6-Class Ringlemann Classification (MobileNetV2)
| Metric | Val | Test |
|--------|-----|------|
| Accuracy | 53.3% | 25–50%* |
| MAE | 0.80 | 0.75–1.5* |
| ±1 Tolerance Acc | — | 56–81% |
| Macro F1 | — | 0.20 |

*Varies between training run (50%) and saved model evaluation (25%) — likely different model checkpoints tested.

### Physics-Based Pipelines (on 13 test images, no ground truth labels available)
| Method | Ringlemann Range | Notes |
|--------|-----------------|-------|
| DCP Fusion (`physics_methods/smoke_opacity/`) | R1–R3 | 3 high, 4 medium, 6 low confidence |
| SmokeScreen (`physics_methods/smokescreen/`) | R1–R3 | Consistent with DCP |
| ViTMatte (`physics_methods/vitmatte/`) | R3–R5 (mostly R4–R5) | Overestimates systematically |
| Temporal Reference (`physics_methods/temporal_reference/`) | R2–R5 | Strong alignment (1137–1967 matches) |
| Gemini Inpainting (`physics_methods/gemini_inpainting/`) | R2 (1 tested) | API dependency |

---

## 8. KEY MODELS AND LIBRARIES

- **Object detection**: Ultralytics YOLOv8/YOLOv12 (`ultralytics>=8.0.0`)
- **CNN backbone**: PyTorch MobileNetV2, ResNet18/34 (torchvision pretrained weights)
- **Segmentation**: SAM3 (Meta, vendored at `sam3/`, weights from HuggingFace), HSV color fallback
- **Alpha matting**: ViTMatte `hustvl/vitmatte-small-composition-1k` (HuggingFace transformers)
- **Dark Channel Prior**: Manual implementation using `cv2.boxFilter` (guided filter)
- **AI inpainting**: Google Gemini API (gemini-2.5-flash-image, gemini-3.1-flash-image-preview)
- **Feature matching**: ORB + RANSAC homography (cv2)
- **Framework**: PyTorch 2.6+, OpenCV 4.8+, scikit-learn 1.3+
- **Hardware**: NVIDIA Tesla V100-SXM3-32GB (32 GB VRAM)

---

## 9. TEST INFRASTRUCTURE

All pipelines have unit tests using pytest:
- `vitmatte/tests/`: 62 tests passing
- `smoke_opacity/tests/`: 54 tests passing
- `smokescreen/tests/`: 46 tests passing
- `temporal_reference/tests/`: opacity unit tests
- `gemini_inpainting/tests/`: opacity unit tests

Run all: `conda run -n smokescreen python -m pytest -v`

---

## 10. COMMANDS QUICK REFERENCE

```bash
# Environment
conda activate smokescreen

# Full 3-stage pipeline inference
python scripts/infer_smoke_pipeline.py assets/test_image.jpg

# Train Ringlemann (6-class)
python scripts/train_ringlemann.py --model-type mobilenet --use-class-weights --epochs 50

# Evaluate Ringlemann
python scripts/evaluate_ringlemann.py --model-type mobilenet

# DCP opacity pipeline
cd physics_methods && python -m smoke_opacity.main --input ../assets/test_image.jpg

# ViTMatte pipeline
cd physics_methods && python -m vitmatte.main --input ../assets/test_image.jpg

# Temporal Reference (needs ground_truth.jpg in folder)
cd physics_methods/temporal_reference && python run.py

# Gemini Inpainting (needs GEMINI_API_KEY in physics_methods/gemini_inpainting/.env)
cd physics_methods/gemini_inpainting && python run.py --input path/to/image.jpg

# SmokeScreen
cd physics_methods/smokescreen && python scripts/run_inference.py --input ../../assets/ --output-dir ../../outputs/smokescreen/

# Run all tests
conda run -n smokescreen python -m pytest -v
```

---

## 11. NEXT STEPS

1. **More labeled Ringlemann data** — 253 images across 6 classes is insufficient; need 1000+ per class
2. **Data augmentation** — synthetic smoke compositing, style transfer, Gaussian blur for opacity variation
3. **Calibrated alpha matting** — train a matting model on synthetic semi-transparent smoke
4. **Temporal methods at scale** — extend Temporal Reference to video streams with continuous GT reference
5. **Gemini inpainting evaluation** — test on a larger dataset with known Ringlemann grades
6. **Regression instead of classification** — predict continuous Ringlemann float rather than 6 discrete classes
7. **Multi-scale detection** — handle wide-angle shots where chimney is small
8. **Video pipeline** — temporal consistency across frames, not just per-frame classification
9. **Real-time deployment** — optimize for edge inference (TensorRT, ONNX export)
10. **Ground truth collection** — partner with regulators for labeled images with verified Ringlemann grades
