"""
Centralized Configuration for Chimney Smoke Detection

This module contains all paths, constants, and configuration settings
used throughout the project. This eliminates hard-coded paths and makes
the codebase more maintainable and portable.
"""

from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# DATA DIRECTORIES
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"

# Chimney Detection Datasets
CHIMNEY_DETECTION_DIR = DATA_DIR / "chimney_detection"
CHIMNEY_DATASET_V1 = CHIMNEY_DETECTION_DIR / "v1"
CHIMNEY_DATASET_V3 = CHIMNEY_DETECTION_DIR / "v3"
CHIMNEY_DATASET_V4 = CHIMNEY_DETECTION_DIR / "v4"
CHIMNEY_DATASET_V6 = CHIMNEY_DETECTION_DIR / "v6_yolo12"

# Smoke Classification Datasets
SMOKE_CLASSIFICATION_DIR = DATA_DIR / "smoke_classification"
SMOKE_DATASET_V1 = SMOKE_CLASSIFICATION_DIR / "v1"

# Ringlemann Classification Datasets
RINGLEMANN_DATASET_DIR = DATA_DIR / "ringlemann_classification"
RINGLEMANN_DATASET_V1 = DATA_DIR / "Ringlemann Test.v1i.folder"

# Raw and Processed Data
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHIMNEY_CROPS_DIR = PROCESSED_DATA_DIR / "chimney_crops"

# ============================================================================
# MODEL DIRECTORIES
# ============================================================================
MODELS_DIR = PROJECT_ROOT / "models"

# Pretrained Models
PRETRAINED_DIR = MODELS_DIR / "pretrained"
YOLOV8_PRETRAINED = PRETRAINED_DIR / "yolov8n.pt"
YOLO11_PRETRAINED = PRETRAINED_DIR / "yolo11n.pt"
YOLO12_PRETRAINED = PRETRAINED_DIR / "yolo12n.pt"
YOLO26_PRETRAINED = PRETRAINED_DIR / "yolo26n.pt"

# Trained Chimney Detection Models
CHIMNEY_MODELS_DIR = MODELS_DIR / "chimney_detection"
CHIMNEY_MODEL_YOLOV8 = CHIMNEY_MODELS_DIR / "yolov8_best.pt"
CHIMNEY_MODEL_YOLOV12 = CHIMNEY_MODELS_DIR / "yolov12_best.pt"
CHIMNEY_MODEL_YOLO26 = CHIMNEY_MODELS_DIR / "yolo26_best.pt"

# YOLOv12 Experiment Checkpoint (exp1 - best performing)
CHIMNEY_MODEL_YOLOV12_EXP1 = PROJECT_ROOT / "experiments" / "chimney_detection_yolov12" / "exp1" / "weights" / "best.pt"

# SAM3 Model Path (downloads from HuggingFace by default)
SAM3_ROOT = PROJECT_ROOT / "sam3"
SAM3_CHECKPOINT = None  # None means download from HuggingFace

# Trained Smoke Classification Models
SMOKE_MODELS_DIR = MODELS_DIR / "smoke_classification"
SMOKE_MODEL_MOBILENET = SMOKE_MODELS_DIR / "mobilenet_best.pt"
SMOKE_MODEL_CNN = SMOKE_MODELS_DIR / "cnn_best.pt"

# Trained Ringlemann Classification Models
RINGLEMANN_MODELS_DIR = MODELS_DIR / "ringlemann_classification"
RINGLEMANN_MODEL_MOBILENET = RINGLEMANN_MODELS_DIR / "mobilenet_best.pt"
RINGLEMANN_MODEL_RESNET18 = RINGLEMANN_MODELS_DIR / "resnet18_best.pt"
RINGLEMANN_MODEL_CNN = RINGLEMANN_MODELS_DIR / "custom_cnn_best.pt"

# Binary smoke/no-smoke models trained on Ringlemann dataset
RINGLEMANN_BINARY_MODELS_DIR = MODELS_DIR / "ringlemann_binary"
RINGLEMANN_BINARY_MODEL_MOBILENET = RINGLEMANN_BINARY_MODELS_DIR / "mobilenet_best.pt"
RINGLEMANN_BINARY_MODEL_RESNET18 = RINGLEMANN_BINARY_MODELS_DIR / "resnet18_best.pt"


# ============================================================================
# MODEL PATH UTILITIES
# ============================================================================

def get_pretrained_model_path(model_name: str) -> Path:
    """
    Get absolute path for pretrained model.

    This ensures the Ultralytics YOLO library receives an absolute path,
    preventing it from caching model files in the current working directory.

    Args:
        model_name: Model identifier ('yolo11', 'yolo12', 'yolov8')

    Returns:
        Path: Absolute path to the pretrained model file

    Raises:
        ValueError: If model_name is not recognized
    """
    model_map = {
        'yolo11': YOLO11_PRETRAINED,
        'yolo12': YOLO12_PRETRAINED,
        'yolo26': YOLO26_PRETRAINED,
        'yolov8': YOLOV8_PRETRAINED,
    }

    model_path = model_map.get(model_name.lower())
    if not model_path:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: {', '.join(model_map.keys())}"
        )

    # Return absolute path to prevent CWD caching
    return model_path.resolve()


# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_RESULTS_DIR = OUTPUT_DIR / "results"

# ============================================================================
# LOG DIRECTORIES
# ============================================================================
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# EXPERIMENT DIRECTORIES
# ============================================================================
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHIMNEY_YOLOV8_EXPERIMENTS = EXPERIMENTS_DIR / "chimney_detection_yolov8"
CHIMNEY_YOLOV12_EXPERIMENTS = EXPERIMENTS_DIR / "chimney_detection_yolov12"
CHIMNEY_YOLO26_EXPERIMENTS = EXPERIMENTS_DIR / "chimney_detection_yolo26"
SMOKE_EXPERIMENTS = EXPERIMENTS_DIR / "smoke_classification"
RINGLEMANN_EXPERIMENTS = EXPERIMENTS_DIR / "ringlemann_classification"

# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

class ChimneyDetectionConfig:
    """Configuration for chimney detection training"""
    EPOCHS = 100
    IMG_SIZE = 800
    BATCH_SIZE = 16
    PATIENCE = 100
    CONF_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.5

    # Dataset paths
    DEFAULT_DATASET = CHIMNEY_DATASET_V4
    DATASET_YAML = DEFAULT_DATASET / "data.yaml"

    # Model paths
    DEFAULT_PRETRAINED = YOLOV8_PRETRAINED
    EXPERIMENT_DIR = CHIMNEY_YOLOV8_EXPERIMENTS


class SmokeClassificationConfig:
    """Configuration for smoke classification training"""
    EPOCHS = 30
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    CONF_THRESHOLD = 0.5

    # Dataset paths
    DEFAULT_DATASET = SMOKE_DATASET_V1
    TRAIN_DIR = DEFAULT_DATASET / "train"
    VALID_DIR = DEFAULT_DATASET / "valid"
    TEST_DIR = DEFAULT_DATASET / "test"

    # Model paths
    EXPERIMENT_DIR = SMOKE_EXPERIMENTS


class RinglemannClassificationConfig:
    """Configuration for Ringlemann scale classification (0-5)"""
    EPOCHS = 50
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience
    NUM_CLASSES = 6  # R0, R1, R2, R3, R4, R5

    # Ringlemann class names
    CLASS_NAMES = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']

    # Dataset paths
    DEFAULT_DATASET = RINGLEMANN_DATASET_V1
    TRAIN_DIR = DEFAULT_DATASET / "train"
    VALID_DIR = DEFAULT_DATASET / "valid"
    TEST_DIR = DEFAULT_DATASET / "test"

    # Model paths
    MODEL_SAVE_DIR = RINGLEMANN_MODELS_DIR
    EXPERIMENT_DIR = RINGLEMANN_EXPERIMENTS

    # Evaluation thresholds
    GOOD_MAE_THRESHOLD = 0.5  # MAE < 0.5 is good performance
    ACCEPTABLE_MAE_THRESHOLD = 1.0  # MAE < 1.0 is acceptable


class SAM3DetectionConfig:
    """Configuration for SAM3 chimney detection (inference)"""
    IMG_SIZE = 1008  # SAM3 uses 1008x1008 resolution
    CONF_THRESHOLD = 0.3  # Higher default for SAM3
    TEXT_PROMPT = "chimney"  # Default text prompt

    # Model settings
    LOAD_FROM_HF = True  # Download from HuggingFace
    CHECKPOINT = SAM3_CHECKPOINT


class InferencePipelineConfig:
    """Configuration for the three-stage inference pipeline (chimney → smoke → Ringlemann)"""
    # Stage 1: Chimney Detection
    CHIMNEY_MODEL = CHIMNEY_MODEL_YOLOV8  # or CHIMNEY_MODEL_YOLOV12
    CHIMNEY_CONF_THRESHOLD = 0.15
    CHIMNEY_IMG_SIZE = 800

    # Stage 2: Smoke Classification
    SMOKE_MODEL = SMOKE_MODEL_MOBILENET  # or SMOKE_MODEL_CNN
    SMOKE_CONF_THRESHOLD = 0.5
    SMOKE_IMG_SIZE = 224

    # Chimney crop padding scales (proportional to bounding box size)
    # Smoke rises upward, so top padding is the most important
    CROP_PAD_TOP_SCALE = 1.5      # pad_top = box_height * 1.5
    CROP_PAD_BOTTOM_SCALE = 0.2   # pad_bottom = box_height * 0.2
    CROP_PAD_SIDE_SCALE = 0.3     # pad_sides = box_width * 0.3

    # Output settings
    SAVE_ANNOTATED = True
    OUTPUT_DIR = OUTPUT_IMAGES_DIR


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device():
    """
    Get the appropriate device for training/inference

    Returns:
        str: 'cuda' if GPU is available, 'cpu' otherwise
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


DEVICE = get_device()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path):
    """
    Ensure a directory exists, create it if it doesn't

    Args:
        path: Path to directory (Path object or string)

    Returns:
        Path: The directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_experiment(experiment_dir):
    """
    Get the latest experiment folder in an experiment directory

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Path: Path to latest experiment, or None if no experiments found
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        return None

    # Find all exp* folders
    exp_folders = sorted(
        [f for f in experiment_dir.iterdir() if f.is_dir() and f.name.startswith('exp')],
        key=lambda x: int(x.name[3:]) if x.name[3:].isdigit() else 0,
        reverse=True
    )

    return exp_folders[0] if exp_folders else None


# ============================================================================
# CONSTANTS
# ============================================================================

# Class names for datasets
CHIMNEY_CLASSES = ['chimney']
SMOKE_CLASSES = ['no_smoke', 'smoke']
RINGLEMANN_CLASSES = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']

# Image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Model file extensions
MODEL_EXTENSIONS = ['.pt', '.pth', '.weights']
