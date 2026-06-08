import pytest
from pathlib import Path


def test_project_imports():
    from src.config import PROJECT_ROOT, InferencePipelineConfig
    assert PROJECT_ROOT.exists()


def test_assets_exist():
    from src.config import PROJECT_ROOT
    assets = list((PROJECT_ROOT / "assets").glob("*.jpg"))
    assert len(assets) > 0, "assets/ must contain at least one .jpg for smoke-test images"


def test_config_model_paths_are_absolute():
    from src.config import InferencePipelineConfig
    assert hasattr(InferencePipelineConfig, 'CHIMNEY_MODEL')
    assert hasattr(InferencePipelineConfig, 'SMOKE_MODEL')


def test_config_constants():
    from src.config import (
        PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR,
        RINGLEMANN_DATASET_V1,
    )
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert MODELS_DIR == PROJECT_ROOT / "models"
    assert OUTPUT_DIR == PROJECT_ROOT / "outputs"


def test_ringlemann_config():
    from src.config import RinglemannClassificationConfig
    assert RinglemannClassificationConfig.NUM_CLASSES == 6
    assert len(RinglemannClassificationConfig.CLASS_NAMES) == 6
