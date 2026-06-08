"""Inference pipeline and utilities"""

from .chimney_detector import ChimneyDetector
from .smoke_classifier import SmokeClassifier
from .ringlemann_classifier import RinglemannClassifier
from .pipeline import SmokeDetectionPipeline

# SAM3 detector - import only when needed to avoid dependency issues
def get_sam3_detector(*args, **kwargs):
    """Lazy import for SAM3ChimneyDetector to avoid import errors when SAM3 is not installed."""
    from .sam3_detector import SAM3ChimneyDetector
    return SAM3ChimneyDetector(*args, **kwargs)

__all__ = [
    'ChimneyDetector',
    'SmokeClassifier',
    'RinglemannClassifier',
    'SmokeDetectionPipeline',
    'get_sam3_detector'
]
