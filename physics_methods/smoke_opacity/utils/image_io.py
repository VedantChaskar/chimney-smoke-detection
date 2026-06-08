"""
Image and video I/O utilities.
"""

from pathlib import Path
from typing import Optional, Generator
import numpy as np
import cv2


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image as a BGR uint8 numpy array (OpenCV convention).

    Args:
        path: Path to the image file.

    Returns:
        (H, W, 3) uint8 BGR array.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if cv2 cannot decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cv2 could not read image: {path}")
    return img


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save a BGR numpy array to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR (OpenCV)."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def extract_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """
    Extract a single frame from a video file.

    Args:
        video_path: Path to the video.
        frame_index: 0-based frame index.

    Returns:
        (H, W, 3) uint8 BGR array.

    Raises:
        FileNotFoundError: if video does not exist.
        ValueError: if frame cannot be read.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raise ValueError(
                f"Could not read frame {frame_index} "
                f"(video has {total} frames)"
            )
        return frame
    finally:
        cap.release()


def iter_frames(
    video_path: str | Path,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Iterate over video frames, yielding (frame_index, frame_bgr).

    Args:
        video_path: Path to video.
        start: First frame index (inclusive).
        end: Last frame index (exclusive); None → end of video.
        step: Frame step (default 1).

    Yields:
        (index, frame_bgr) tuples.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while idx < end:
            ret, frame = cap.read()
            if not ret:
                break
            if (idx - start) % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def collect_images(folder: str | Path, extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    """Return sorted list of image paths in a folder."""
    folder = Path(folder)
    paths = []
    for ext in extensions:
        paths.extend(folder.glob(f"*{ext}"))
        paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(paths))
