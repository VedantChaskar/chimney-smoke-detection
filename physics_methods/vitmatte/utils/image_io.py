"""Image and video I/O utilities."""

from pathlib import Path
from typing import Generator, Optional
import numpy as np
import cv2


def load_image(path: str | Path) -> np.ndarray:
    """Load image as BGR uint8. Raises FileNotFoundError / ValueError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cv2 could not read: {path}")
    return img


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_alpha_png(alpha: np.ndarray, path: str | Path) -> None:
    """Save float32 alpha [0,1] as 16-bit PNG for lossless storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha16 = (np.clip(alpha, 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(str(path), alpha16)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def extract_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raise ValueError(f"Cannot read frame {frame_index} (video has {total} frames)")
        return frame
    finally:
        cap.release()


def collect_images(
    folder: str | Path,
    extensions=(".jpg", ".jpeg", ".png", ".bmp"),
) -> list[Path]:
    folder = Path(folder)
    paths = []
    for ext in extensions:
        paths.extend(folder.glob(f"*{ext}"))
        paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def resize_keep_aspect(
    image: np.ndarray,
    max_size: int,
    interpolation=cv2.INTER_LINEAR,
) -> tuple[np.ndarray, float]:
    """
    Resize image so the longest edge equals max_size.

    Returns:
        resized image, scale factor (new/old)
    """
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return image.copy(), 1.0
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation), scale
