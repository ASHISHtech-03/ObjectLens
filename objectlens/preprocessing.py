"""Stage 1 - image loading, validation and pre-processing."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from .config import Config
from .exceptions import ImageLoadError

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PreprocessResult:
    """All intermediate images produced by pre-processing."""

    color: np.ndarray      # (possibly resized) BGR image
    blurred: np.ndarray    # Gaussian-smoothed BGR image
    gray: np.ndarray       # grey-scale version
    enhanced: np.ndarray   # CLAHE-enhanced grey image
    scale: float           # resize factor applied to the original


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk as BGR ``uint8``; validate path & format."""
    p = Path(path)
    if not p.is_file():
        raise ImageLoadError(f"Image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Unsupported image format '{p.suffix}'. "
                             f"Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageLoadError(f"Could not decode image: {p}")
    return image


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Normalise any array into a 3-channel uint8 BGR image."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ImageLoadError("Input is not a valid non-empty image array")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ImageLoadError(f"Unsupported image shape {image.shape}")


def resize_max_dimension(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Down-scale so the longer side is at most ``max_dim``. Returns (image, scale)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


def denoise(image: np.ndarray, kernel: int) -> np.ndarray:
    """Gaussian smoothing to suppress sensor noise."""
    if kernel <= 1:
        return image.copy()
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def preprocess(image: np.ndarray, cfg: Config) -> PreprocessResult:
    """Run the full pre-processing stage."""
    image = ensure_bgr(image)
    color, scale = resize_max_dimension(image, cfg.max_dimension)
    blurred = denoise(color, cfg.blur_kernel)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_contrast(gray) if cfg.use_clahe else gray.copy()
    log.debug("Pre-processed image %s (scale=%.3f)", color.shape, scale)
    return PreprocessResult(color=color, blurred=blurred, gray=gray,
                            enhanced=enhanced, scale=scale)
