"""Central configuration for the ObjectLens pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .exceptions import ConfigError

SHAPE_CLASSES = (
    "circle", "ellipse", "triangle", "square",
    "rectangle", "pentagon", "hexagon", "star",
)

COLOR_NAMES = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
               "black", "gray", "white")


@dataclass(frozen=True)
class Config:
    """Immutable settings used by every stage of the pipeline."""

    max_dimension: int = 1024          # images larger than this are down-scaled
    blur_kernel: int = 5               # Gaussian kernel size (odd)
    use_clahe: bool = True             # contrast enhancement of the grey image
    min_area_px: int = 150             # ignore blobs smaller than this
    morph_kernel: int = 3              # kernel size for opening / closing
    watershed: bool = True             # split touching objects
    watershed_ratio: float = 0.55      # distance-transform threshold ratio
    min_confidence: float = 0.45       # below this a label becomes "unknown"
    # --- calibration (all optional) ---
    pixels_per_mm: Optional[float] = None   # direct calibration for ORIGINAL image
    ref_color: Optional[str] = None         # colour of the reference square
    ref_side_mm: Optional[float] = None     # real side length of the reference
    # --- paths ---
    model_path: str = "models/shape_classifier.joblib"
    db_path: str = "outputs/objectlens.db"

    def validate(self) -> "Config":
        if self.max_dimension < 64:
            raise ConfigError("max_dimension must be >= 64")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ConfigError("blur_kernel must be a positive odd integer")
        if self.morph_kernel < 1:
            raise ConfigError("morph_kernel must be >= 1")
        if self.min_area_px < 1:
            raise ConfigError("min_area_px must be >= 1")
        if not 0.1 <= self.watershed_ratio <= 0.95:
            raise ConfigError("watershed_ratio must be within [0.1, 0.95]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError("min_confidence must be within [0, 1]")
        if self.pixels_per_mm is not None and self.pixels_per_mm <= 0:
            raise ConfigError("pixels_per_mm must be positive")
        if (self.ref_color is None) != (self.ref_side_mm is None):
            raise ConfigError("ref_color and ref_side_mm must be given together")
        if self.ref_side_mm is not None and self.ref_side_mm <= 0:
            raise ConfigError("ref_side_mm must be positive")
        if self.ref_color is not None and self.ref_color not in COLOR_NAMES:
            raise ConfigError(f"unknown reference colour '{self.ref_color}'")
        return self
