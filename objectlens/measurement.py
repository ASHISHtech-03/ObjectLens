"""Stage 5 - calibration and real-world measurement (pixels -> millimetres)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .exceptions import CalibrationError
from .features import ObjectFeatures


@dataclass(frozen=True)
class Measurement:
    """Real-world size of an object."""

    length_mm: float          # long side of the min-area rectangle
    width_mm: float           # short side
    area_mm2: float
    perimeter_mm: float
    equivalent_diameter_mm: float   # diameter of a circle with the same area


def pixels_per_mm_from_reference(side_px: float, side_mm: float) -> float:
    """Scale factor from a reference object of known side length."""
    if side_px <= 0 or side_mm <= 0:
        raise CalibrationError("Reference sizes must be positive")
    return side_px / side_mm


def find_reference(features: Sequence[ObjectFeatures], labels: Sequence[str],
                   color: str) -> int:
    """Index of the reference square: a 'square' with the requested colour.
    If several match, the largest one is used."""
    candidates = [i for i, (f, l) in enumerate(zip(features, labels))
                  if l == "square" and f.color_name == color]
    if not candidates:
        raise CalibrationError(
            f"No {color} square found in the image to use as a calibration reference")
    return max(candidates, key=lambda i: features[i].area)


def reference_side_px(f: ObjectFeatures) -> float:
    """Side length (px) of a square from its area and bounding rectangle."""
    return 0.5 * (math.sqrt(f.area) + 0.5 * (f.rect_size[0] + f.rect_size[1]))


def measure(f: ObjectFeatures, pixels_per_mm: Optional[float]) -> Optional[Measurement]:
    """Convert features to millimetres, or ``None`` when uncalibrated."""
    if pixels_per_mm is None:
        return None
    if pixels_per_mm <= 0:
        raise CalibrationError("pixels_per_mm must be positive")
    k = 1.0 / pixels_per_mm
    return Measurement(
        length_mm=f.rect_size[0] * k, width_mm=f.rect_size[1] * k,
        area_mm2=f.area * k * k, perimeter_mm=f.perimeter * k,
        equivalent_diameter_mm=2.0 * math.sqrt(f.area / math.pi) * k,
    )
