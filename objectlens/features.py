"""Stage 3 - shape and colour feature extraction for each segmented region."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from .segmentation import Region

FEATURE_NAMES = ("circularity", "aspect_ratio", "extent", "solidity",
                 "vertices", "hull_vertices", "hu1", "hu2")


@dataclass
class ObjectFeatures:
    """Geometric and photometric descriptors of one object (pixel units)."""

    area: float
    perimeter: float
    circularity: float            # 4*pi*A / P^2, 1.0 for a perfect circle
    aspect_ratio: float           # long side / short side of min-area rectangle
    extent: float                 # area / min-area-rectangle area
    solidity: float               # area / convex-hull area
    vertices: int                 # polygon approximation vertices
    hull_vertices: int            # vertices of the approximated convex hull
    hu: Tuple[float, ...]         # log-scaled Hu moments (7)
    centroid: Tuple[float, float]
    rect_size: Tuple[float, float]   # (long side, short side) in pixels
    rect_angle: float
    mean_bgr: Tuple[float, float, float]
    color_name: str
    bbox: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))

    def vector(self) -> np.ndarray:
        """Scale-invariant feature vector fed to the classifier."""
        return np.array([self.circularity, self.aspect_ratio, self.extent,
                         self.solidity, float(self.vertices), float(self.hull_vertices),
                         self.hu[0], self.hu[1]], dtype=np.float64)


def color_name_from_bgr(bgr: Tuple[float, float, float]) -> str:
    """Map a mean BGR colour to a human readable name via HSV thresholds."""
    pixel = np.uint8([[[int(round(c)) for c in bgr]]])
    h, s, v = [int(x) for x in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]]
    if s < 40:
        if v < 60:
            return "black"
        return "white" if v > 190 else "gray"
    if h < 8 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 36:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 160:
        return "purple"
    return "pink"


def _log_hu(hu: np.ndarray) -> Tuple[float, ...]:
    return tuple(float(-math.copysign(1.0, h) * math.log10(abs(h) + 1e-30)) for h in hu)


def extract_features(region: Region, color_image: np.ndarray) -> ObjectFeatures:
    """Compute :class:`ObjectFeatures` for one region."""
    cnt = region.contour
    perimeter = float(cv2.arcLength(cnt, True))
    # After half-maximum edge refinement the pixel-centre contour is an
    # (empirically) unbiased estimate of the object area - see docs/results.
    area = float(cv2.contourArea(cnt))
    raw_area = area
    circularity = min(1.0, 4.0 * math.pi * raw_area / (perimeter ** 2)) if perimeter > 0 else 0.0

    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    long_side, short_side = max(w, h), min(w, h)
    aspect = long_side / short_side if short_side > 0 else 1.0
    extent = area / (long_side * short_side) if short_side > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = raw_area / hull_area if hull_area > 0 else 0.0

    eps = 0.03 * perimeter
    vertices = len(cv2.approxPolyDP(cnt, eps, True))
    hull_vertices = len(cv2.approxPolyDP(hull, eps, True))

    moments = cv2.moments(cnt)
    if moments["m00"] > 0:
        cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    else:
        cx, cy = rect[0]
    hu = _log_hu(cv2.HuMoments(moments).flatten())

    mask = region.mask.astype(np.uint8) * 255
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
    if cv2.countNonZero(eroded) == 0:
        eroded = mask
    mean = cv2.mean(color_image, mask=eroded)[:3]
    mean_bgr = (float(mean[0]), float(mean[1]), float(mean[2]))

    return ObjectFeatures(
        area=area, perimeter=perimeter, circularity=circularity,
        aspect_ratio=float(aspect), extent=float(min(extent, 1.0)),
        solidity=float(min(solidity, 1.0)), vertices=vertices,
        hull_vertices=hull_vertices, hu=hu, centroid=(float(cx), float(cy)),
        rect_size=(float(long_side), float(short_side)), rect_angle=float(rect[2]),
        mean_bgr=mean_bgr, color_name=color_name_from_bgr(mean_bgr),
        bbox=tuple(int(v) for v in region.bbox),
    )


def extract_all(regions: List[Region], color_image: np.ndarray) -> List[ObjectFeatures]:
    """Extract features for every region."""
    return [extract_features(r, color_image) for r in regions]
