"""Stage 2 - foreground segmentation and splitting of touching objects.

Strategy
--------
1. Estimate the background colour from the image border (median in CIE-Lab).
2. Compute the per-pixel Lab distance to that colour -> robust to object colour.
3. Threshold with Otsu (clamped to a sane range) and clean up with morphology.
4. Split touching objects with the distance transform + watershed.
5. Extract one external contour per object.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .config import Config
from .preprocessing import PreprocessResult

log = logging.getLogger(__name__)

MIN_CONTRAST = 12.0          # below this Lab distance the image is "empty"
THRESH_RANGE = (15.0, 40.0)  # clamp for the Otsu threshold (Lab units)
SPLIT_SOLIDITY = 0.96        # only try to split blobs less convex than this


@dataclass
class Region:
    """One segmented object."""

    contour: np.ndarray    # (N,1,2) int32 external contour
    mask: np.ndarray       # full-size boolean mask
    area: float

    @property
    def bbox(self):
        return cv2.boundingRect(self.contour)


@dataclass
class SegmentationResult:
    mask: np.ndarray       # cleaned binary mask (uint8 0/255)
    regions: List[Region]


def background_distance(blurred_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel Lab distance from the median border colour."""
    lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]])
    background = np.median(border, axis=0)
    return np.linalg.norm(lab - background, axis=2)


def binarize(distance: np.ndarray) -> np.ndarray:
    """Threshold the distance map into a binary foreground mask (0/255)."""
    peak = float(distance.max())
    if peak < MIN_CONTRAST:
        return np.zeros(distance.shape, np.uint8)
    d8 = np.clip(distance / peak * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    real = float(otsu) / 255.0 * peak
    thr = float(np.clip(real, *THRESH_RANGE))
    return (distance > thr).astype(np.uint8) * 255


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphological opening (remove speckle) then closing (fill pin-holes)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _components(mask: np.ndarray) -> List[np.ndarray]:
    n, labels = cv2.connectedComponents(mask)
    return [(labels == lab).astype(np.uint8) * 255 for lab in range(1, n)]


def _split_component(comp: np.ndarray, ratio: float) -> List[np.ndarray]:
    """Split one connected component using distance transform + watershed."""
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area <= 0 or area / hull_area >= SPLIT_SOLIDITY:
        return [comp]                       # convex enough -> single object

    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    peak = float(dist.max())
    # Raise the threshold step by step until the "sure foreground" separates into
    # several cores (a weak neck between objects needs a higher ratio). Tiny noise
    # cores (< 15 % of the biggest one) are ignored so they cannot create slivers.
    sure_fg, markers, n_cores = None, None, 0
    for r in np.arange(ratio, 0.86, 0.05):
        _, fg = cv2.threshold(dist, float(r) * peak, 255, cv2.THRESH_BINARY)
        fg = fg.astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            continue
        keep = [i + 1 for i, a in enumerate(areas) if a >= max(5, 0.15 * areas.max())]
        if len(keep) >= 2:
            sure_fg = np.zeros_like(fg)
            markers = np.ones(comp.shape, np.int32)          # 1 = background
            for k, idx in enumerate(keep):
                sure_fg[lab == idx] = 255
                markers[lab == idx] = k + 2
            n_cores = len(keep)
            break
    if sure_fg is None:
        return [comp]

    unknown = cv2.subtract(comp, sure_fg)
    markers[unknown == 255] = 0
    cv2.watershed(cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR), markers)

    kernel = np.ones((3, 3), np.uint8)
    parts = []
    for label in range(2, n_cores + 2):
        sub = (markers == label).astype(np.uint8) * 255
        sub = cv2.bitwise_and(cv2.dilate(sub, kernel), comp)  # recover boundary pixel
        if cv2.countNonZero(sub) > 0:
            parts.append(sub)
    return parts or [comp]


def split_touching(mask: np.ndarray, ratio: float) -> List[np.ndarray]:
    """Return one binary mask per object, splitting touching blobs."""
    parts: List[np.ndarray] = []
    for comp in _components(mask):
        parts.extend(_split_component(comp, ratio))
    return parts


def refine_mask(distance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shrink a (slightly over-grown) object mask to its half-maximum edge.

    The global threshold is deliberately low so dim objects are not lost; this
    per-object step moves the boundary to 50 % of the object's own contrast,
    which is the unbiased edge position of a blurred step edge.
    """
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if cv2.countNonZero(inner) == 0:
        inner = mask
    level = float(np.median(distance[inner > 0]))
    if level < MIN_CONTRAST:
        return mask
    refined = ((distance > 0.5 * level) & (mask > 0)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def extract_regions(masks: List[np.ndarray], min_area: int) -> List[Region]:
    """Convert masks to Region objects, discarding tiny blobs."""
    regions: List[Region] = []
    for m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        regions.append(Region(contour=cnt, mask=m > 0, area=area))
    # deterministic order: top-to-bottom, left-to-right
    regions.sort(key=lambda r: (round(r.bbox[1] / 40), r.bbox[0]))
    return regions


def segment(pre: PreprocessResult, cfg: Config) -> SegmentationResult:
    """Run the full segmentation stage."""
    distance = background_distance(pre.blurred)
    mask = clean_mask(binarize(distance), cfg.morph_kernel)
    masks = split_touching(mask, cfg.watershed_ratio) if cfg.watershed else _components(mask)
    masks = [refine_mask(distance, m) for m in masks]
    regions = extract_regions(masks, cfg.min_area_px)
    log.debug("Segmentation found %d region(s)", len(regions))
    return SegmentationResult(mask=mask, regions=regions)
