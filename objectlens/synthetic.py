"""Synthetic data generation.

Produces annotated scenes with known ground truth (shape, colour, exact area)
so the system can be trained, tested and evaluated without a manual dataset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import SHAPE_CLASSES, Config
from .features import extract_features
from .preprocessing import preprocess
from .segmentation import segment

# BGR palette (name -> colour)
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "red": (40, 40, 220), "blue": (200, 80, 30), "orange": (0, 140, 255),
    "yellow": (0, 220, 240), "purple": (160, 50, 140), "green": (60, 170, 60),
}

BACKGROUNDS = {"light": (238, 238, 238), "dark": (35, 35, 35), "cream": (225, 235, 245)}


@dataclass
class TruthObject:
    """Ground-truth description of a rendered object."""

    label: str
    center: Tuple[float, float]
    radius: float
    area_px: float
    color: str
    is_reference: bool = False


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def shape_points(kind: str, r: float, rot_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Vertices (centred on the origin) of the requested shape."""
    rot = math.radians(rot_deg)
    if kind == "circle":
        t = np.linspace(0, 2 * math.pi, 180, endpoint=False)
        pts = np.stack([0.9 * r * np.cos(t), 0.9 * r * np.sin(t)], axis=1)
    elif kind == "ellipse":
        t = np.linspace(0, 2 * math.pi, 180, endpoint=False)
        a = 0.95 * r
        b = a / rng.uniform(1.35, 2.4)
        pts = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
    elif kind in ("triangle", "square", "pentagon", "hexagon"):
        n = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}[kind]
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    elif kind == "rectangle":
        w = 1.7 * r
        h = w / rng.uniform(1.5, 3.0)
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    elif kind == "star":
        inner = rng.uniform(0.38, 0.5)
        t = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        radii = np.where(np.arange(10) % 2 == 0, r, inner * r)
        pts = np.stack([radii * np.cos(t), radii * np.sin(t)], axis=1)
    else:
        raise ValueError(f"unknown shape '{kind}'")
    rot_m = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    return pts @ rot_m.T


def draw_shape(img: np.ndarray, kind: str, center: Tuple[float, float], r: float,
               color: Tuple[int, int, int], rot_deg: float,
               rng: np.random.Generator) -> float:
    """Draw an anti-aliased filled shape; returns its exact area in pixels."""
    pts = shape_points(kind, r, rot_deg, rng) + np.array(center)
    fixed = np.round(pts * 16).astype(np.int32)
    cv2.fillPoly(img, [fixed], color, lineType=cv2.LINE_AA, shift=4)
    return _polygon_area(pts)


def _background(rng: np.random.Generator, size: Tuple[int, int], theme: str) -> np.ndarray:
    h, w = size
    img = np.full((h, w, 3), BACKGROUNDS[theme], np.float32)
    gradient = np.linspace(rng.uniform(0.94, 1.0), 1.0, w, dtype=np.float32)[None, :, None]
    return img * gradient


def _finish(img: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0, noise, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def render_single(kind: str, rng: np.random.Generator, canvas: int = 256) -> np.ndarray:
    """One random shape centred on a noisy canvas (used for training data)."""
    theme = rng.choice(list(BACKGROUNDS))
    img = _background(rng, (canvas, canvas), theme)
    color = PALETTE[rng.choice(list(PALETTE))]
    r = rng.uniform(28, 95)
    jitter = rng.uniform(-8, 8, size=2)
    draw_shape(img, kind, (canvas / 2 + jitter[0], canvas / 2 + jitter[1]), r, color,
               rng.uniform(0, 360), rng)
    return _finish(img, rng, rng.uniform(2, 9))


def render_scene(seed: int, n_objects: int = 8, size: Tuple[int, int] = (480, 640),
                 shapes: Optional[Sequence[str]] = None, touching_pairs: int = 0,
                 reference: Optional[Tuple[float, float]] = None,
                 theme: Optional[str] = None, noise: float = 5.0):
    """Render a multi-object scene.

    ``reference`` = (side_mm, pixels_per_mm) adds a green reference square in
    the top-left corner. Returns ``(image, [TruthObject, ...])``.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    theme = theme or str(rng.choice(["light", "dark", "cream"]))
    img = _background(rng, size, theme)
    shapes = list(shapes or SHAPE_CLASSES)
    colors = {k: v for k, v in PALETTE.items() if not (reference and k == "green")}
    truths: List[TruthObject] = []
    placed: List[Tuple[float, float, float]] = []     # x, y, radius

    if reference:
        side_mm, ppm = reference
        side = side_mm * ppm
        c = (side * 0.5 + 18, side * 0.5 + 18)
        area = draw_shape(img, "square", c, side / math.sqrt(2), PALETTE["green"], 45, rng)
        truths.append(TruthObject("square", c, side / math.sqrt(2), area, "green", True))
        placed.append((c[0], c[1], side * 0.75))

    def free(x, y, r, margin=10):
        if x - r < 8 or y - r < 8 or x + r > w - 8 or y + r > h - 8:
            return False
        return all(math.hypot(x - px, y - py) > r + pr + margin for px, py, pr in placed)

    # touching circle pairs (to exercise watershed splitting)
    for _ in range(touching_pairs):
        for _try in range(300):
            r = rng.uniform(26, 38)
            x, y = rng.uniform(r * 2, w - r * 2), rng.uniform(r * 2, h - r * 2)
            ang = rng.uniform(0, math.pi)
            dx, dy = math.cos(ang) * r * 1.75, math.sin(ang) * r * 1.75
            if free(x, y, r * 2.2) and free(x + dx, y + dy, r * 2.2):
                col_name = rng.choice(list(colors))
                for cx, cy in ((x, y), (x + dx, y + dy)):
                    area = draw_shape(img, "circle", (cx, cy), r / 0.9, colors[col_name], 0, rng)
                    truths.append(TruthObject("circle", (cx, cy), r, area, col_name))
                placed += [(x, y, r * 2.2), (x + dx, y + dy, r * 2.2)]
                break

    for _ in range(n_objects):
        kind = str(rng.choice(shapes))
        for _try in range(300):
            r = rng.uniform(24, 52)
            x, y = rng.uniform(r, w - r), rng.uniform(r, h - r)
            if free(x, y, r):
                col_name = str(rng.choice(list(colors)))
                area = draw_shape(img, kind, (x, y), r, colors[col_name], rng.uniform(0, 360), rng)
                truths.append(TruthObject(kind, (x, y), r, area, col_name))
                placed.append((x, y, r))
                break
    return _finish(img, rng, noise), truths


def build_feature_dataset(n_per_class: int, seed: int, cfg: Optional[Config] = None):
    """Render single shapes, run them through the real pipeline stages and
    return ``(X, y)`` - features and labels - for classifier training."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    X, y = [], []
    for kind in SHAPE_CLASSES:
        for _ in range(n_per_class):
            img = render_single(kind, rng)
            pre = preprocess(img, cfg)
            regions = segment(pre, cfg).regions
            if not regions:
                continue
            region = max(regions, key=lambda r: r.area)
            X.append(extract_features(region, pre.color).vector())
            y.append(kind)
    # Circles that were separated by the watershed step have a flattened side;
    # include such samples so the classifier learns to recognise them.
    for _ in range(max(1, n_per_class // 3)):
        img, truths = render_scene(int(rng.integers(0, 2**31 - 1)), n_objects=0,
                                   touching_pairs=1)
        pre = preprocess(img, cfg)
        regions = segment(pre, cfg).regions
        if len(regions) == len(truths):
            for region in regions:
                X.append(extract_features(region, pre.color).vector())
                y.append("circle")
    return np.array(X), np.array(y)
