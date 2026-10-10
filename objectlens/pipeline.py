"""Orchestrates all stages: pre-process -> segment -> features -> classify -> measure."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .classifier import ShapeClassifier
from .config import Config
from .exceptions import CalibrationError
from .features import ObjectFeatures, extract_all
from .measurement import (Measurement, find_reference, measure,
                          pixels_per_mm_from_reference, reference_side_px)
from .preprocessing import load_image, preprocess
from .segmentation import Region, segment

log = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """One fully analysed object."""

    id: int
    label: str
    confidence: float
    features: ObjectFeatures
    region: Region
    measurement: Optional[Measurement] = None
    is_reference: bool = False


@dataclass
class AnalysisResult:
    """Everything produced for one image."""

    image_name: str
    image: np.ndarray                      # BGR image actually analysed
    objects: List[DetectedObject] = field(default_factory=list)
    pixels_per_mm: Optional[float] = None  # in analysed-image pixels
    scale: float = 1.0
    mask: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return sum(1 for o in self.objects if not o.is_reference)


class ObjectLensPipeline:
    """High-level API used by the CLI, tests and evaluation code."""

    def __init__(self, config: Optional[Config] = None,
                 classifier: Optional[ShapeClassifier] = None):
        self.config = (config or Config()).validate()
        self._classifier = classifier

    @property
    def classifier(self) -> ShapeClassifier:
        if self._classifier is None:
            self._classifier = ShapeClassifier.load(self.config.model_path)
        return self._classifier

    def analyze_image(self, image: np.ndarray, name: str = "image") -> AnalysisResult:
        cfg = self.config
        pre = preprocess(image, cfg)
        seg = segment(pre, cfg)
        feats = extract_all(seg.regions, pre.color)
        labels, confs = self.classifier.predict(feats, cfg.min_confidence)

        ppm = cfg.pixels_per_mm * pre.scale if cfg.pixels_per_mm else None
        ref_idx = None
        if cfg.ref_color and cfg.ref_side_mm:
            ref_idx = find_reference(feats, labels, cfg.ref_color)
            ppm = pixels_per_mm_from_reference(reference_side_px(feats[ref_idx]),
                                               cfg.ref_side_mm)
            log.info("Calibrated from %s reference: %.3f px/mm", cfg.ref_color, ppm)

        objects = []
        for i, (reg, f, lab, conf) in enumerate(zip(seg.regions, feats, labels, confs), 1):
            objects.append(DetectedObject(
                id=i, label=lab, confidence=conf, features=f, region=reg,
                measurement=measure(f, ppm), is_reference=(i - 1 == ref_idx)))
        log.info("%s: %d object(s) found", name, len(objects))
        return AnalysisResult(image_name=name, image=pre.color, objects=objects,
                              pixels_per_mm=ppm, scale=pre.scale, mask=seg.mask)

    def analyze_file(self, path: Union[str, Path]) -> AnalysisResult:
        return self.analyze_image(load_image(path), Path(path).name)
