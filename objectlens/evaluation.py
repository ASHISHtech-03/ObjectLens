"""Quantitative evaluation of the classifier and the complete pipeline."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List

import numpy as np

from .classifier import ShapeClassifier, compare_models
from .config import Config
from .pipeline import ObjectLensPipeline
from .synthetic import TruthObject, build_feature_dataset, render_scene


def evaluate_classifier(n_train: int = 150, n_test: int = 60, seed: int = 7,
                        cfg: Config = None) -> Dict:
    """Train on ``n_train`` per class, test on independent ``n_test`` per class."""
    cfg = cfg or Config()
    X_tr, y_tr = build_feature_dataset(n_train, seed, cfg)
    X_te, y_te = build_feature_dataset(n_test, seed + 1000, cfg)
    clf = ShapeClassifier("svm").fit(X_tr, y_tr)
    metrics = clf.evaluate(X_te, y_te)
    metrics["model_comparison"] = compare_models(X_tr, y_tr)
    metrics["n_train"], metrics["n_test"] = int(len(X_tr)), int(len(X_te))
    return metrics


def _match(truths: List[TruthObject], objects) -> List:
    """Greedy nearest-centroid matching of ground truth to detections."""
    used, pairs = set(), []
    for t in truths:
        best, best_d = None, 1e9
        for o in objects:
            if o.id in used:
                continue
            d = math.hypot(t.center[0] - o.features.centroid[0],
                           t.center[1] - o.features.centroid[1])
            if d < best_d:
                best, best_d = o, d
        if best is not None and best_d <= 0.6 * t.radius:
            used.add(best.id)
            pairs.append((t, best))
    return pairs


def evaluate_scenes(pipeline: ObjectLensPipeline, n_scenes: int = 40, seed: int = 500,
                    ref_side_mm: float = 30.0, ppm: float = 4.0) -> Dict:
    """End-to-end evaluation on freshly rendered scenes with known ground truth."""
    cfg = replace(pipeline.config, ref_color="green", ref_side_mm=ref_side_mm)
    pipeline = ObjectLensPipeline(cfg, pipeline.classifier)
    tp = fp = fn = cls_ok = col_ok = exact_count = 0
    area_errors: List[float] = []
    for i in range(n_scenes):
        img, truths = render_scene(seed + i, n_objects=7, touching_pairs=1 if i % 3 == 0 else 0,
                                   reference=(ref_side_mm, ppm))
        result = pipeline.analyze_image(img, f"scene_{i}")
        pairs = _match(truths, result.objects)
        tp += len(pairs)
        fn += len(truths) - len(pairs)
        fp += len(result.objects) - len(pairs)
        exact_count += int(len(result.objects) == len(truths))
        for t, o in pairs:
            cls_ok += int(t.label == o.label)
            col_ok += int(t.color == o.features.color_name)
            if o.measurement and not t.is_reference:
                true_mm2 = t.area_px / (ppm ** 2)
                area_errors.append(abs(o.measurement.area_mm2 - true_mm2) / true_mm2)
    return {
        "scenes": n_scenes,
        "detection_precision": tp / max(tp + fp, 1),
        "detection_recall": tp / max(tp + fn, 1),
        "classification_accuracy": cls_ok / max(tp, 1),
        "colour_accuracy": col_ok / max(tp, 1),
        "exact_count_rate": exact_count / n_scenes,
        "mean_area_error_pct": 100.0 * float(np.mean(area_errors)) if area_errors else None,
        "median_area_error_pct": 100.0 * float(np.median(area_errors)) if area_errors else None,
    }


def evaluate_robustness(pipeline: ObjectLensPipeline, noise_levels=(5, 15, 30, 45),
                        n_scenes: int = 20, seed: int = 900) -> List[Dict]:
    """Repeat the end-to-end evaluation while increasing sensor noise (std. dev.)."""
    from dataclasses import replace
    cfg = replace(pipeline.config, ref_color="green", ref_side_mm=30.0)
    pipe = ObjectLensPipeline(cfg, pipeline.classifier)
    rows = []
    for sigma in noise_levels:
        tp = fp = fn = ok = 0
        for i in range(n_scenes):
            img, truths = render_scene(seed + i, n_objects=7, reference=(30.0, 4.0),
                                       noise=float(sigma))
            try:
                result = pipe.analyze_image(img, f"noise_{sigma}_{i}")
            except Exception:       # e.g. reference marker lost under heavy noise
                fn += len(truths)
                continue
            pairs = _match(truths, result.objects)
            tp += len(pairs)
            fn += len(truths) - len(pairs)
            fp += len(result.objects) - len(pairs)
            ok += sum(int(t.label == o.label) for t, o in pairs)
        rows.append({"noise_sigma": sigma, "recall": tp / max(tp + fn, 1),
                     "precision": tp / max(tp + fp, 1),
                     "classification_accuracy": ok / max(tp, 1)})
    return rows
