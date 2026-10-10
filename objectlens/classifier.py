"""Stage 4 - shape classification (machine learning + rule-based baseline)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .exceptions import ModelNotTrainedError
from .features import ObjectFeatures

log = logging.getLogger(__name__)

ALGORITHMS = ("svm", "random_forest", "knn")


def _make_estimator(algorithm: str, random_state: int):
    if algorithm == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                   random_state=random_state)
    if algorithm == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    if algorithm == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {ALGORITHMS}")


class ShapeClassifier:
    """Scaled feature vector -> shape label with a confidence score."""

    def __init__(self, algorithm: str = "svm", random_state: int = 42):
        self.algorithm = algorithm
        self.random_state = random_state
        self.pipeline = Pipeline([("scale", StandardScaler()),
                                  ("clf", _make_estimator(algorithm, random_state))])
        self.trained = False

    # ---- training / persistence -------------------------------------------------
    def fit(self, X: np.ndarray, y: Sequence[str]) -> "ShapeClassifier":
        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset")
        self.pipeline.fit(X, y)
        self.trained = True
        log.info("Trained %s on %d samples", self.algorithm, len(X))
        return self

    def save(self, path: str) -> None:
        if not self.trained:
            raise ModelNotTrainedError("Train the classifier before saving it")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"algorithm": self.algorithm, "pipeline": self.pipeline}, path)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ShapeClassifier":
        p = Path(path)
        if not p.is_file():
            raise ModelNotTrainedError(
                f"Model file '{path}' not found. Run: python main.py train")
        payload = joblib.load(p)
        obj = cls(payload["algorithm"])
        obj.pipeline = payload["pipeline"]
        obj.trained = True
        return obj

    # ---- inference ---------------------------------------------------------------
    def predict_vectors(self, X: np.ndarray) -> Tuple[List[str], List[float]]:
        if not self.trained:
            raise ModelNotTrainedError("Classifier has not been trained or loaded")
        X = np.atleast_2d(X)
        probs = self.pipeline.predict_proba(X)
        classes = self.pipeline.classes_
        idx = probs.argmax(axis=1)
        return [str(classes[i]) for i in idx], [float(probs[r, i]) for r, i in enumerate(idx)]

    def predict(self, features: Sequence[ObjectFeatures], min_confidence: float = 0.0):
        """Classify feature objects; low-confidence results become ``unknown``."""
        if not features:
            return [], []
        labels, confs = self.predict_vectors(np.vstack([f.vector() for f in features]))
        labels = [l if c >= min_confidence else "unknown" for l, c in zip(labels, confs)]
        return labels, confs

    # ---- evaluation --------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: Sequence[str]) -> Dict:
        pred = self.pipeline.predict(X)
        labels = sorted(set(y) | set(pred))
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "report": classification_report(y, pred, labels=labels, output_dict=True,
                                            zero_division=0),
        }


class RuleBasedClassifier:
    """Hand-written baseline using only vertex count and simple ratios."""

    def predict_one(self, f: ObjectFeatures) -> str:
        if f.solidity < 0.8 and f.vertices >= 8:
            return "star"
        if f.vertices == 3:
            return "triangle"
        if f.vertices == 4:
            return "square" if f.aspect_ratio < 1.2 else "rectangle"
        if f.vertices == 5:
            return "pentagon"
        if f.vertices == 6:
            return "hexagon"
        if f.circularity > 0.85:
            return "circle" if f.aspect_ratio < 1.2 else "ellipse"
        return "ellipse"

    def predict_vectors(self, X: np.ndarray) -> List[str]:
        out = []
        for row in np.atleast_2d(X):
            circ, aspect, _ext, sol, verts, *_ = row
            f = ObjectFeatures(0, 0, circ, aspect, _ext, sol, int(verts), int(row[5]),
                               (0.0,) * 7, (0.0, 0.0), (0.0, 0.0), 0.0, (0, 0, 0), "")
            out.append(self.predict_one(f))
        return out


def compare_models(X: np.ndarray, y: Sequence[str], folds: int = 5,
                   seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Cross-validate every ML algorithm and the rule-based baseline."""
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: Dict[str, Dict[str, float]] = {}
    for algo in ALGORITHMS:
        pipe = Pipeline([("scale", StandardScaler()), ("clf", _make_estimator(algo, seed))])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        results[algo] = {"mean": float(scores.mean()), "std": float(scores.std())}
    rule = RuleBasedClassifier()
    accs = []
    for _, test_idx in cv.split(X, y):
        accs.append(accuracy_score(y[test_idx], rule.predict_vectors(X[test_idx])))
    results["rule_based"] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    return results
