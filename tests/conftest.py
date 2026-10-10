import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from objectlens.classifier import ShapeClassifier          # noqa: E402
from objectlens.config import Config                       # noqa: E402
from objectlens.synthetic import build_feature_dataset     # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def classifier(cfg):
    """Small, fast classifier shared by the whole test session."""
    X, y = build_feature_dataset(40, seed=1, cfg=cfg)
    return ShapeClassifier("svm").fit(X, y)
