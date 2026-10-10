import numpy as np
import pytest

from objectlens.classifier import RuleBasedClassifier, ShapeClassifier, compare_models
from objectlens.exceptions import ModelNotTrainedError
from objectlens.synthetic import build_feature_dataset


def test_untrained_classifier_raises():
    with pytest.raises(ModelNotTrainedError):
        ShapeClassifier().predict_vectors(np.zeros((1, 8)))
    with pytest.raises(ModelNotTrainedError):
        ShapeClassifier().save("x.joblib")


def test_load_missing_model(tmp_path):
    with pytest.raises(ModelNotTrainedError):
        ShapeClassifier.load(str(tmp_path / "missing.joblib"))


def test_empty_training_rejected():
    with pytest.raises(ValueError):
        ShapeClassifier().fit(np.empty((0, 8)), [])


def test_unknown_algorithm():
    with pytest.raises(ValueError):
        ShapeClassifier("magic")


def test_held_out_accuracy(classifier, cfg):
    X, y = build_feature_dataset(15, seed=99, cfg=cfg)
    assert classifier.evaluate(X, y)["accuracy"] >= 0.95


def test_save_and_load_roundtrip(classifier, cfg, tmp_path):
    path = str(tmp_path / "m.joblib")
    classifier.save(path)
    loaded = ShapeClassifier.load(path)
    X, _ = build_feature_dataset(3, seed=5, cfg=cfg)
    assert classifier.predict_vectors(X)[0] == loaded.predict_vectors(X)[0]


def test_low_confidence_becomes_unknown(classifier):
    weird = np.array([[0.2, 9.0, 0.1, 0.3, 30, 30, 5, 5]])
    labels, _ = classifier.predict_vectors(weird)

    class F:                     # minimal stand-in for ObjectFeatures
        def vector(self):
            return weird[0]
    out, conf = classifier.predict([F()], min_confidence=1.01)
    assert out == ["unknown"] and conf[0] <= 1.0 and labels


def test_rule_based_baseline(cfg):
    X, y = build_feature_dataset(10, seed=3, cfg=cfg)
    pred = RuleBasedClassifier().predict_vectors(X)
    assert np.mean(np.array(pred) == y) > 0.8


def test_compare_models_keys(cfg):
    X, y = build_feature_dataset(12, seed=4, cfg=cfg)
    res = compare_models(X, y, folds=3)
    assert set(res) == {"svm", "random_forest", "knn", "rule_based"}
