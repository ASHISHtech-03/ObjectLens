import cv2
import numpy as np
import pytest

from objectlens.cli import main
from objectlens.config import Config
from objectlens.evaluation import evaluate_scenes
from objectlens.exceptions import CalibrationError, ImageLoadError, ModelNotTrainedError
from objectlens.pipeline import ObjectLensPipeline
from objectlens.synthetic import render_scene


def test_pipeline_counts_and_calibrates(classifier):
    img, truths = render_scene(21, n_objects=6, touching_pairs=1, reference=(30.0, 4.0))
    res = ObjectLensPipeline(Config(ref_color="green", ref_side_mm=30.0), classifier).analyze_image(img)
    assert len(res.objects) == len(truths)
    assert res.pixels_per_mm == pytest.approx(4.0, rel=0.03)
    assert sum(o.is_reference for o in res.objects) == 1


@pytest.mark.parametrize("seed", [3, 12, 21])
def test_no_spurious_fragments_between_touching_objects(classifier, seed):
    """Regression: a 3-pixel noise core once produced a sliver 'object'."""
    img, truths = render_scene(seed, n_objects=8, touching_pairs=1, reference=(30.0, 4.0))
    res = ObjectLensPipeline(Config(ref_color="green", ref_side_mm=30.0), classifier).analyze_image(img)
    assert len(res.objects) == len(truths)


def test_pipeline_direct_calibration_with_resize(classifier):
    img, _ = render_scene(5, n_objects=3)
    big = cv2.resize(img, None, fx=3, fy=3)           # 1920x1440 -> will be down-scaled
    res = ObjectLensPipeline(Config(pixels_per_mm=12.0), classifier).analyze_image(big)
    assert res.scale < 1.0
    assert res.pixels_per_mm == pytest.approx(12.0 * res.scale)


def test_missing_reference_raises(classifier):
    img, _ = render_scene(5, n_objects=3)             # no green marker
    with pytest.raises(CalibrationError):
        ObjectLensPipeline(Config(ref_color="green", ref_side_mm=30.0),
                           classifier).analyze_image(img)


def test_blank_image_gives_zero_objects(classifier):
    res = ObjectLensPipeline(Config(), classifier).analyze_image(np.full((100, 100, 3), 200, np.uint8))
    assert res.objects == [] and res.count == 0


def test_pipeline_file_errors(classifier, tmp_path):
    with pytest.raises(ImageLoadError):
        ObjectLensPipeline(Config(), classifier).analyze_file(tmp_path / "x.png")


def test_pipeline_without_model(tmp_path):
    pipe = ObjectLensPipeline(Config(model_path=str(tmp_path / "none.joblib")))
    with pytest.raises(ModelNotTrainedError):
        pipe.analyze_image(np.zeros((50, 50, 3), np.uint8))


def test_end_to_end_metrics(classifier):
    m = evaluate_scenes(ObjectLensPipeline(Config(), classifier), n_scenes=6)
    assert m["detection_recall"] >= 0.9
    assert m["classification_accuracy"] >= 0.9
    assert m["median_area_error_pct"] < 5.0


def test_cli_train_analyze_history(tmp_path, capsys):
    model, db = str(tmp_path / "m.joblib"), str(tmp_path / "t.db")
    common = ["--model", model, "--db", db, "--log-file", str(tmp_path / "l.log")]
    assert main(common + ["train", "--samples", "20"]) == 0
    img, _ = render_scene(8, n_objects=4, reference=(30.0, 4.0))
    path = tmp_path / "scene.png"
    cv2.imwrite(str(path), img)
    assert main(common + ["analyze", str(path), "--out", str(tmp_path / "o"),
                          "--ref-color", "green", "--ref-side-mm", "30", "--save-db"]) == 0
    assert main(common + ["history"]) == 0
    out = capsys.readouterr().out
    assert "scene.png" in out and "run #1" in out


def test_cli_reports_errors_gracefully(tmp_path, capsys):
    common = ["--model", str(tmp_path / "none.joblib"), "--db", str(tmp_path / "t.db"),
              "--log-file", str(tmp_path / "l.log")]
    assert main(common + ["analyze", str(tmp_path / "ghost.png")]) == 2
    assert "Error" in capsys.readouterr().err
