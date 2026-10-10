import csv
import json

import pytest

from objectlens.config import Config
from objectlens.exceptions import StorageError
from objectlens.pipeline import ObjectLensPipeline
from objectlens.reporting import annotate, format_summary, save_outputs, summarize
from objectlens.storage import Repository
from objectlens.synthetic import render_scene


@pytest.fixture(scope="module")
def result(classifier):
    cfg = Config(ref_color="green", ref_side_mm=30.0)
    img, _ = render_scene(11, n_objects=5, reference=(30.0, 4.0))
    return ObjectLensPipeline(cfg, classifier).analyze_image(img, "scene.png")


def test_repository_roundtrip(tmp_path, result):
    repo = Repository(str(tmp_path / "t.db"))
    run_id = repo.save_analysis(result, "svm")
    assert repo.list_runs()[0]["id"] == run_id
    rows = repo.get_detections(run_id)
    assert len(rows) == len(result.objects)
    assert sum(r["is_reference"] for r in rows) == 1


def test_repository_delete_cascades(tmp_path, result):
    repo = Repository(str(tmp_path / "t.db"))
    run_id = repo.save_analysis(result, "svm")
    assert repo.delete_run(run_id) is True
    assert repo.get_detections(run_id) == []
    assert repo.delete_run(run_id) is False


def test_repository_unwritable_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(StorageError):
        Repository(str(blocker / "sub" / "t.db"))


def test_sql_injection_is_harmless(tmp_path, result):
    repo = Repository(str(tmp_path / "t.db"))
    repo.save_analysis(result, "svm")
    with pytest.raises((ValueError, StorageError)):
        repo.get_detections("1; DROP TABLE runs")     # rejected, never executed as SQL
    assert len(repo.list_runs()) == 1


def test_annotate_keeps_shape(result):
    assert annotate(result).shape == result.image.shape


def test_summary_excludes_reference(result):
    s = summarize(result)
    assert s["total_objects"] == len(result.objects) - 1
    assert s["pixels_per_mm"] == pytest.approx(4.0, rel=0.03)
    assert "Objects found" in format_summary(result)


def test_save_outputs(tmp_path, result):
    paths = save_outputs(result, str(tmp_path), "scene")
    for p in paths.values():
        assert p.exists()
    with open(paths["csv"], newline="") as fh:
        assert len(list(csv.DictReader(fh))) == len(result.objects)
    assert json.loads(paths["json"].read_text())["summary"]["image"] == "scene.png"
