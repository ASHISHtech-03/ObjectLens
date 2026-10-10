import pytest

from objectlens.exceptions import CalibrationError
from objectlens.features import ObjectFeatures
from objectlens.measurement import (find_reference, measure,
                                    pixels_per_mm_from_reference, reference_side_px)


def _f(area=10000.0, color="green", size=(100.0, 100.0)):
    return ObjectFeatures(area=area, perimeter=400.0, circularity=0.78, aspect_ratio=1.0,
                          extent=1.0, solidity=1.0, vertices=4, hull_vertices=4,
                          hu=(0.0,) * 7, centroid=(0.0, 0.0), rect_size=size, rect_angle=0.0,
                          mean_bgr=(0, 0, 0), color_name=color)


def test_scale_from_reference():
    assert pixels_per_mm_from_reference(120.0, 30.0) == pytest.approx(4.0)


def test_invalid_reference_sizes():
    with pytest.raises(CalibrationError):
        pixels_per_mm_from_reference(0, 10)
    with pytest.raises(CalibrationError):
        pixels_per_mm_from_reference(10, -1)


def test_measure_conversion():
    m = measure(_f(), 4.0)
    assert m.length_mm == pytest.approx(25.0)
    assert m.area_mm2 == pytest.approx(625.0)
    assert m.perimeter_mm == pytest.approx(100.0)
    assert m.equivalent_diameter_mm == pytest.approx(28.2, abs=0.1)


def test_measure_uncalibrated_returns_none():
    assert measure(_f(), None) is None


def test_measure_rejects_bad_scale():
    with pytest.raises(CalibrationError):
        measure(_f(), 0.0)


def test_find_reference_picks_largest_matching():
    feats = [_f(100, "green"), _f(900, "green"), _f(5000, "red")]
    assert find_reference(feats, ["square", "square", "square"], "green") == 1


def test_find_reference_missing():
    with pytest.raises(CalibrationError):
        find_reference([_f(color="red")], ["square"], "green")
    with pytest.raises(CalibrationError):
        find_reference([_f()], ["circle"], "green")


def test_reference_side_px():
    assert reference_side_px(_f(area=10000.0, size=(100.0, 100.0))) == pytest.approx(100.0)
