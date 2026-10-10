import cv2
import numpy as np
import pytest

from objectlens.features import color_name_from_bgr, extract_features
from objectlens.preprocessing import preprocess
from objectlens.segmentation import segment


def _features(draw, cfg):
    img = np.full((240, 320, 3), 235, np.uint8)
    draw(img)
    pre = preprocess(img, cfg)
    region = segment(pre, cfg).regions[0]
    return extract_features(region, pre.color)


def test_circle_features(cfg):
    f = _features(lambda im: cv2.circle(im, (160, 120), 50, (40, 40, 220), -1, cv2.LINE_AA), cfg)
    # pixel-staircase contours over-estimate perimeter, so ideal circles score ~0.90
    assert f.circularity > 0.88
    assert f.aspect_ratio < 1.1
    assert f.color_name == "red"
    assert f.centroid == pytest.approx((160, 120), abs=1.5)


def test_rectangle_features(cfg):
    f = _features(lambda im: cv2.rectangle(im, (60, 80), (260, 160), (60, 170, 60), -1), cfg)
    assert f.vertices == 4
    assert f.aspect_ratio == pytest.approx(2.5, rel=0.08)
    assert f.solidity > 0.97
    assert f.color_name == "green"


def test_feature_vector_shape(cfg):
    f = _features(lambda im: cv2.circle(im, (160, 120), 40, (0, 140, 255), -1), cfg)
    assert f.vector().shape == (8,)
    assert np.isfinite(f.vector()).all()


@pytest.mark.parametrize("bgr,name", [
    ((40, 40, 220), "red"), ((200, 80, 30), "blue"), ((60, 170, 60), "green"),
    ((0, 220, 240), "yellow"), ((0, 140, 255), "orange"), ((20, 20, 20), "black"),
    ((250, 250, 250), "white"), ((128, 128, 128), "gray"),
])
def test_color_names(bgr, name):
    assert color_name_from_bgr(bgr) == name
