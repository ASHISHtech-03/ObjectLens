import cv2
import numpy as np

from objectlens.preprocessing import preprocess
from objectlens.segmentation import segment


def _canvas(bg=235):
    return np.full((240, 320, 3), bg, np.uint8)


def test_blank_image_has_no_regions(cfg):
    assert segment(preprocess(_canvas(), cfg), cfg).regions == []


def test_single_circle_found(cfg):
    img = _canvas()
    cv2.circle(img, (160, 120), 40, (40, 40, 220), -1, cv2.LINE_AA)
    regions = segment(preprocess(img, cfg), cfg).regions
    assert len(regions) == 1
    assert abs(regions[0].area - np.pi * 40 ** 2) / (np.pi * 40 ** 2) < 0.05


def test_dark_background_supported(cfg):
    img = _canvas(bg=30)
    cv2.rectangle(img, (50, 50), (150, 130), (0, 220, 240), -1)
    assert len(segment(preprocess(img, cfg), cfg).regions) == 1


def test_small_specks_are_ignored(cfg):
    img = _canvas()
    cv2.circle(img, (50, 50), 3, (0, 0, 0), -1)
    assert segment(preprocess(img, cfg), cfg).regions == []


def test_touching_circles_are_split(cfg):
    img = _canvas()
    cv2.circle(img, (130, 120), 40, (200, 80, 30), -1, cv2.LINE_AA)
    cv2.circle(img, (200, 120), 40, (200, 80, 30), -1, cv2.LINE_AA)
    assert len(segment(preprocess(img, cfg), cfg).regions) == 2


def test_without_watershed_touching_circles_merge(cfg):
    from dataclasses import replace
    img = _canvas()
    cv2.circle(img, (130, 120), 40, (200, 80, 30), -1, cv2.LINE_AA)
    cv2.circle(img, (200, 120), 40, (200, 80, 30), -1, cv2.LINE_AA)
    c = replace(cfg, watershed=False)
    assert len(segment(preprocess(img, c), c).regions) == 1


def test_separate_objects_not_split(cfg):
    img = _canvas()
    cv2.circle(img, (80, 120), 30, (40, 40, 220), -1, cv2.LINE_AA)
    cv2.rectangle(img, (180, 80), (260, 160), (60, 170, 60), -1)
    assert len(segment(preprocess(img, cfg), cfg).regions) == 2
