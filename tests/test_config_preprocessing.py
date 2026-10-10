import cv2
import numpy as np
import pytest

from objectlens.config import Config
from objectlens.exceptions import ConfigError, ImageLoadError
from objectlens.preprocessing import (ensure_bgr, load_image, preprocess,
                                      resize_max_dimension)


def test_default_config_is_valid():
    assert Config().validate() is not None


@pytest.mark.parametrize("kwargs", [
    {"blur_kernel": 4}, {"min_area_px": 0}, {"watershed_ratio": 1.5},
    {"pixels_per_mm": -1.0}, {"ref_color": "green"}, {"ref_color": "pink", "ref_side_mm": -2},
    {"max_dimension": 10}, {"min_confidence": 2.0},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ConfigError):
        Config(**kwargs).validate()


def test_load_missing_file(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "nope.png")


def test_load_unsupported_extension(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    with pytest.raises(ImageLoadError):
        load_image(f)


def test_load_corrupt_image(tmp_path):
    f = tmp_path / "bad.png"
    f.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        load_image(f)


def test_load_valid_image(tmp_path):
    f = tmp_path / "ok.png"
    cv2.imwrite(str(f), np.full((20, 30, 3), 128, np.uint8))
    assert load_image(f).shape == (20, 30, 3)


def test_ensure_bgr_variants():
    assert ensure_bgr(np.zeros((5, 5), np.uint8)).shape == (5, 5, 3)
    assert ensure_bgr(np.zeros((5, 5, 4), np.uint8)).shape == (5, 5, 3)
    assert ensure_bgr(np.zeros((5, 5, 3), np.float32)).dtype == np.uint8
    with pytest.raises(ImageLoadError):
        ensure_bgr(np.zeros((0, 0), np.uint8))
    with pytest.raises(ImageLoadError):
        ensure_bgr("not an array")


def test_resize_only_when_needed():
    img = np.zeros((200, 400, 3), np.uint8)
    same, s1 = resize_max_dimension(img, 500)
    small, s2 = resize_max_dimension(img, 100)
    assert s1 == 1.0 and same.shape == img.shape
    assert small.shape[1] == 100 and s2 == pytest.approx(0.25)


def test_preprocess_outputs(cfg):
    pre = preprocess(np.full((64, 64, 3), 200, np.uint8), cfg)
    assert pre.gray.shape == pre.enhanced.shape == (64, 64)
    assert pre.blurred.shape == (64, 64, 3)
