"""
adjust_brightness_contrast (microvasc/data.py). Needs OpenCV (`io` extra), skipped without it.
"""
import numpy as np
import pytest

from microvasc.data import adjust_brightness_contrast

cv2 = pytest.importorskip("cv2")


def ramp():
    """Gray levels 20 to 119, one pixel each, as a 10x10 image."""
    return np.arange(20, 120, dtype=np.uint8).reshape(10, 10)


def bgr(gray):
    return np.stack([gray, gray, gray], axis=-1)


def test_normal_image():
    # 1% clip: half a pixel at each end, so the cuts are at 20 and 118
    image = bgr(ramp())
    adjusted, alpha, beta = adjust_brightness_contrast(image)
    assert alpha == 255 / 98 and beta == -20 * 255 / 98
    np.testing.assert_array_equal(adjusted, cv2.convertScaleAbs(image, alpha=alpha, beta=beta))
    assert adjusted[0, 0, 0] == 0 and adjusted[-1, -1, 0] == 255


@pytest.mark.parametrize("value", [0, 100, 255])
@pytest.mark.parametrize("to_image", [lambda g: g, bgr], ids=["2D", "BGR"])
def test_constant_image_is_returned_unchanged(value, to_image):
    image = to_image(np.full((6, 7), value, np.uint8))
    adjusted, alpha, beta = adjust_brightness_contrast(image)
    assert (alpha, beta) == (1.0, 0.0)
    np.testing.assert_array_equal(adjusted, image)
    assert adjusted is not image


def test_two_neighboring_levels_are_returned_unchanged():
    # both cuts fall on level 50; before, 255 / 0 raised ZeroDivisionError
    image = bgr(np.array([[50, 51] * 4] * 3, np.uint8))
    adjusted, alpha, beta = adjust_brightness_contrast(image)
    assert (alpha, beta) == (1.0, 0.0)
    np.testing.assert_array_equal(adjusted, image)


@pytest.mark.parametrize("clip", [0.5, 1, 5])
def test_2d_image_gives_the_same_result_as_bgr(clip):
    gray = np.random.default_rng(0).integers(30, 200, (12, 9)).astype(np.uint8)
    adjusted, alpha, beta = adjust_brightness_contrast(gray, clip)
    expected, expected_alpha, expected_beta = adjust_brightness_contrast(bgr(gray), clip)
    assert adjusted.shape == gray.shape
    assert (alpha, beta) == (expected_alpha, expected_beta)
    np.testing.assert_array_equal(adjusted, expected[..., 0])


def test_input_is_not_changed():
    image = ramp()
    adjust_brightness_contrast(image)
    np.testing.assert_array_equal(image, ramp())
