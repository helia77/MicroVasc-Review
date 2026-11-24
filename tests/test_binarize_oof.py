"""Binarizer.oof: background, thresholds, input checks and response on the synthetic tube."""
import numpy as np
import pytest
import scipy.ndimage as ndi

import synthetic
from microvasc import _oof
from microvasc.binarize import Binarizer

RADII = [1.0, 2.0, 3.0, 4.0, 5.0]


def centerline():
    """Voxels on the middle part of the tube axis."""
    p, q = (np.array(a, float) for a in synthetic.TUBE_AXIS[0])
    points = np.rint([p + t * (q - p) for t in np.linspace(0.3, 0.7, 9)]).astype(int)
    return tuple(points.T)


@pytest.fixture(scope="module")
def tube():
    return synthetic.tube()


@pytest.fixture(scope="module")
def black(tube):
    return Binarizer(tube, "black").oof(RADII)


def test_defaults_are_laws_defaults(tube, black):
    np.testing.assert_array_equal(black, _oof.oof_response(tube.astype(float), RADII, sigma=1.0))
    assert black.dtype == np.float64 and black.shape == tube.shape


def test_white_background_negates_the_volume(tube, black):
    white = Binarizer(255 - tube, "white").oof(RADII)
    np.testing.assert_allclose(white, black, atol=1e-6 * np.abs(black).max())
    assert black[centerline()].min() > 0 and white[centerline()].min() > 0


def test_centerline_above_background(black):
    far = ~ndi.binary_dilation(synthetic.tube_mask(), iterations=3)
    # vessels are positive; dark surroundings give negative values
    assert black[centerline()].min() > 5 * black[far].max()


def test_best_radius_close_to_the_tube_radius(tube):
    # the tube has radius 3
    radii = np.arange(1.0, 6.01, 0.5)
    binzr = Binarizer(tube)
    values = [binzr.oof([r])[centerline()].mean() for r in radii]
    assert abs(radii[int(np.argmax(values))] - 3.0) <= 0.5


@pytest.mark.parametrize("threshold", ["otsu2d", "otsu3d"])
def test_otsu_threshold(tube, black, threshold):
    mask = Binarizer(tube).oof(RADII, threshold=threshold)
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, getattr(Binarizer(black), threshold)())


@pytest.mark.parametrize("threshold", [2.0, 2, np.float32(2.0), -1.5])
def test_number_threshold(tube, black, threshold):
    mask = Binarizer(tube).oof(RADII, threshold=threshold)
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, (black > threshold).astype(np.uint8))


def test_number_threshold_finds_the_tube(tube):
    mask = Binarizer(tube).oof(RADII, threshold=2.0).astype(bool)
    truth = synthetic.tube_mask()
    assert (mask & truth).sum() > 0.95 * truth.sum()
    assert (mask & ~ndi.binary_dilation(truth, iterations=2)).sum() < 0.01 * truth.sum()


@pytest.mark.parametrize("threshold", ["bogus", "otsu", "OTSU3D", "", True, float("nan"), float("inf"), [1.0]])
def test_bad_threshold_raises(tube, threshold):
    with pytest.raises(ValueError, match="threshold must be None, 'otsu2d', 'otsu3d' or a finite number"):
        Binarizer(tube).oof(RADII, threshold=threshold)


@pytest.mark.parametrize("kwargs, match", [
    ({"radii": []}, "radii"),
    ({"radii": [-1.0, 2.0]}, "radii"),
    ({"sigma": 0.0}, "sigma"),
    ({"spacing": (1.0, 1.0)}, "spacing"),
    ({"spacing": 0.0}, "spacing"),
])
def test_bad_input_raises(tube, kwargs, match):
    options = {"radii": RADII, **kwargs}
    with pytest.raises(ValueError, match=match):
        Binarizer(tube).oof(**options)


def test_2d_volume_raises():
    with pytest.raises(ValueError, match="3D"):
        Binarizer(np.zeros((8, 8))).oof(RADII)


def test_radius_below_sigma_warns(tube):
    with pytest.warns(UserWarning, match="falls back to 0"):
        Binarizer(tube[:12, :12, :12]).oof([0.5, 1.0, 2.0])


@pytest.mark.parametrize("background", ["black", "white"])
def test_call_does_not_change_the_volume(tube, background):
    x = tube.astype(float) if background == "black" else 255.0 - tube
    original = x.copy()
    binzr = Binarizer(x, background)
    first = binzr.oof(RADII, threshold="otsu3d")
    np.testing.assert_array_equal(binzr.oof(RADII, threshold="otsu3d"), first)
    assert binzr.volume is x
    np.testing.assert_array_equal(x, original)
