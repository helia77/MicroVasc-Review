"""Behavior checks for the Binarizer: Otsu, Hessian, Frangi, Beyond Frangi, state and CLI."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu

import synthetic
from implementations.Binarize import Binarizer

ROOT = Path(__file__).parents[1]
SCALES = [1.0, 2.0]
FRANGI = dict(alpha=0.5, beta=0.5, c=25.0)
TAU = 0.5
VOLUMES = {"tube": synthetic.tube, "bifurcation": synthetic.bifurcation, "noise": synthetic.noise}
MASKS = {"tube": synthetic.tube_mask, "bifurcation": synthetic.bifurcation_mask}


def bimodal():
    """Two intensity levels on a known mask, as float. Without noise, because the threshold is the
    center of the last background bin (as in skimage), so the brightest background voxels can lie above it."""
    mask = synthetic.tube_mask()
    return mask, np.where(mask, 200.0, 50.0)


# Otsu

@pytest.mark.parametrize("name", list(VOLUMES))
def test_threshold_matches_skimage(name):
    # skimage uses one bin per integer value for integer images, so compare on float input,
    # where both use 256 bins over [min, max] and return a bin center
    x = VOLUMES[name]().astype(float)
    assert Binarizer(x)._compute_threshold() == threshold_otsu(x, nbins=256)


def test_otsu3d_compares_with_threshold():
    x = synthetic.tube().astype(float)
    thr = threshold_otsu(x, nbins=256)
    np.testing.assert_array_equal(Binarizer(x, "black").otsu3d(), x >= thr)
    np.testing.assert_array_equal(Binarizer(x, "white").otsu3d(), x <= thr)


def test_otsu3d_splits_bimodal_volume():
    mask, x = bimodal()
    out = Binarizer(x, "black").otsu3d()
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, mask)
    np.testing.assert_array_equal(Binarizer(255 - x, "white").otsu3d(), mask)


@pytest.mark.parametrize("background", ["black", "white"])
def test_otsu2d_thresholds_each_slice_along_axis_0(background):
    x = synthetic.bifurcation().astype(float)
    out = Binarizer(x, background).otsu2d()
    assert out.dtype == np.uint8 and out.shape == x.shape
    for i in range(x.shape[0]):
        thr = threshold_otsu(x[i], nbins=256)
        expected = x[i] >= thr if background == "black" else x[i] <= thr
        np.testing.assert_array_equal(out[i], expected)


# Hessian

def reference_hessian(volume, scale):
    """Scale normalized Hessian (gamma = 2) from scipy, H[..., a, b] = d2/(da db) in array axis order."""
    n = volume.ndim
    H = np.zeros(volume.shape + (n, n))
    for a in range(n):
        for b in range(n):
            order = [0] * n
            order[a] += 1
            order[b] += 1
            H[..., a, b] = ndi.gaussian_filter(volume, (scale,) * n, order, output=np.float64) * scale * scale
    return H


@pytest.mark.parametrize("background", ["black", "white"])
@pytest.mark.parametrize("scale", SCALES)
def test_hessian_matches_reference_where_not_skipped(background, scale):
    x = synthetic.bifurcation()
    x = x if background == "black" else 255 - x
    H, T = Binarizer(x, background)._hessian(scale)
    np.testing.assert_array_equal(H, reference_hessian(x, scale) * T[..., None, None])


@pytest.mark.parametrize("background", ["black", "white"])
@pytest.mark.parametrize("scale", SCALES)
def test_hessian_of_quadratic(background, scale):
    # f = x0^2 + 2 x1^2 + 3 x2^2 + 0.5 x0 x1 has second derivatives [[2, .5, 0], [.5, 4, 0], [0, 0, 6]];
    # with gamma = 2 the Hessian is sigma^2 times that. Checked at the center voxel, where f = 0,
    # because scipy's truncated second derivative kernel does not sum exactly to 0.
    n = 33
    i, j, k = np.meshgrid(*[np.arange(n, dtype=float) - n // 2] * 3, indexing="ij")
    f = i**2 + 2 * j**2 + 3 * k**2 + 0.5 * i * j
    sign = 1 if background == "white" else -1    # dark valley on white, bright ridge on black
    H, T = Binarizer(sign * f, background)._hessian(scale)
    c = n // 2
    assert T[c, c, c] == 1
    expected = sign * scale**2 * np.array([[2, .5, 0], [.5, 4, 0], [0, 0, 6]])
    np.testing.assert_allclose(H[c, c, c], expected, rtol=1e-2, atol=1e-9)


# Frangi and Beyond Frangi

def run_filter(binzr, method, threshold=None):
    if method == "frangi":
        return binzr.frangi(scale_range=SCALES, threshold=threshold, **FRANGI)
    return binzr.bfrangi(tau=TAU, scale_range=SCALES, threshold=threshold)


class NoSkipBinarizer(Binarizer):
    """Binarizer with the Yang and Cheng skip mask turned off."""

    def _hessian(self, scale):
        H = reference_hessian(self.volume, scale)
        return H, np.ones(self.volume.shape, dtype=np.uint8)


@pytest.fixture(scope="module", params=[(m, n) for m in ("frangi", "bfrangi") for n in MASKS],
                ids=lambda p: f"{p[0]}-{p[1]}")
def filtered(request):
    method, name = request.param
    v = getattr(synthetic, name)().astype(float)
    return method, name, v, run_filter(Binarizer(v, "black"), method)


def test_filter_values_in_unit_interval(filtered):
    _, _, _, out = filtered
    assert out.dtype == np.float64
    assert out.min() >= 0 and out.max() <= 1


def test_centerline_brighter_than_background(filtered):
    _, name, _, out = filtered
    mask = MASKS[name]()
    far = ~ndi.binary_dilation(mask, iterations=3)
    assert out[mask].mean() > 10 * out[far].mean()


def test_black_on_v_equals_white_on_minus_v(filtered):
    method, _, v, out = filtered
    np.testing.assert_array_equal(run_filter(Binarizer(-v, "white"), method), out)


@pytest.mark.parametrize("background", ["black", "white"])
@pytest.mark.parametrize("name", list(MASKS))
def test_frangi_unchanged_by_skip_mask(name, background):
    v = getattr(synthetic, name)().astype(float)
    v = v if background == "black" else 255 - v
    np.testing.assert_array_equal(run_filter(NoSkipBinarizer(v, background), "frangi"),
                                  run_filter(Binarizer(v, background), "frangi"))


# State

@pytest.mark.parametrize("method", ["frangi", "bfrangi"])
@pytest.mark.parametrize("background", ["black", "white"])
def test_calls_do_not_change_state(method, background):
    x = synthetic.tube()
    x = x if background == "black" else 255 - x
    original = x.copy()
    binzr = Binarizer(x, background)
    before = binzr.otsu3d()
    first = run_filter(binzr, method, "otsu3d")
    second = run_filter(binzr, method, "otsu3d")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(binzr.otsu3d(), before)
    assert binzr.volume is x and binzr.background == background
    np.testing.assert_array_equal(x, original)


# CLI

def binarize_cli(tmp_path, *args):
    np.save(tmp_path / "in.npy", synthetic.tube())
    result = subprocess.run([sys.executable, "-m", "implementations.Binarize", *args,
                             "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return np.load(tmp_path / "out.npy")


@pytest.mark.parametrize("method", ["otsu2d", "otsu3d"])
def test_cli_otsu(tmp_path, method):
    out = binarize_cli(tmp_path, method, "--background", "white")
    np.testing.assert_array_equal(out, getattr(Binarizer(synthetic.tube(), "white"), method)())


def test_cli_frangi_otsu2d(tmp_path):
    out = binarize_cli(tmp_path, "frangi", "--params", "0.5", "0.5", "25", "--scale", "1", "2", "--th", "otsu2d")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, run_filter(Binarizer(synthetic.tube()), "frangi", "otsu2d"))


def test_cli_bfrangi_otsu3d(tmp_path):
    out = binarize_cli(tmp_path, "bfrangi", "--params", "0.5", "--scale", "1", "2", "--th", "otsu3d")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, run_filter(Binarizer(synthetic.tube()), "bfrangi", "otsu3d"))
