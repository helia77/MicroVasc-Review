"""
Optimally oriented flux (microvasc/_oof.py), a port of Law's oof3response.m.

With normalization type 0, the OOF matrix of radius r at x is
    -1 / (4 pi^2 r^2) * integral over the sphere |y - x| = r of grad(v)(y) n^T dA
with v the image smoothed by a Gaussian of width sigma and n the outward normal. By the
divergence theorem this is -1 / (4 pi^2 r^2) times the integral of the Hessian of v over the
ball. Bright tubes give positive values.
"""
import numpy as np
import pytest
import scipy.fft as sfft
from scipy.integrate import quad

import synthetic
from microvasc import _oof

SIGMA, SMOOTH, RADIUS = 1.0, 2.0, 3.0   # Gaussian smoothing, width of the test shapes, radius
S2 = SMOOTH**2 + SIGMA**2               # width^2 after smoothing


def grid(shape, spacing=(1.0, 1.0, 1.0)):
    """Physical coordinates, centered at index n // 2 on each axis."""
    return np.meshgrid(*[(np.arange(n) - n // 2) * s for n, s in zip(shape, spacing)], indexing="ij")


def tensor(image, radius, sigma=SIGMA, normalization=0):
    """OOF matrix components of one radius, without padding."""
    shape = image.shape
    spectrum = sfft.rfftn(np.asarray(image, float)).astype(np.complex64)
    freqs = _oof._frequencies(shape, np.ones(3))
    rho = np.sqrt(freqs[0]**2 + freqs[1]**2 + freqs[2]**2) + 1e-12
    filt = _oof._flux_filter(rho, radius, sigma, normalization)
    return _oof._flux_tensor(spectrum, shape, freqs, filt, (slice(None),) * 3)


def blob_value(radius=RADIUS, sigma=SIGMA):
    """All 3 eigenvalues at the center of exp(-|x|^2 / (2 SMOOTH^2)), normalization type 0."""
    s2 = SMOOTH**2 + sigma**2
    return radius * (SMOOTH**2 / s2)**1.5 * np.exp(-radius**2 / (2 * s2)) / (3 * np.pi * s2)


def line_value():
    """The 2 nonzero eigenvalues on the axis of a Gaussian line, normalization type 0."""
    integral = quad(lambda t: (RADIUS**2 - t**2) * np.exp(-(RADIUS**2 - t**2) / (2 * S2)), -RADIUS, RADIUS)[0]
    return SMOOTH**2 / S2 * integral / (4 * np.pi * RADIUS**2 * S2)


@pytest.fixture(scope="module")
def blob():
    z, y, x = grid((64, 64, 64))
    return np.exp(-(x**2 + y**2 + z**2) / (2 * SMOOTH**2))


@pytest.fixture(scope="module")
def line():
    """Gaussian line along axis 2. The FFT is periodic, so the line has no ends."""
    z, y, _ = grid((64, 64, 16))
    return np.exp(-(z**2 + y**2) / (2 * SMOOTH**2))


# eigenvalues

def test_eigenvalues_match_eigvalsh():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20000, 3, 3)) * 10.0**rng.uniform(-3, 3, size=(20000, 1, 1))
    a = (a + a.transpose(0, 2, 1)) / 2
    # repeated eigenvalues, a zero matrix and a rank-1 matrix
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    extra = [q @ np.diag(d) @ q.T for d in ([2, 2, 2], [2, 2, -1], [-1, 3, 3], [0, 0, 0], [5, 0, 0])]
    a = np.concatenate([a, np.array(extra)])
    expected = np.linalg.eigvalsh(a)
    largest, smallest, middle = _oof._eigenvalues(a[:, 0, 0], a[:, 0, 1], a[:, 0, 2],
                                                  a[:, 1, 1], a[:, 1, 2], a[:, 2, 2])
    scale = np.maximum(np.abs(expected).max(axis=1), 1e-300)
    error = np.abs(np.stack([smallest, middle, largest], axis=1) - expected).max(axis=1) / scale
    assert error.max() < 1e-5


# closed forms

@pytest.mark.parametrize("normalization", [0, 1, 2])
def test_gaussian_blob(blob, normalization):
    expected = blob_value() * (RADIUS / np.sqrt(2 * RADIUS * SIGMA - SIGMA**2))**normalization
    t = tensor(blob, RADIUS, normalization=normalization)
    center = (32, 32, 32)
    np.testing.assert_allclose([t[0][center], t[3][center], t[5][center]], expected, rtol=1e-5)
    np.testing.assert_allclose([t[1][center], t[2][center], t[4][center]], 0, atol=1e-6 * expected)
    out = _oof.oof_response(blob, [RADIUS], sigma=SIGMA, normalization=normalization, pad=False)
    np.testing.assert_allclose(out[center], expected, rtol=1e-5)


def test_gaussian_line(line):
    t = tensor(line, RADIUS)
    p = (32, 32, 5)
    expected = line_value()
    np.testing.assert_allclose([t[0][p], t[3][p]], expected, rtol=1e-5)
    np.testing.assert_allclose([t[1][p], t[2][p], t[4][p], t[5][p]], 0, atol=1e-6 * expected)


# response types on the bright line (eigenvalues l, l, 0) and the dark line (-l, -l, 0)
BRIGHT = {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 2}
DARK = {0: -1, 1: -2, 2: 1, 3: 0, 4: 0, 5: 0}


@pytest.mark.parametrize("response_type", range(6))
def test_response_types(line, response_type):
    expected = line_value()
    for sign, table in ((1, BRIGHT), (-1, DARK)):
        out = _oof.oof_response(sign * line, [RADIUS], sigma=SIGMA, normalization=0,
                                response_type=response_type, pad=False)
        np.testing.assert_allclose(out[32, 32, 5], table[response_type] * expected, rtol=1e-5,
                                   atol=1e-6 * expected)


def test_use_absolute_false_compares_values(line):
    # dark line: eigenvalues 0, -l, -l. By value, the largest is 0.
    out = _oof.oof_response(-line, [RADIUS], sigma=SIGMA, normalization=0, use_absolute=False, pad=False)
    assert abs(out[32, 32, 5]) < 1e-6 * line_value()


@pytest.mark.parametrize("use_absolute, l1, l2", [(True, -3.0, 2.0), (False, 2.0, -3.0)])
def test_eigenvalue_sorting_as_in_law(use_absolute, l1, l2):
    # eigenvalues 2, -3, 1 in Law's order (largest, smallest, middle). With use_absolute=False
    # Law compares the raw value with the magnitude, so 1 ends up as the smallest, not -3.
    e = [np.array([v]) for v in (2.0, -3.0, 1.0)]
    assert _oof._response(*e, 0, use_absolute)[0] == l1
    assert _oof._response(*e, 1, use_absolute)[0] == l1 + l2


def test_spacing():
    # the same blob in physical units, sampled twice as fine along axis 2
    spacing = (1.0, 1.0, 0.5)
    z, y, x = grid((64, 64, 128), spacing)
    blob = np.exp(-(x**2 + y**2 + z**2) / (2 * SMOOTH**2))
    out = _oof.oof_response(blob, [RADIUS], sigma=SIGMA, spacing=spacing, normalization=0, pad=False)
    np.testing.assert_allclose(out[32, 32, 64], blob_value(), rtol=1e-5)


def test_largest_magnitude_across_radii(blob):
    radii = [1.0, 2.0, 3.0, 4.0]
    out = _oof.oof_response(blob, radii, sigma=SIGMA, normalization=0, pad=False)
    expected = max((blob_value(r) for r in radii), key=abs)
    np.testing.assert_allclose(out[32, 32, 32], expected, rtol=1e-5)


# flux in the spatial domain

def sphere(radius, n_theta=40, n_phi=80):
    """Quadrature points (outward normals) and weights on a sphere."""
    c, wc = np.polynomial.legendre.leggauss(n_theta)
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    c, phi = np.meshgrid(c, phi, indexing="ij")
    s = np.sqrt(1 - c**2)
    normals = np.stack([c, s * np.cos(phi), s * np.sin(phi)], axis=-1).reshape(-1, 3)
    weights = (np.broadcast_to(wc[:, None], c.shape) * 2 * np.pi / n_phi).ravel() * radius**2
    return normals, weights


def spatial_flux(image, center, radius, sigma):
    """-1 / (4 pi^2 r^2) * flux of grad(v) n^T through the sphere, from the Fourier series of v."""
    shape = image.shape
    u = [sfft.fftfreq(n) for n in shape]
    smooth = sfft.fftn(image) * np.exp(-2 * np.pi**2 * sigma**2 * (
        u[0][:, None, None]**2 + u[1][None, :, None]**2 + u[2][None, None, :]**2)) / image.size
    normals, weights = sphere(radius)
    points = np.asarray(center, float) + radius * normals
    e = [np.exp(2j * np.pi * np.outer(u[k], points[:, k])) for k in range(3)]
    grad = []
    for k in range(3):
        d = smooth * (2j * np.pi * u[k]).reshape([-1 if a == k else 1 for a in range(3)])
        grad.append(np.einsum("abc,ap,bp,cp->p", d, *e, optimize=True).real)
    flux = np.einsum("ip,pj,p->ij", np.array(grad), normals, weights)
    return -flux / (4 * np.pi**2 * radius**2)


def test_flux_matches_the_spatial_domain():
    # odd sizes, so the Fourier series has no Nyquist term
    image = synthetic.render(synthetic.capsule((25, 25, 25), *synthetic.TUBE_AXIS[0], 3.0), seed=1)
    radius = 2.5
    t = tensor(image, radius)
    full = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
    for voxel in [(12, 12, 12), (12, 15, 12), (5, 18, 4), (20, 3, 9)]:
        expected = spatial_flux(image.astype(float), voxel, radius, SIGMA)
        got = np.array([[t[full[i][j]][voxel] for j in range(3)] for i in range(3)])
        np.testing.assert_allclose(got, expected, atol=1e-4 * np.abs(expected).max())


# options

def test_fallback_to_normalization_0_warns(blob):
    radii = [0.5, 2.0, 3.0]
    with pytest.warns(UserWarning, match="falls back to 0"):
        out = _oof.oof_response(blob, radii, sigma=SIGMA, normalization=1, pad=False)
    np.testing.assert_array_equal(out, _oof.oof_response(blob, radii, sigma=SIGMA, normalization=0, pad=False))


def test_no_warning_when_radii_reach_sigma(blob, recwarn):
    _oof.oof_response(blob[:16, :16, :16], [1.0, 2.0], sigma=1.0)
    assert not recwarn.list


def test_sigma_defaults_to_the_smallest_spacing(blob):
    a = _oof.oof_response(blob, [3.0], spacing=(1.0, 1.0, 2.0), pad=False)
    b = _oof.oof_response(blob, [3.0], sigma=1.0, spacing=(1.0, 1.0, 2.0), pad=False)
    np.testing.assert_array_equal(a, b)


def test_padding_stops_wrap_around():
    # bright blob next to the border of axis 0
    z, y, x = np.meshgrid(*[np.arange(40)] * 3, indexing="ij")
    image = 100 * np.exp(-((z - 3)**2 + (y - 20)**2 + (x - 20)**2) / (2 * SMOOTH**2))
    far = (36, 20, 20)  # 7 voxels away through the border, 33 inside
    wrapped = _oof.oof_response(image, [3.0], pad=False)
    padded = _oof.oof_response(image, [3.0])
    assert padded.shape == image.shape and padded.dtype == np.float64
    assert abs(wrapped[far]) > 100 * abs(padded[far])
    # away from the border, padding changes only rounding
    np.testing.assert_allclose(padded[12:28], wrapped[12:28], atol=1e-4 * np.abs(wrapped).max())


def test_integer_input(blob):
    image = np.rint(blob * 200).astype(np.uint8)
    np.testing.assert_array_equal(_oof.oof_response(image, [2.0, 3.0]),
                                  _oof.oof_response(image.astype(float), [2.0, 3.0]))


@pytest.mark.parametrize("kwargs, match", [
    ({"radii": []}, "radii"),
    ({"radii": [0.0, 1.0]}, "radii"),
    ({"radii": [1.0, np.nan]}, "radii"),
    ({"sigma": 0.0}, "sigma"),
    ({"spacing": (1.0, 2.0)}, "spacing"),
    ({"response_type": 6}, "response_type"),
    ({"normalization": 3}, "normalization"),
])
def test_rejects_bad_options(kwargs, match):
    options = {"radii": [1.0, 2.0], **kwargs}
    with pytest.raises(ValueError, match=match):
        _oof.oof_response(np.zeros((8, 8, 8)), **options)


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="3D"):
        _oof.oof_response(np.zeros((8, 8)), [1.0])
