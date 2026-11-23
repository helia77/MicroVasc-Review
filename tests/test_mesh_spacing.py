"""
spacing in npy2obj and npy2off (microvasc/data.py): vertices scaled per array axis.
"""
import hashlib

import numpy as np
import pytest

import synthetic
from microvasc.data import npy2obj, npy2off

WRITERS = {"obj": npy2obj, "off": npy2off}

# SHA-256 of the files written before spacing existed: npy2obj(tube, level=0.5), npy2off(tube)
BEFORE = {
    "obj": "7352eb4201118a5340ba7b5451fb4213f645d82c274205d4d2dd035aff0da2e7",
    "off": "dce984cee2209d87f9914e0a7bd4593082066bdb37b3aa59486063d14f5fffa2",
}


@pytest.fixture(scope="module")
def tube():
    return synthetic.tube_mask().astype(np.uint8)


def write(tmp_path, ext, volume, **kwargs):
    path = tmp_path / f"mesh_{len(list(tmp_path.iterdir()))}.{ext}"
    WRITERS[ext](volume, path, level=0.5, **kwargs)
    return path


def split(path):
    """Vertices as an array, and all other lines (header and faces) as text."""
    lines = path.read_text().splitlines()
    if lines[0] == "OFF":
        n = int(lines[1].split()[0])
        rows, rest = lines[2:2 + n], lines[:2] + lines[2 + n:]
    else:
        rows = [line[2:] for line in lines if line.startswith("v ")]
        rest = [line for line in lines if not line.startswith("v ")]
    return np.array([list(map(float, r.split())) for r in rows]), rest


@pytest.mark.parametrize("ext", WRITERS)
@pytest.mark.parametrize("spacing", [None, 1.0, 1, (1, 1, 1), [1.0, 1.0, 1.0]])
def test_default_spacing_keeps_the_old_file(tmp_path, tube, ext, spacing):
    kwargs = {} if spacing is None else {"spacing": spacing}
    path = write(tmp_path, ext, tube, **kwargs)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BEFORE[ext]


@pytest.mark.parametrize("ext", WRITERS)
@pytest.mark.parametrize("spacing", [(0.75, 1.0, 1.0), (1.0, 0.5, 2.0), (0.3, 0.3, 0.3)])
def test_vertices_scale_per_axis(tmp_path, tube, ext, spacing):
    v1, rest1 = split(write(tmp_path, ext, tube))
    v2, rest2 = split(write(tmp_path, ext, tube, spacing=spacing))
    np.testing.assert_allclose(v2, v1 * spacing, rtol=1e-6)
    assert rest2 == rest1  # same header and faces


@pytest.mark.parametrize("ext", WRITERS)
def test_one_number_is_used_for_all_axes(tmp_path, tube, ext):
    a = write(tmp_path, ext, tube, spacing=0.5)
    b = write(tmp_path, ext, tube, spacing=(0.5, 0.5, 0.5))
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("ext", WRITERS)
def test_npy_path_gets_the_same_spacing(tmp_path, tube, ext):
    np.save(tmp_path / "tube.npy", tube)
    a = write(tmp_path, ext, tube, spacing=(1.0, 0.5, 2.0))
    b = write(tmp_path, ext, tmp_path / "tube.npy", spacing=(1.0, 0.5, 2.0))
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("ext", WRITERS)
@pytest.mark.parametrize("spacing, message", [
    ((1.0, 2.0), "one number or 3 values"),
    ([[1.0, 1.0, 1.0]], "one number or 3 values"),
    (0, "greater than 0"),
    ((1.0, -1.0, 1.0), "greater than 0"),
    ((1.0, np.nan, 1.0), "finite"),
    (np.inf, "finite"),
])
def test_bad_spacing_raises_and_writes_nothing(tmp_path, tube, ext, spacing, message):
    with pytest.raises(ValueError, match=message):
        WRITERS[ext](tube, tmp_path / f"x.{ext}", level=0.5, spacing=spacing)
    assert not (tmp_path / f"x.{ext}").exists()
