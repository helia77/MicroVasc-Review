"""
npy2obj (microvasc/data.py): surface mesh of a volume as OBJ, from an array or a .npy path.
"""
import numpy as np
import pytest
from skimage.measure import marching_cubes

import synthetic
from microvasc.data import npy2obj


def read_obj(path):
    lines = path.read_text().splitlines()
    verts = np.array([list(map(float, line.split()[1:])) for line in lines if line.startswith("v ")])
    faces = np.array([list(map(int, line.split()[1:])) for line in lines if line.startswith("f ")])
    assert len(verts) + len(faces) == len(lines)
    return verts, faces


@pytest.fixture(scope="module")
def tube():
    return synthetic.tube_mask().astype(np.uint8)


@pytest.mark.parametrize("source", ["array", "str", "Path"])
def test_array_str_and_path_give_the_same_mesh(tmp_path, tube, source):
    np.save(tmp_path / "tube.npy", tube)
    volume = {"array": tube, "str": str(tmp_path / "tube.npy"), "Path": tmp_path / "tube.npy"}[source]
    npy2obj(volume, tmp_path / "tube.obj", level=0.5)

    verts, faces, _, _ = marching_cubes(tube, level=0.5)
    v, f = read_obj(tmp_path / "tube.obj")
    np.testing.assert_allclose(v, verts, rtol=1e-6)
    # OBJ face indices are 1-based
    np.testing.assert_array_equal(f, faces + 1)


def test_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        npy2obj(tmp_path / "missing.npy", tmp_path / "x.obj")
    assert not (tmp_path / "x.obj").exists()


def test_rejects_other_input(tmp_path):
    with pytest.raises(ValueError, match="numpy array or file path"):
        npy2obj([[0, 1]], tmp_path / "x.obj")
