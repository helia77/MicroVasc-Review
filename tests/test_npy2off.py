import numpy as np
import pytest
from skimage.measure import marching_cubes

import synthetic
from manage_data import manage_data as md


def read_off(path):
    lines = path.read_text().splitlines()
    n_verts, n_faces, n_edges = map(int, lines[1].split())
    verts = np.array([list(map(float, line.split())) for line in lines[2:2 + n_verts]])
    faces = np.array([list(map(int, line.split())) for line in lines[2 + n_verts:]])
    return lines[0], (n_verts, n_faces, n_edges), verts, faces


@pytest.fixture(scope="module")
def tube():
    mask = synthetic.tube_mask().astype(np.uint8)
    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    return mask, verts, faces


@pytest.mark.parametrize("from_file", [False, True], ids=["array", "npy file"])
def test_tube_mesh(tmp_path, tube, from_file):
    mask, verts, faces = tube
    if from_file:
        np.save(tmp_path / "tube.npy", mask)
        source = tmp_path / "tube.npy"
    else:
        source = mask
    counts = md.npy2off(source, tmp_path / "tube.off")

    header, (n_verts, n_faces, n_edges), v, f = read_off(tmp_path / "tube.off")
    assert header == "OFF"
    assert counts == (n_verts, n_faces) == (len(verts), len(faces))
    assert n_edges == 0
    assert v.shape == (n_verts, 3)
    np.testing.assert_allclose(v, verts, rtol=1e-6)
    # every face is a triangle with 0-based indices (OBJ would be 1-based)
    assert f.shape == (n_faces, 4)
    assert np.all(f[:, 0] == 3)
    assert f[:, 1:].min() == 0
    assert f[:, 1:].max() == n_verts - 1
    np.testing.assert_array_equal(f[:, 1:], faces)


def test_rejects_other_input(tmp_path):
    with pytest.raises(ValueError):
        md.npy2off([[0, 1]], tmp_path / "x.off")
