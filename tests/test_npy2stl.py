"""
npy2stl (microvasc/data.py): surface mesh of a volume as binary STL, faces pointing outward.
"""
import numpy as np
import pytest
from skimage.measure import marching_cubes

import synthetic
from microvasc.data import npy2stl

TRIANGLE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def read_stl(path):
    raw = path.read_bytes()
    n = int(np.frombuffer(raw, "<u4", count=1, offset=80)[0])
    assert len(raw) == 84 + 50 * n
    return raw[:80], np.frombuffer(raw, TRIANGLE, offset=84)


def signed_volume(corners):
    """Enclosed volume of a closed mesh (divergence theorem). Positive when faces point outward."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6


MASKS = {
    "tube": synthetic.tube_mask,
    "bifurcation": synthetic.bifurcation_mask,
    "torus": synthetic.torus_mask,
}


@pytest.mark.parametrize("name", MASKS)
def test_file_layout_and_vertices(tmp_path, name):
    mask = MASKS[name]().astype(np.uint8)
    n = npy2stl(mask, tmp_path / "m.stl")
    header, tri = read_stl(tmp_path / "m.stl")

    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    assert n == len(tri) == len(faces)
    assert len(header) == 80
    assert not header.lower().startswith(b"solid")
    # same triangles as marching_cubes, with two vertices swapped
    np.testing.assert_array_equal(tri["vertices"], verts[faces[:, [0, 2, 1]]])
    assert np.all(tri["attribute"] == 0)


@pytest.mark.parametrize("name", MASKS)
def test_normals_are_unit_and_match_the_vertex_order(tmp_path, name):
    npy2stl(MASKS[name]().astype(np.uint8), tmp_path / "m.stl")
    _, tri = read_stl(tmp_path / "m.stl")
    np.testing.assert_allclose(np.linalg.norm(tri["normal"], axis=1), 1.0, atol=1e-6)
    v = tri["vertices"].astype(float)
    cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    np.testing.assert_allclose(tri["normal"], cross / np.linalg.norm(cross, axis=1, keepdims=True),
                               atol=1e-5)


def test_closed_mesh_has_positive_volume(tmp_path):
    # the torus does not touch the volume border, so its mesh is closed
    npy2stl(synthetic.torus_mask().astype(np.uint8), tmp_path / "torus.stl")
    _, tri = read_stl(tmp_path / "torus.stl")
    volume = signed_volume(tri["vertices"].astype(float))
    exact = 2 * np.pi**2 * 9.0 * 3.0**2  # torus with major radius 9 and minor radius 3
    assert volume > 0
    assert abs(volume / exact - 1) < 0.05


def test_spacing_scales_vertices_and_volume(tmp_path):
    mask = synthetic.torus_mask().astype(np.uint8)
    spacing = (0.75, 1.0, 2.0)
    npy2stl(mask, tmp_path / "a.stl")
    npy2stl(mask, tmp_path / "b.stl", spacing=spacing)
    _, a = read_stl(tmp_path / "a.stl")
    _, b = read_stl(tmp_path / "b.stl")
    np.testing.assert_allclose(b["vertices"], a["vertices"] * spacing, rtol=1e-6)
    np.testing.assert_allclose(signed_volume(b["vertices"].astype(float)),
                               signed_volume(a["vertices"].astype(float)) * np.prod(spacing), rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(b["normal"], axis=1), 1.0, atol=1e-6)


def test_array_and_npy_path_give_the_same_file(tmp_path):
    mask = synthetic.tube_mask().astype(np.uint8)
    np.save(tmp_path / "tube.npy", mask)
    npy2stl(mask, tmp_path / "a.stl", spacing=0.5)
    npy2stl(str(tmp_path / "tube.npy"), tmp_path / "b.stl", spacing=(0.5, 0.5, 0.5))
    assert (tmp_path / "a.stl").read_bytes() == (tmp_path / "b.stl").read_bytes()


@pytest.mark.parametrize("spacing", [(1.0, 2.0), 0, (1.0, np.nan, 1.0)])
def test_bad_spacing_raises_and_writes_nothing(tmp_path, spacing):
    with pytest.raises(ValueError, match="spacing"):
        npy2stl(synthetic.tube_mask().astype(np.uint8), tmp_path / "x.stl", spacing=spacing)
    assert not (tmp_path / "x.stl").exists()


def test_rejects_other_input(tmp_path):
    with pytest.raises(ValueError, match="numpy array or file path"):
        npy2stl([[0, 1]], tmp_path / "x.stl")
