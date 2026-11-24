"""
close in npy2obj, npy2off and npy2stl (microvasc/data.py): pad one voxel below `level` so the
mesh is closed at the volume border.
"""
import hashlib

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import synthetic
from microvasc.data import mesh_is_closed, npy2obj, npy2off, npy2stl

WRITERS = {"obj": npy2obj, "off": npy2off, "stl": npy2stl}

# SHA-256 of the tube files written before `close` existed, all at level 0.5
BEFORE = {
    "obj": "7352eb4201118a5340ba7b5451fb4213f645d82c274205d4d2dd035aff0da2e7",
    "off": "dce984cee2209d87f9914e0a7bd4593082066bdb37b3aa59486063d14f5fffa2",
    "stl": "b61b148bb2a9d4711c29464cefc516e84c2222d5a89e44693315af8bc0e9d3b0",
}

MASKS = {"tube": synthetic.tube_mask, "bifurcation": synthetic.bifurcation_mask}
SPACINGS = [1.0, (0.75, 1.0, 2.0)]
STL_TRIANGLE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def read_mesh(path):
    """Vertices and 0-based faces of an OBJ, OFF or binary STL file."""
    if path.suffix == ".stl":
        corners = np.frombuffer(path.read_bytes(), STL_TRIANGLE, offset=84)["vertices"].reshape(-1, 3)
        verts, index = np.unique(corners, axis=0, return_inverse=True)
        return verts.astype(float), index.reshape(-1, 3)
    lines = path.read_text().splitlines()
    if path.suffix == ".off":
        nv, nf = map(int, lines[1].split()[:2])
        verts = np.array([line.split() for line in lines[2:2 + nv]], dtype=float)
        faces = np.array([line.split()[1:] for line in lines[2 + nv:2 + nv + nf]], dtype=int)
        return verts, faces
    verts = np.array([line.split()[1:] for line in lines if line.startswith("v ")], dtype=float)
    faces = np.array([line.split()[1:] for line in lines if line.startswith("f ")], dtype=int) - 1
    return verts, faces


def write(tmp_path, ext, volume, name="mesh", **kwargs):
    path = tmp_path / f"{name}.{ext}"
    kwargs.setdefault("level", 0.5)
    WRITERS[ext](volume, path, **kwargs)
    return path


def inside_pad_layer(verts, shape, spacing):
    """With a pad strictly below the level, no vertex lies on a pad voxel."""
    s = np.broadcast_to(np.asarray(spacing, float), (3,))
    margin = (1 - 1e-6) * s
    return bool(np.all(verts > -margin) and np.all(verts < (np.array(shape) - 1) * s + margin))


def blurred(mask):
    """Non-binary float volume in [0, 1]."""
    return ndi.gaussian_filter(mask.astype(float), 1.0)


@pytest.mark.parametrize("ext", WRITERS)
def test_close_false_writes_the_same_bytes(tmp_path, ext):
    tube = synthetic.tube_mask().astype(np.uint8)
    path = write(tmp_path, ext, tube, close=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BEFORE[ext]
    a = write(tmp_path, ext, tube, "a", spacing=(0.75, 1.0, 2.0))
    b = write(tmp_path, ext, tube, "b", spacing=(0.75, 1.0, 2.0), close=False)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("ext", WRITERS)
@pytest.mark.parametrize("name", MASKS)
@pytest.mark.parametrize("spacing", SPACINGS)
def test_close_gives_a_closed_mesh(tmp_path, ext, name, spacing):
    mask = MASKS[name]().astype(np.uint8)
    _, open_faces = read_mesh(write(tmp_path, ext, mask, "open", spacing=spacing))
    _, faces = read_mesh(write(tmp_path, ext, mask, "closed", spacing=spacing, close=True))
    assert not mesh_is_closed(open_faces)
    assert mesh_is_closed(faces)


@pytest.mark.parametrize("ext", WRITERS)
@pytest.mark.parametrize("name", MASKS)
def test_close_on_a_float_volume(tmp_path, ext, name):
    _, faces = read_mesh(write(tmp_path, ext, blurred(MASKS[name]()), close=True))
    assert mesh_is_closed(faces)


@pytest.mark.parametrize("name", MASKS)
def test_close_on_a_bool_volume_at_level_0(tmp_path, name):
    # npy2obj defaults to level 0.0, so a pad of 0 (False) would not be below the level
    mask = MASKS[name]()
    path = tmp_path / "m.obj"
    npy2obj(mask, path, close=True)
    verts, faces = read_mesh(path)
    assert mesh_is_closed(faces)
    assert inside_pad_layer(verts, mask.shape, 1.0)


def test_close_on_a_volume_with_negative_values(tmp_path):
    # signed distance to the tube wall: negative outside, so the pad must be below the minimum
    mask = synthetic.tube_mask()
    volume = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    path = tmp_path / "m.obj"
    npy2obj(volume, path, level=0.0, spacing=(0.75, 1.0, 2.0), close=True)
    verts, faces = read_mesh(path)
    assert mesh_is_closed(faces)
    assert inside_pad_layer(verts, mask.shape, (0.75, 1.0, 2.0))


@pytest.mark.parametrize("ext", WRITERS)
@pytest.mark.parametrize("volume", ["mask", "float"])
@pytest.mark.parametrize("spacing", SPACINGS)
def test_vertices_away_from_the_border_are_unchanged(tmp_path, ext, volume, spacing):
    mask = synthetic.bifurcation_mask()
    data = mask.astype(np.uint8) if volume == "mask" else blurred(mask)
    old, _ = read_mesh(write(tmp_path, ext, data, "open", spacing=spacing))
    new, _ = read_mesh(write(tmp_path, ext, data, "closed", spacing=spacing, close=True))
    s = np.broadcast_to(np.asarray(spacing, float), (3,))
    top = (np.array(mask.shape) - 1) * s

    def inner(v):
        return v[np.all((v >= s) & (v <= top - s), axis=1)]

    # same points both ways, up to float32 rounding
    assert len(inner(old)) == len(inner(new)) > 0
    assert cKDTree(new).query(inner(old))[0].max() < 1e-5
    assert cKDTree(old).query(inner(new))[0].max() < 1e-5
