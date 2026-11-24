"""
mesh_is_closed (microvasc/data.py): every edge in exactly two faces, each directed edge once.
"""
import numpy as np
import pytest
from skimage.measure import marching_cubes

import synthetic
from microvasc.data import mesh_is_closed

# each directed edge appears once
TETRAHEDRON = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])


def test_tetrahedron_is_closed():
    assert mesh_is_closed(TETRAHEDRON) is True
    assert mesh_is_closed(TETRAHEDRON.tolist()) is True


def test_tetrahedron_with_a_face_removed_is_open():
    for i in range(4):
        assert mesh_is_closed(np.delete(TETRAHEDRON, i, axis=0)) is False


def test_tetrahedron_with_a_face_flipped_is_not_closed():
    # every edge is still in two faces, but the flipped face repeats 3 directed edges
    for i in range(4):
        faces = TETRAHEDRON.copy()
        faces[i] = faces[i, ::-1]
        assert mesh_is_closed(faces) is False


def test_all_faces_flipped_is_closed():
    assert mesh_is_closed(TETRAHEDRON[:, ::-1]) is True


def test_two_separate_tetrahedra_are_closed():
    assert mesh_is_closed(np.vstack([TETRAHEDRON, TETRAHEDRON + 4])) is True


def test_edge_in_four_faces_is_not_closed():
    # two tetrahedra that share the edge (0, 1)
    other = np.where(TETRAHEDRON >= 2, TETRAHEDRON + 2, TETRAHEDRON)
    assert mesh_is_closed(np.vstack([TETRAHEDRON, other])) is False


def test_duplicate_face_is_not_closed():
    assert mesh_is_closed(np.vstack([TETRAHEDRON, TETRAHEDRON[:1]])) is False


def test_torus_mesh_is_closed():
    # the torus does not touch the volume border
    _, faces, _, _ = marching_cubes(synthetic.torus_mask().astype(np.uint8), level=0.5)
    assert mesh_is_closed(faces) is True


def test_tube_mesh_is_open():
    # the tube ends at the volume border
    _, faces, _, _ = marching_cubes(synthetic.tube_mask().astype(np.uint8), level=0.5)
    assert mesh_is_closed(faces) is False


def test_empty_mesh_is_not_closed():
    assert mesh_is_closed(np.zeros((0, 3), dtype=int)) is False


@pytest.mark.parametrize("faces", [[0, 1, 2], [[0, 1, 2, 3]], [[0.0, 1.0, 2.0]]])
def test_rejects_bad_faces(faces):
    with pytest.raises(ValueError, match="faces"):
        mesh_is_closed(faces)
