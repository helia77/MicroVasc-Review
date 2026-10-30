"""
Tests for the NetMets precision and recall (metrics/skeleton_metric.py) and the NWT/OBJ reader.

Reference values in tests/data/netmets_golden.npz (see capture_netmets_golden.py): the reader
outputs equal those of the reader before the NWT point fix, for everything that worked then
(the OBJ path and the NWT vertices); the NetMets values were captured right after NetMets was added.
"""
import math

import numpy as np
import pytest

import synthetic
from capture_netmets_golden import (DATA, SKELETONS, VOXEL_SIZES, case_key, netmets_case,
                                    reader_outputs)
from manage_data.manage_data import NWT
from metrics.skeleton_metric import (as_segments, netmets, netmets_similarity,
                                     voxel_skeleton_to_segments)

# a planar polyline in z = 0
POLYLINE = np.array([[[0, 0, 0], [3, 1, 0]], [[3, 1, 0], [7, -2, 0]], [[7, -2, 0], [12, 0, 0]]], float)


@pytest.fixture(scope="module")
def golden():
    with np.load(DATA) as data:
        return {k: data[k] for k in data.files}


def total_length(segments):
    return np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum()


# NetMets on line segments

@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_identical_networks(sigma):
    assert netmets(POLYLINE, POLYLINE, sigma) == pytest.approx((1.0, 1.0), rel=1e-12)


@pytest.mark.parametrize("sigma", [1.0, 2.0])
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_parallel_copy(sigma, delta):
    # every point of either copy is exactly delta from the other one
    expected = math.exp(-delta**2 / (2 * sigma**2))
    shifted = POLYLINE + [0, 0, delta]
    assert netmets(shifted, POLYLINE, sigma) == pytest.approx((expected, expected), rel=1e-12)


def test_far_apart_networks():
    P, R = netmets(POLYLINE + [0, 0, 20], POLYLINE, sigma=1.0)
    assert P < 1e-12 and R < 1e-12


@pytest.mark.parametrize("sigma,branch", [(1.0, 0.5), (1.0, 4.0), (2.0, 3.0)])
def test_missing_branch(sigma, branch):
    # the ground truth has an extra branch perpendicular to the test line, starting on it;
    # a point at height h on the branch is h away from the test line
    test = np.array([[[0, 0, 0], [10, 0, 0]]], float)
    truth = np.concatenate([test, [[[5, 0, 0], [5, branch, 0]]]])
    matched = sigma * math.sqrt(math.pi / 2) * math.erf(branch / (sigma * math.sqrt(2)))
    P, R = netmets(test, truth, sigma)
    assert P == pytest.approx(1.0, rel=1e-12)
    # midpoint rule with steps of sigma / 10
    assert R == pytest.approx((10 + matched) / (10 + branch), abs=1e-4)


def test_sigma_scales_with_coordinates():
    shifted = POLYLINE + [0, 0, 1.0]
    np.testing.assert_allclose(netmets(shifted * 3, POLYLINE * 3, sigma=3.0), netmets(shifted, POLYLINE, sigma=1.0))


def test_each_input_has_its_own_scale():
    # ground truth stored in units of 0.5 um, test in um
    shifted = POLYLINE + [0, 0, 1.0]
    assert netmets(POLYLINE, POLYLINE / 2, truth_scale=2.0) == pytest.approx((1.0, 1.0), rel=1e-12)
    assert netmets(shifted, POLYLINE / 2, truth_scale=2.0) == netmets(shifted, POLYLINE)
    assert netmets(POLYLINE / 2, shifted, test_scale=(2.0, 2.0, 2.0)) == netmets(POLYLINE, shifted)


def test_long_segments_split_into_short_ones_give_the_same_result():
    # B is the same curve, cut into 100 pieces
    t = np.linspace(0, 1, 101)[:, None]
    p0, p1 = POLYLINE[1]
    pieces = np.stack([p0 + (p1 - p0) * t[:-1], p0 + (p1 - p0) * t[1:]], axis=1)
    A = POLYLINE + [0, 0.7, 0.3]
    assert netmets_similarity(A, pieces) == pytest.approx(netmets_similarity(A, POLYLINE[1:2]), rel=1e-12)


def test_empty_networks():
    empty = np.zeros((0, 2, 3))
    assert netmets(empty, empty) == (1.0, 1.0)
    assert netmets(empty, POLYLINE) == (0.0, 0.0)
    assert netmets(POLYLINE, empty) == (0.0, 0.0)


def test_bad_input():
    with pytest.raises(ValueError):
        netmets(POLYLINE, POLYLINE, sigma=0)
    with pytest.raises(ValueError):
        as_segments(np.zeros((4, 3)))


# voxel skeletons

def test_voxel_line_lengths_follow_voxel_size():
    mask = np.zeros((6, 6, 6), np.uint8)
    mask[1:6, 2, 3] = 1
    assert total_length(voxel_skeleton_to_segments(mask, (2.0, 1.0, 1.0))) == pytest.approx(8.0)
    mask = np.zeros((6, 6, 6), np.uint8)
    mask[2, 3, 1:5] = 1
    assert total_length(voxel_skeleton_to_segments(mask, (1.0, 1.0, 0.75))) == pytest.approx(2.25)
    mask = np.zeros((3, 3, 3), np.uint8)
    mask[0, 0, 0] = mask[1, 1, 1] = 1
    assert total_length(voxel_skeleton_to_segments(mask, (1.0, 1.0, 0.75))) == pytest.approx(math.sqrt(2.5625))


def test_voxel_coordinates_are_scaled_array_indices():
    mask = np.zeros((4, 4, 4), bool)
    mask[1, 2, 3] = mask[2, 2, 3] = True
    segments = voxel_skeleton_to_segments(mask, (0.5, 2.0, 3.0))
    np.testing.assert_array_equal(segments, [[[0.5, 4.0, 9.0], [1.0, 4.0, 9.0]]])


def test_isolated_voxel_is_a_zero_length_segment():
    mask = np.zeros((3, 3, 3), np.uint8)
    mask[1, 1, 1] = 7
    segments = voxel_skeleton_to_segments(mask)
    np.testing.assert_array_equal(segments, [[[1, 1, 1], [1, 1, 1]]])


def test_voxel_line_matches_its_segment():
    mask = np.zeros((5, 12, 6), np.uint8)
    mask[3, 0:10, 4] = 1
    truth = np.array([[[3, 0, 4], [3, 9, 4]]], float) * (0.75, 1.0, 1.0)
    P, R = netmets(voxel_skeleton_to_segments(mask, (0.75, 1.0, 1.0)), truth)
    assert P == pytest.approx(1.0, rel=1e-12) and R == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("spacing", list(VOXEL_SIZES))
@pytest.mark.parametrize("name", list(SKELETONS))
def test_netmets_matches_golden(golden, name, spacing):
    shape = tuple(golden[f"shape__{name}"])
    skeleton = np.unpackbits(golden[f"skeleton__{name}"], count=math.prod(shape)).reshape(shape)
    result = netmets_case(skeleton, SKELETONS[name][1], VOXEL_SIZES[spacing])
    np.testing.assert_allclose(result, golden[case_key(name, spacing)], rtol=1e-10)


# NWT and OBJ reader

def test_reader_matches_golden(golden):
    for key, value in reader_outputs().items():
        np.testing.assert_array_equal(value, golden[key], err_msg=key)


def test_nwt_edge_points_are_x_y_z_radius(tmp_path):
    synthetic.write_nwt(tmp_path / "network.nwt")
    network = NWT(str(tmp_path / "network.nwt"))
    assert network.header == "nwtfileformat "
    for e, (v0, v1, interior) in zip(network.e, synthetic.NETWORK_EDGES):
        assert e.v == (v0, v1)
        np.testing.assert_array_equal(np.array(e.p, float).reshape(-1, 4), np.reshape(interior, (-1, 4)))


def test_nwt_and_obj_give_the_same_segments(tmp_path):
    synthetic.write_nwt(tmp_path / "network.nwt")
    synthetic.write_obj(tmp_path / "network.obj")
    nwt = as_segments(NWT(str(tmp_path / "network.nwt")))
    obj = as_segments(NWT(str(tmp_path / "network.obj")))
    assert nwt.shape == (6, 2, 3)
    np.testing.assert_array_equal(nwt, obj)
    assert netmets(NWT(str(tmp_path / "network.nwt")), obj) == pytest.approx((1.0, 1.0), rel=1e-12)


def test_save_obj_round_trip(tmp_path):
    # save_obj keeps only the end vertices of each edge
    synthetic.write_obj(tmp_path / "network.obj")
    NWT(str(tmp_path / "network.obj")).save_obj(tmp_path / "saved.obj")
    saved = NWT(str(tmp_path / "saved.obj"))
    expected = [(synthetic.NETWORK_VERTICES[v0], synthetic.NETWORK_VERTICES[v1]) for v0, v1, _ in synthetic.NETWORK_EDGES]
    np.testing.assert_array_equal(as_segments(saved), expected)
