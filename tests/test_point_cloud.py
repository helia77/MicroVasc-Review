"""
LineSegment.point_cloud and NWT.point_cloud (microvasc/data.py): spacing must be a finite number
greater than 0.
"""
import numpy as np
import pytest

import synthetic
from microvasc.data import NWT, LineSegment

SEGMENT = LineSegment(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0]))   # length 5
POINT = LineSegment(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))     # length 0
BAD = [0, 0.0, -1, -10, np.float64(-0.5), float("nan"), float("inf"), float("-inf"), np.array([1.0])]


@pytest.mark.parametrize("spacing,count", [(1, 6), (1.0, 6), (2, 4), (0.7, 9), (5, 2), (100, 2),
                                           (np.float32(2.5), 3)])
def test_valid_spacing(spacing, count):
    points = SEGMENT.point_cloud(spacing)
    assert len(points) == count
    np.testing.assert_array_equal(points[0], SEGMENT.p[0])
    np.testing.assert_array_equal(points[-1], SEGMENT.p[1])
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    np.testing.assert_allclose(gaps, 5 / (count - 1))
    assert gaps.max() <= spacing * (1 + 1e-12)


def test_zero_length_segment_gives_both_end_points():
    np.testing.assert_array_equal(POINT.point_cloud(0.5), [POINT.p[0], POINT.p[1]])


@pytest.mark.parametrize("segment", [SEGMENT, POINT], ids=["length 5", "length 0"])
@pytest.mark.parametrize("spacing", BAD, ids=repr)
def test_other_spacing_raises(segment, spacing):
    with pytest.raises(ValueError, match="spacing must be a finite number greater than 0"):
        segment.point_cloud(spacing)


@pytest.mark.parametrize("spacing", [0, -1, float("nan")], ids=repr)
def test_nwt_point_cloud_raises(tmp_path, spacing):
    synthetic.write_obj(tmp_path / "network.obj")
    with pytest.raises(ValueError, match="spacing must be a finite number greater than 0"):
        NWT(tmp_path / "network.obj").point_cloud(spacing)
