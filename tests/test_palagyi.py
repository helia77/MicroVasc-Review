import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage.measure import euler_number

import synthetic
from capture_skeleton_golden import DATA, MASKS
from implementations.Skeleton import Skeletonize

CONN26 = ndi.generate_binary_structure(3, 3)


@pytest.fixture(scope="module", params=list(MASKS))
def case(request):
    mask = MASKS[request.param]()
    return request.param, mask, Skeletonize().palagyi(mask.astype(np.uint8))


def test_output_is_uint8_zero_one(case):
    name, mask, skel = case
    assert skel.dtype == np.uint8
    assert skel.shape == mask.shape
    assert set(np.unique(skel)) == {0, 1}


def test_subset_of_input(case):
    name, mask, skel = case
    assert not np.any(skel.astype(bool) & ~mask)


def test_one_voxel_thick(case):
    name, mask, skel = case
    # thinning a skeleton again changes nothing
    np.testing.assert_array_equal(Skeletonize().palagyi(skel), skel)
    # no 2x2x2 block of foreground voxels
    s = skel.astype(int)
    blocks = sum(s[i:s.shape[0] - 1 + i, j:s.shape[1] - 1 + j, k:s.shape[2] - 1 + k]
                 for i in (0, 1) for j in (0, 1) for k in (0, 1))
    assert blocks.max() < 8


def test_same_connected_components(case):
    name, mask, skel = case
    assert ndi.label(skel, CONN26)[1] == ndi.label(mask, CONN26)[1]


def test_same_euler_number(case):
    name, mask, skel = case
    assert euler_number(skel, connectivity=3) == euler_number(mask, connectivity=3)


def test_matches_reference(case):
    name, mask, skel = case
    with np.load(DATA) as data:
        expected = np.unpackbits(data[f"palagyi__{name}"], count=mask.size).reshape(mask.shape)
    np.testing.assert_array_equal(skel, expected)


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int32, float])
def test_nonzero_values_count_as_foreground(dtype):
    mask = synthetic.tube_mask()
    expected = Skeletonize().palagyi(mask.astype(np.uint8))
    np.testing.assert_array_equal(Skeletonize().palagyi((mask * 255).astype(dtype)), expected)
    out = Skeletonize().palagyi(mask.astype(dtype))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)


def test_explicit_templates_path_gives_same_result():
    mask = synthetic.tube_mask().astype(np.uint8)
    default = Skeletonize().palagyi(mask)
    explicit = Skeletonize().palagyi(mask, Skeletonize.DEFAULT_TEMPLATES)
    np.testing.assert_array_equal(default, explicit)
