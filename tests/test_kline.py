import numpy as np
import pytest
import scipy.ndimage as ndi

import synthetic
from capture_kline_golden import CASES, DATA, WEIGHTS, case_key
from implementations.Skeleton import Skeletonize

CONN26 = ndi.generate_binary_structure(3, 3)


@pytest.fixture(scope="module", params=[(name, weights) for weights in WEIGHTS for name in CASES],
                ids=lambda p: f"{p[1]}-{p[0]}")
def case(request):
    pytest.importorskip("skfmm")
    name, weights = request.param
    make, root = CASES[name]
    dmw, mbtr = WEIGHTS[weights]
    mask = make()
    return name, mask, root, Skeletonize().kline(mask.astype(np.uint8), root, dmw=dmw, mbtr=mbtr)


def test_output_is_uint8_zero_one(case):
    name, mask, root, skel = case
    assert skel.dtype == np.uint8
    assert skel.shape == mask.shape
    assert set(np.unique(skel)) == {0, 1}


def test_subset_of_input(case):
    name, mask, root, skel = case
    assert not np.any(skel.astype(bool) & ~mask)


def test_contains_root(case):
    name, mask, root, skel = case
    assert skel[root] == 1


def test_connected(case):
    name, mask, root, skel = case
    assert ndi.label(skel, CONN26)[1] == 1


# Exact match only for the default weights. With the KESM weights (dmw 19.25) the travel times
# far from the root are equal to the last bit, so the path depends on floating point rounding and
# differs between machines and library builds. Those cases get the property checks above.
@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("weights", ["defaults"])
def test_matches_reference(name, weights):
    pytest.importorskip("skfmm")
    make, root = CASES[name]
    dmw, mbtr = WEIGHTS[weights]
    mask = make().astype(np.uint8)
    with np.load(DATA) as data:
        expected = np.unpackbits(data[case_key(name, weights)], count=mask.size).reshape(mask.shape)
    np.testing.assert_array_equal(Skeletonize().kline(mask, root, dmw=dmw, mbtr=mbtr), expected)


def test_nonzero_values_count_as_foreground():
    pytest.importorskip("skfmm")
    make, root = CASES["tube"]
    mask = make()
    np.testing.assert_array_equal(Skeletonize().kline(mask.astype(np.uint8) * 255, root),
                                  Skeletonize().kline(mask, root))


# root validation runs before scikit-fmm is imported, so these tests need no skip
@pytest.mark.parametrize("root", [(-1, 5, 2), (24, 5, 2), (3, 5, 99)])
def test_root_outside_volume(root):
    with pytest.raises(ValueError, match="outside the volume"):
        Skeletonize().kline(synthetic.tube_mask().astype(np.uint8), root)


def test_root_not_on_vessel():
    with pytest.raises(ValueError, match="not on a vessel voxel"):
        Skeletonize().kline(synthetic.tube_mask().astype(np.uint8), (0, 23, 0))


def test_root_needs_three_indices():
    with pytest.raises(ValueError, match="3 indices"):
        Skeletonize().kline(synthetic.tube_mask().astype(np.uint8), (3, 5))
