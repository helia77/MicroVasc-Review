"""
Regression tests: the Binarizer must reproduce the reference outputs stored in
tests/data/binarize_golden.npz (see capture_golden.py).
"""
import numpy as np
import pytest

import synthetic
from capture_golden import BACKGROUNDS, DATA, METHODS, case_key, run_case

CASES = [(name, bg, method, th)
         for name in synthetic.VOLUMES
         for bg in BACKGROUNDS
         for method, th in METHODS]


@pytest.fixture(scope="module")
def golden():
    with np.load(DATA) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize("name", list(synthetic.VOLUMES))
def test_synthetic_volumes_are_reproducible(golden, name):
    # If this fails, the random number stream changed. The stored inputs stay the reference.
    np.testing.assert_array_equal(synthetic.VOLUMES[name](), golden[f"input__{name}"])


@pytest.mark.parametrize("name,background,method,threshold", CASES,
                         ids=[case_key(*c) for c in CASES])
def test_matches_golden(golden, name, background, method, threshold):
    expected = golden[case_key(name, background, method, threshold)]
    result = run_case(golden[f"input__{name}"], background, method, threshold)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    if expected.dtype == np.uint8:
        # binary masks must match exactly
        np.testing.assert_array_equal(result, expected)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-10)
