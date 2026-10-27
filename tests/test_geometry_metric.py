import warnings

import numpy as np
import pytest

from metrics.geometry_metric import metric

# 3 x 4 masks with TP = 2, FP = 1, FN = 3, TN = 6
TRUE = np.array([[1, 1, 1, 0],
                 [1, 1, 0, 0],
                 [0, 0, 0, 0]], dtype=bool)
PRED = np.array([[1, 0, 0, 1],
                 [0, 1, 0, 0],
                 [0, 0, 0, 0]], dtype=bool)


@pytest.fixture
def m():
    return metric(TRUE, PRED)


def test_counts(m):
    assert (m.TP, m.FP, m.FN, m.TN) == (2, 1, 3, 6)


def test_hand_computed_values(m):
    assert m.jaccard() == pytest.approx(2 / 6)
    assert m.dice() == pytest.approx(4 / 8)
    assert m.f_score() == pytest.approx(4 / 8)
    assert m.precision() == pytest.approx(2 / 3)
    assert m.recall() == pytest.approx(2 / 5)
    assert m.TPR() == m.recall()
    assert m.specificity() == pytest.approx(6 / 7)
    assert m.FPR() == pytest.approx(1 / 7)
    assert m.fall_out() == pytest.approx(1 / 7)
    assert m.accuracy() == pytest.approx(8 / 12)


def test_nonzero_values_count_as_foreground():
    m = metric(TRUE.astype(np.uint8) * 255, PRED.astype(np.uint8))
    assert (m.TP, m.FP, m.FN, m.TN) == (2, 1, 3, 6)


@pytest.mark.parametrize("seed", range(5))
def test_paper_identities(seed):
    # F = 2pr / (p + r) = D = 2J / (1 + J)
    rng = np.random.default_rng(seed)
    true = rng.random((10, 12, 14)) < 0.3
    pred = rng.random((10, 12, 14)) < 0.4
    m = metric(true, pred)
    p, r, J, D = m.precision(), m.recall(), m.jaccard(), m.dice()
    assert m.f_score() == pytest.approx(2 * p * r / (p + r))
    assert D == pytest.approx(2 * p * r / (p + r))
    assert D == pytest.approx(2 * J / (1 + J))


ZEROS = np.zeros((4, 4, 4), dtype=bool)
ONES = np.ones((4, 4, 4), dtype=bool)

# (true, predicted, expected values when a denominator is zero)
EMPTY_CASES = {
    "both empty": (ZEROS, ZEROS, dict(jaccard=1.0, dice=1.0, f_score=1.0, precision=1.0,
                                      recall=1.0, TPR=1.0, specificity=1.0, FPR=0.0,
                                      fall_out=0.0, accuracy=1.0)),
    "empty truth": (ZEROS, ONES, dict(jaccard=0.0, dice=0.0, f_score=0.0, precision=0.0,
                                      recall=0.0, TPR=0.0, specificity=0.0, FPR=1.0,
                                      fall_out=1.0, accuracy=0.0)),
    "empty prediction": (ONES, ZEROS, dict(jaccard=0.0, dice=0.0, f_score=0.0, precision=0.0,
                                           recall=0.0, TPR=0.0, specificity=0.0, FPR=0.0,
                                           fall_out=0.0, accuracy=0.0)),
    "all foreground": (ONES, ONES, dict(jaccard=1.0, dice=1.0, f_score=1.0, precision=1.0,
                                        recall=1.0, TPR=1.0, specificity=0.0, FPR=0.0,
                                        fall_out=0.0, accuracy=1.0)),
}


@pytest.mark.parametrize("case", list(EMPTY_CASES))
def test_empty_cases(case):
    true, pred, expected = EMPTY_CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = metric(true, pred)
        values = {name: getattr(m, name)() for name in expected}
    assert values == expected


def test_zero_size_arrays():
    empty = np.zeros((0,), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert metric(empty, empty).accuracy() == 0.0
