"""
oof_response against reference output from Law's MATLAB code.

tests/data/oof_matlab_kesm.npz and oof_matlab_micro_ct.npz hold 40^3 uint8 crops of our KESM
and micro-CT volumes (CC BY 4.0), with the matching crop of the reference output, the central
region only. MATLAB ran oof3response(double(volume), radii) on the full 200^3 volumes, so with
the defaults: response type 0, absolute values, normalization type 1, sigma 1, spacing 1. Its
output is single precision, stored here as float32 without loss.

Crop origins in the full volumes: KESM (125, 29, 29), micro-CT (71, 103, 43). Both are far from
the volume border, so the wrap-around of the MATLAB FFT does not reach them.

The port runs on the crop without padding. Its FFT wraps at the crop border, so the central
region (margin ceil(max(radii) + 3 sigma)) differs from the full-volume result by a few 1e-4.
Run on the full volumes, the port agrees with the MATLAB output in these regions to 2e-6.
"""
from pathlib import Path

import numpy as np
import pytest

from microvasc import _oof

DATA = Path(__file__).parent / "data"

CASES = {
    "kesm": ("oof_matlab_kesm.npz", np.arange(1.0, 6.01, 0.5), 9),
    "micro_ct": ("oof_matlab_micro_ct.npz", [1.01, 2.01, 3.01], 7),
}


@pytest.mark.parametrize("name", CASES)
def test_matches_law_matlab_output(name):
    filename, radii, margin = CASES[name]
    with np.load(DATA / filename) as f:
        volume, reference = f["input"], f["reference"]
    response = _oof.oof_response(volume, radii, pad=False)
    inner = response[margin:-margin, margin:-margin, margin:-margin]
    assert inner.shape == reference.shape
    # measured: 3.9e-4 (KESM, max |value| 6.9) and 3.3e-4 (micro-CT, max |value| 10.7)
    np.testing.assert_allclose(inner, reference, rtol=0, atol=5e-4)
