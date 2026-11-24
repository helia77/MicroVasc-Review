# Derived from oof3response.m and eigenvaluefield33.m by Max W.K. Law, under this license:
#
# Copyright (c) 2013, Max W.K. Law
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution
#
# * Neither the name of  nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Optimally oriented flux (OOF), a port of Max W.K. Law's MATLAB oof3response.m.

M.W.K. Law and A.C.S. Chung, "Three Dimensional Curvilinear Structure Detection using
Optimally Oriented Flux", ECCV 2008, pp. 368-382.
M.W.K. Law et al., "Dilated Divergence based Scale-Space Representation for Curve Analysis",
ECCV 2012, pp. 557-571.

Normalization types 1 and 2 need min(radii) >= sigma, as in Law's code. Otherwise type 0 is
used, with a warning.

Differences from the MATLAB code:
- The image is padded before the FFT by default (`pad=True`). The MATLAB code does not pad.
- Eigenvalues use the trigonometric closed form on the shifted matrix instead of the cubic
  coefficients. Both are closed forms of the same roots.
"""
import math
import warnings
from typing import Sequence, Tuple, Union

import numpy as np
import scipy.fft as sfft

RESPONSE_TYPES = (0, 1, 2, 3, 4, 5)
NORMALIZATION_TYPES = (0, 1, 2)

# slices of axis 0 per block in the eigenvalue step, to keep float64 temporaries small
_BLOCK = 16


def _spacing3(spacing: Union[float, Sequence[float]]) -> np.ndarray:
    s = np.asarray(spacing, dtype=float)
    if s.ndim > 1 or s.size not in (1, 3):
        raise ValueError(f"spacing must be one number or 3 values (one per axis), got shape {s.shape}")
    if not np.all(np.isfinite(s) & (s > 0)):
        raise ValueError(f"spacing must be finite and greater than 0, got {s.tolist()}")
    return np.broadcast_to(s, (3,)).copy()


def _frequencies(shape: Tuple[int, int, int], spacing: np.ndarray):
    """Fourier coordinates (cycles per unit length) of the rfftn grid, broadcastable."""
    u0 = sfft.fftfreq(shape[0], d=spacing[0])[:, None, None]
    u1 = sfft.fftfreq(shape[1], d=spacing[1])[None, :, None]
    u2 = sfft.rfftfreq(shape[2], d=spacing[2])[None, None, :]
    return u0, u1, u2


def _flux_filter(rho: np.ndarray, radius: float, sigma: float, normalization: int) -> np.ndarray:
    """
    Fourier filter of one radius, without the u_i * u_j factor.

    Law's code computes
        4/3 pi r^3 / (besselj(1.5, 2 pi r e) / e^1.5) / r^2 * exp(-2 pi^2 sigma^2 rho^2) / rho^1.5
        * (sin(x) / x - cos(x)) * sqrt(1 / (pi^2 r rho)) * (r / sqrt(2 r sigma - sigma^2))^ntype
    with x = 2 pi r rho and e = 1e-12. The first part is r^-0.5 (small argument limit of the
    Bessel function), so this is the same filter written out.
    """
    x = 2 * np.pi * radius * rho
    filt = np.exp(-2 * np.pi**2 * sigma**2 * rho**2) / rho**1.5
    filt *= (np.sin(x) / x - np.cos(x)) * np.sqrt(1 / (np.pi**2 * radius * rho))
    filt *= radius**-0.5
    if normalization > 0:
        # needs radius >= sigma / 2; oof_response falls back to type 0 below sigma
        filt *= (radius / math.sqrt(2 * radius * sigma - sigma**2))**normalization
    return filt


def _flux_tensor(spectrum: np.ndarray, shape: Tuple[int, int, int], freqs, filt: np.ndarray,
                 crop: Tuple[slice, slice, slice]):
    """
    The 6 components (11, 12, 13, 22, 23, 33) of the oriented flux matrix of one radius, as
    float32 arrays cropped with `crop`. `spectrum` is the complex64 rfftn of the image.
    Single precision, as in Law's code.
    """
    weighted = filt.astype(np.float32) * spectrum
    out = []
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        uij = (freqs[i] * freqs[j]).astype(np.float32)
        component = sfft.irfftn(uij * weighted, s=shape, workers=-1)
        out.append(np.ascontiguousarray(component[crop]))
    return out


def _eigenvalues(a11, a12, a13, a22, a23, a33):
    """
    Eigenvalues of symmetric 3x3 matrices, computed in float64.

    Returned in the order of Law's eigenvaluefield33: largest, smallest, middle.
    """
    a11, a12, a13, a22, a23, a33 = (np.asarray(a, dtype=np.float64) for a in (a11, a12, a13, a22, a23, a33))
    q = (a11 + a22 + a33) / 3
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p = np.sqrt((b11**2 + b22**2 + b33**2 + 2 * (a12**2 + a13**2 + a23**2)) / 6)
    det = b11 * b22 * b33 + 2 * a12 * a13 * a23 - b11 * a23**2 - b22 * a13**2 - b33 * a12**2
    safe = np.where(p > 0, p, 1.0)
    r = np.clip(np.where(p > 0, det / (2 * safe**3), 0.0), -1.0, 1.0)
    phi = np.arccos(r) / 3
    largest = q + 2 * p * np.cos(phi)
    smallest = q + 2 * p * np.cos(phi + 2 * np.pi / 3)
    middle = 3 * q - largest - smallest
    return largest, smallest, middle


def _response(e1, e2, e3, response_type: int, use_absolute: bool) -> np.ndarray:
    """Law's sorting of the eigenvalues and the response of one radius."""
    maxe = e1.copy()
    mine = e1.copy()
    mide = e1 + e2 + e3
    for e in (e2, e3):
        if use_absolute:
            bigger, smaller = np.abs(e) > np.abs(maxe), np.abs(e) < np.abs(mine)
        else:
            # as in Law's code: the raw value is compared with the magnitude
            bigger, smaller = e > np.abs(maxe), e < np.abs(mine)
        maxe[bigger] = e[bigger]
        mine[smaller] = e[smaller]
    mide -= maxe + mine
    if response_type == 0:
        return maxe
    if response_type == 1:
        return maxe + mide
    if response_type == 2:
        return np.sqrt(np.maximum(0, maxe * mide))
    if response_type == 3:
        return np.sqrt(np.maximum(0, maxe) * np.maximum(0, mide))
    if response_type == 4:
        return np.maximum(maxe, 0)
    return np.maximum(maxe + mide, 0)


def oof_response(image: np.ndarray, radii: Sequence[float], sigma: float = None,
                 spacing: Union[float, Sequence[float]] = 1.0, response_type: int = 0,
                 use_absolute: bool = True, normalization: int = 1, pad: bool = True) -> np.ndarray:
    """
    Multi-radius OOF response of a 3D image, as Law's oof3response.

    At each radius, the response is a function of the eigenvalues l1, l2 of the oriented flux
    matrix (l1 the largest, see `use_absolute`):
    0: l1, 1: l1 + l2, 2: sqrt(max(0, l1 * l2)), 3: sqrt(max(0, l1) * max(0, l2)),
    4: max(0, l1), 5: max(0, l1 + l2).
    The output holds, per voxel, the signed response with the largest magnitude across radii.
    Bright tubes on a dark background give positive values.

    Args:
        image: 3D image. It is converted to float64, like double(volume) in MATLAB.
        radii: Radii in the units of `spacing`.
        sigma: Gaussian smoothing, in the units of `spacing`. None (default) uses
            min(spacing), as Law's code.
        spacing: Voxel size along array axes 0, 1 and 2, as one number or 3 values.
        response_type: 0 to 5, see above.
        use_absolute: Sort the eigenvalues by magnitude (True) or by value (False).
        normalization: 0 (surface area, ECCV 2008), 1 (curvilinear, ECCV 2012) or 2 (planar,
            ECCV 2012). If min(radii) < sigma, types 1 and 2 are replaced by 0 with a warning.
        pad: Pad each side by ceil((max(radii) + 3 sigma) / spacing) voxels, mirrored, before
            the FFT, and crop after. Without it the FFT wraps around at the border, as in
            Law's code.

    Returns:
        float64 array with the shape of `image`.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image must be 3D, got {image.ndim} dimensions")
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii) & (radii > 0)):
        raise ValueError(f"radii must be a non-empty list of numbers greater than 0, got {radii.tolist()}")
    spacing = _spacing3(spacing)
    sigma = float(spacing.min()) if sigma is None else float(sigma)
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be finite and greater than 0, got {sigma}")
    if response_type not in RESPONSE_TYPES:
        raise ValueError(f"response_type must be one of {RESPONSE_TYPES}, got {response_type!r}")
    if normalization not in NORMALIZATION_TYPES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_TYPES}, got {normalization!r}")
    if normalization > 0 and radii.min() < sigma:
        warnings.warn(f"min(radii) = {radii.min()} is below sigma = {sigma}, so normalization "
                      f"type {normalization} falls back to 0", stacklevel=2)
        normalization = 0

    shape = image.shape
    if pad:
        margin = [math.ceil((radii.max() + 3 * sigma) / s) for s in spacing]
        size = [sfft.next_fast_len(n + 2 * m, real=True) for n, m in zip(shape, margin)]
        image = np.pad(image, [(m, k - n - m) for n, m, k in zip(shape, margin, size)], mode='symmetric')
        crop = tuple(slice(m, m + n) for n, m in zip(shape, margin))
    else:
        crop = (slice(None),) * 3

    padded_shape = image.shape
    # single precision from here on, as the buffers in Law's code
    spectrum = sfft.rfftn(image, workers=-1).astype(np.complex64)
    del image
    freqs = _frequencies(padded_shape, spacing)
    # Law adds 1e-12 so the zero frequency does not divide by 0
    rho = np.sqrt(freqs[0]**2 + freqs[1]**2 + freqs[2]**2) + 1e-12

    output = np.zeros(shape)
    for radius in radii:
        filt = _flux_filter(rho, radius, sigma, normalization)
        tensor = _flux_tensor(spectrum, padded_shape, freqs, filt, crop)
        del filt
        for start in range(0, shape[0], _BLOCK):
            block = slice(start, start + _BLOCK)
            e1, e2, e3 = _eigenvalues(*(t[block] for t in tensor))
            feature = _response(e1, e2, e3, response_type, use_absolute)
            out = output[block]
            better = np.abs(feature) > np.abs(out)
            out[better] = feature[better]
        del tensor
    return output
