"""
NetMets geometry metric for comparing skeletons (Mayerich et al., BMC Bioinformatics 2012,
13(Suppl 8):S7).

For two sets of curves A and B,

    M(A, B) = (1 / L) sum_{a in A} integral exp(-d(a(t), B)^2 / (2 sigma^2)) dt

where L is the total length of A, d(x, B) is the distance from x to the closest point on B
and sigma is the sensitivity. For a test case T and ground truth G, precision is M(T, G) and
recall is M(G, T). NetMets reports the same quantities as error rates: GFPR = 1 - precision and
GFNR = 1 - recall.

Curves are straight line segments. The integral is evaluated with the midpoint rule, with steps
of at most sigma / 10 along each segment, and d is the exact distance to the nearest segment
of B. The NetMets software measures d between sample points instead, so values may differ
slightly from it.

Coordinates and sigma must use the same unit (um in the paper, which uses sigma = 1 um).
"""
import itertools
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from microvasc.data import NWT

# the 13 neighbor offsets of a 26-neighborhood that point "forward", so each pair is found once
_FORWARD_26 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                        if (i, j, k) > (0, 0, 0)])
_CHUNK = 65536


def as_segments(network: Union["NWT", ArrayLike], scale: ArrayLike = 1.0) -> np.ndarray:
    """
    Line segments of a network as a float array of shape (N, 2, 3).

    Args:
        network: A `microvasc.data.NWT` object (loaded from .nwt or .obj), or an array-like of
            segments with shape (N, 2, 3).
        scale: Factor applied to the coordinates (for example the voxel size in um), a number
            or one value per axis.

    Returns:
        Float array of shape (N, 2, 3), the coordinates times `scale`. An empty input gives
            shape (0, 2, 3).

    Raises:
        ValueError: If the segments do not have shape (N, 2, 3).

    Examples:
        >>> from microvasc.metrics.network import as_segments
        >>> as_segments([[[0, 0, 0], [1, 2, 3]]], scale=[1, 1, 2]).tolist()
        [[[0.0, 0.0, 0.0], [1.0, 2.0, 6.0]]]
    """
    if hasattr(network, "line_segments"):
        network = [(s.p[0], s.p[1]) for s in network.line_segments()]
    segments = np.asarray(network, dtype=float)
    if segments.size == 0:
        return segments.reshape(0, 2, 3)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"expected segments of shape (N, 2, 3), got {segments.shape}")
    return segments * np.broadcast_to(np.asarray(scale, dtype=float), (3,))


def voxel_skeleton_to_segments(mask: np.ndarray, voxel_size: ArrayLike = 1.0) -> np.ndarray:
    """
    Turn a voxel skeleton into line segments by joining every pair of 26-adjacent voxels.

    The output is in array axis order (axis 0, 1, 2) times the voxel size. A ground truth
    compared with it must use the same axis order and units.

    Where three or more voxels touch each other (junctions, some diagonal steps), all pairs are
    joined, so small triangles add some length. An isolated voxel becomes a zero-length segment:
    it can be matched, but adds no length. The paper does not describe how voxel skeletons were
    converted, so values from this conversion are not expected to equal the published ones.

    Args:
        mask: 3D skeleton. Any nonzero voxel is part of it.
        voxel_size: Voxel spacing (for example in um), a number or one value per axis.

    Returns:
        Float array of shape (N, 2, 3): array indices times `voxel_size`. An empty mask gives
            shape (0, 2, 3).

    Raises:
        ValueError: If `mask` is not 3D.

    Examples:
        >>> import numpy as np
        >>> from microvasc.metrics.network import voxel_skeleton_to_segments
        >>> mask = np.zeros((1, 1, 3), dtype=np.uint8)
        >>> mask[0, 0, :] = 1
        >>> voxel_skeleton_to_segments(mask, voxel_size=2.0).tolist()
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]]
    """
    mask = np.asarray(mask) != 0
    if mask.ndim != 3:
        raise ValueError(f"expected a 3D mask, got {mask.ndim} dimensions")
    spacing = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,))

    coords = np.argwhere(mask)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(len(coords))
    joined = np.zeros(len(coords), dtype=bool)
    pairs = []
    for offset in _FORWARD_26:
        other = coords + offset
        inside = np.all((other >= 0) & (other < mask.shape), axis=1)
        j = np.full(len(coords), -1, dtype=np.int64)
        j[inside] = index[tuple(other[inside].T)]
        i = np.flatnonzero(j >= 0)
        pairs.append(np.stack([i, j[i]], axis=1))
        joined[i] = True
        joined[j[i]] = True
    lone = np.flatnonzero(~joined)
    pairs.append(np.stack([lone, lone], axis=1))
    pairs = np.concatenate(pairs)
    return coords[pairs] * spacing


def _point_segment_distance(x, p0, p1):
    """Distance from each point x[i] to the segment p0[i]-p1[i]."""
    d = p1 - p0
    dd = np.einsum("ij,ij->i", d, d)
    t = np.einsum("ij,ij->i", x - p0, d) / np.where(dd > 0, dd, 1.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(x - (p0 + t[:, None] * d), axis=1)


def _distance_to_segments(points, segments, max_piece):
    """Exact distance from each point to the closest of the segments."""
    # split long segments, so a point's nearest segment is among those with a nearby midpoint
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    n = np.maximum(np.ceil(lengths / max_piece).astype(np.int64), 1)
    seg = np.repeat(np.arange(len(segments)), n)
    k = np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n)
    d = segments[seg, 1] - segments[seg, 0]
    p0 = segments[seg, 0] + d * (k / n[seg])[:, None]
    p1 = segments[seg, 0] + d * ((k + 1) / n[seg])[:, None]
    half = np.linalg.norm(p1 - p0, axis=1) / 2
    tree = cKDTree((p0 + p1) / 2)

    out = np.empty(len(points))
    for start in range(0, len(points), _CHUNK):
        x = points[start:start + _CHUNK]
        _, nearest = tree.query(x)
        best = _point_segment_distance(x, p0[nearest], p1[nearest])
        # any piece closer than `best` has its midpoint within best + half of x
        found = tree.query_ball_point(x, best + half.max())
        counts = np.fromiter(map(len, found), dtype=np.int64, count=len(found))
        cand = np.fromiter(itertools.chain.from_iterable(found), dtype=np.int64, count=counts.sum())
        owner = np.repeat(np.arange(len(x)), counts)
        np.minimum.at(best, owner, _point_segment_distance(x[owner], p0[cand], p1[cand]))
        out[start:start + _CHUNK] = best
    return out


def netmets_similarity(a: Union["NWT", ArrayLike], b: Union["NWT", ArrayLike],
                       sigma: float = 1.0, scale_a: ArrayLike = 1.0,
                       scale_b: ArrayLike = 1.0) -> float:
    """
    NetMets similarity M(a, b): the length-weighted fraction of a that lies close to b.

    Args:
        a: Network to measure, as accepted by `as_segments`.
        b: Network it is compared with, as accepted by `as_segments`.
        sigma: Sensitivity, in the unit of the scaled coordinates (the paper uses 1 um).
        scale_a: Factor that converts the coordinates of a to the unit of sigma (for example
            the voxel size in um), a number or one value per axis.
        scale_b: The same for b.

    Returns:
        A value in [0, 1]. When a has no length, 1.0 if b has no length either, and 0.0
            otherwise.

    Raises:
        ValueError: If sigma is not positive, or a network does not have shape (N, 2, 3).

    Examples:
        A line compared with a copy moved by sigma gives exp(-1/2):

        >>> from microvasc.metrics.network import netmets_similarity
        >>> a = [[[0, 0, 0], [10, 0, 0]]]
        >>> b = [[[0, 1, 0], [10, 1, 0]]]
        >>> round(netmets_similarity(a, b, sigma=1.0), 4)
        0.6065
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    a, b = as_segments(a, scale_a), as_segments(b, scale_b)
    lengths = np.linalg.norm(a[:, 1] - a[:, 0], axis=1)
    total = lengths.sum()
    if total == 0:
        b_length = np.linalg.norm(b[:, 1] - b[:, 0], axis=1).sum()
        return 1.0 if b_length == 0 else 0.0
    if len(b) == 0:
        return 0.0

    # midpoint rule along each segment of a, steps of at most sigma / 10
    keep = lengths > 0
    a, lengths = a[keep], lengths[keep]
    n = np.ceil(lengths / (sigma / 10)).astype(np.int64)
    seg = np.repeat(np.arange(len(a)), n)
    k = np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n)
    t = (k + 0.5) / n[seg]
    points = a[seg, 0] + (a[seg, 1] - a[seg, 0]) * t[:, None]
    dt = (lengths / n)[seg]

    d = _distance_to_segments(points, b, max_piece=sigma)
    return float(np.sum(dt * np.exp(-d**2 / (2 * sigma**2))) / total)


def netmets(ground_truth: Union["NWT", ArrayLike], test: Union["NWT", ArrayLike],
            sigma: float = 1.0, truth_scale: ArrayLike = 1.0,
            test_scale: ArrayLike = 1.0) -> Tuple[float, float]:
    """
    NetMets precision and recall of a test skeleton against a ground truth.

    The ground truth comes first, as in `SegmentationMetrics`. Precision P = M(test, ground_truth)
    is the fraction of the test skeleton that matches the ground truth. Recall
    R = M(ground_truth, test) is the fraction of the ground truth found in the test skeleton.
    Voxel skeletons must first go through `voxel_skeleton_to_segments`.

    Args:
        ground_truth: Ground truth network, as accepted by `as_segments`.
        test: Test network, as accepted by `as_segments`, with the same axis order as the
            ground truth.
        sigma: Sensitivity, in the unit of the scaled coordinates (the paper uses 1 um).
        truth_scale: Factor that converts the ground truth coordinates to the unit of sigma
            (for example its voxel size in um), a number or one value per axis.
        test_scale: The same for the test network.

    Returns:
        Precision and recall (P, R), each in [0, 1].

    Raises:
        ValueError: If sigma is not positive, or a network does not have shape (N, 2, 3).

    Examples:
        A test skeleton that finds the whole ground truth line and adds a false line far
        from it:

        >>> from microvasc.metrics.network import netmets
        >>> truth = [[[0, 0, 0], [10, 0, 0]]]
        >>> test = truth + [[[0, 50, 0], [10, 50, 0]]]
        >>> P, R = netmets(truth, test)
        >>> round(P, 4), round(R, 4)
        (0.5, 1.0)
    """
    return (netmets_similarity(test, ground_truth, sigma, test_scale, truth_scale),
            netmets_similarity(ground_truth, test, sigma, truth_scale, test_scale))
