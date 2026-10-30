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

import numpy as np
from scipy.spatial import cKDTree

# the 13 neighbor offsets of a 26-neighborhood that point "forward", so each pair is found once
_FORWARD_26 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                        if (i, j, k) > (0, 0, 0)])
_CHUNK = 65536


def as_segments(network, scale=1.0) -> np.ndarray:
    """
    Line segments of a network as a float array of shape (N, 2, 3).

    Args:
        network: a manage_data.NWT object (loaded from .nwt or .obj), or an array-like of
            segments with shape (N, 2, 3).
        scale: factor applied to the coordinates (for example the voxel size in um), a number
            or one value per axis.
    """
    if hasattr(network, "linesegments"):
        network = [(s.p[0], s.p[1]) for s in network.linesegments()]
    segments = np.asarray(network, dtype=float)
    if segments.size == 0:
        return segments.reshape(0, 2, 3)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"expected segments of shape (N, 2, 3), got {segments.shape}")
    return segments * np.broadcast_to(np.asarray(scale, dtype=float), (3,))


def voxel_skeleton_to_segments(mask: np.ndarray, voxel_size=1.0) -> np.ndarray:
    """
    Turn a voxel skeleton into line segments by joining every pair of 26-adjacent voxels.

    The output is in array axis order (axis 0, 1, 2) times the voxel size. A ground truth
    compared with it must use the same axis order and units.

    Where three or more voxels touch each other (junctions, some diagonal steps), all pairs are
    joined, so small triangles add some length. An isolated voxel becomes a zero-length segment:
    it can be matched, but adds no length. The paper does not describe how voxel skeletons were
    converted, so values from this conversion are not expected to equal the published ones.

    Args:
        mask: 3D skeleton; any nonzero voxel is part of it.
        voxel_size: voxel spacing in um, a number or one value per axis.

    Returns:
        float array of shape (N, 2, 3), in um.
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


def netmets_similarity(A, B, sigma: float = 1.0, scale_a=1.0, scale_b=1.0) -> float:
    """
    NetMets similarity M(A, B): the length-weighted fraction of A that lies close to B.

    Args:
        A, B: networks, as accepted by `as_segments`.
        sigma: sensitivity, in the unit of the scaled coordinates (the paper uses 1 um).
        scale_a, scale_b: factors that convert the coordinates of A and B to that unit
            (for example the voxel size in um), a number or one value per axis.

    Returns:
        A value in [0, 1]. When A has no length, returns 1.0 if B has no length either,
        and 0.0 otherwise.
    """
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    A, B = as_segments(A, scale_a), as_segments(B, scale_b)
    lengths = np.linalg.norm(A[:, 1] - A[:, 0], axis=1)
    total = lengths.sum()
    if total == 0:
        b_length = np.linalg.norm(B[:, 1] - B[:, 0], axis=1).sum()
        return 1.0 if b_length == 0 else 0.0
    if len(B) == 0:
        return 0.0

    # midpoint rule along each segment of A, steps of at most sigma / 10
    keep = lengths > 0
    A, lengths = A[keep], lengths[keep]
    n = np.ceil(lengths / (sigma / 10)).astype(np.int64)
    seg = np.repeat(np.arange(len(A)), n)
    k = np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n)
    t = (k + 0.5) / n[seg]
    points = A[seg, 0] + (A[seg, 1] - A[seg, 0]) * t[:, None]
    dt = (lengths / n)[seg]

    d = _distance_to_segments(points, B, max_piece=sigma)
    return float(np.sum(dt * np.exp(-d**2 / (2 * sigma**2))) / total)


def netmets(test, ground_truth, sigma: float = 1.0, test_scale=1.0, truth_scale=1.0):
    """
    NetMets precision and recall of a test skeleton against a ground truth.

    Precision P = M(test, ground_truth) is the fraction of the test skeleton that matches the
    ground truth. Recall R = M(ground_truth, test) is the fraction of the ground truth found in
    the test skeleton. Voxel skeletons must first go through `voxel_skeleton_to_segments`.

    Args:
        test, ground_truth: networks, as accepted by `as_segments`, with the same axis order.
        sigma: sensitivity, in the unit of the scaled coordinates (the paper uses 1 um).
        test_scale, truth_scale: factors that convert each input's coordinates to the unit of
            sigma (for example its voxel size in um), a number or one value per axis.

    Returns:
        (P, R)
    """
    return (netmets_similarity(test, ground_truth, sigma, test_scale, truth_scale),
            netmets_similarity(ground_truth, test, sigma, truth_scale, test_scale))
