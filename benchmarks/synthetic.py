"""
Synthetic vessel trees for the benchmarks.

A tree of tubes grows from one root until vessels fill a given fraction of a cubic volume.
Radii are in voxels and do not depend on the volume size, so a larger volume holds more
branches of the same width. The tree is one connected component, and its root voxel is in
the mask, so it can be used as the Kline root.
"""
from typing import Tuple

import numpy as np
import scipy.ndimage as ndi

ROOT_RADIUS = 4.0
MIN_RADIUS = 1.5
RADIUS_FACTOR = 0.85
SEGMENT_LENGTH = (12.0, 30.0)
TURN_SD = 0.6
MAX_SEGMENTS = 100_000


def _add_capsule(mask: np.ndarray, p: np.ndarray, q: np.ndarray, radius: float) -> int:
    """Set all voxels within `radius` of the segment p-q. Returns the number of new voxels."""
    lo = np.maximum(np.floor(np.minimum(p, q) - radius).astype(int), 0)
    hi = np.minimum(np.ceil(np.maximum(p, q) + radius).astype(int) + 1, mask.shape)
    grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), axis=-1)
    d = q - p
    t = np.clip(((grid - p) @ d) / (d @ d), 0, 1)
    inside = np.linalg.norm(grid - p - t[..., None] * d, axis=-1) <= radius
    box = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
    new = int(np.count_nonzero(inside & ~box))
    box |= inside
    return new


def _end_point(p: np.ndarray, direction: np.ndarray, length: float, low: float,
               high: float) -> Tuple[np.ndarray, np.ndarray]:
    """End of a segment from p, clipped to [low, high], and the direction for its children.
    The direction is reversed on every axis where the segment was clipped."""
    q = p + direction * length
    outside = (q < low) | (q > high)
    direction = np.where(outside, -direction, direction)
    return np.clip(q, low, high), direction


def _render(mask: np.ndarray, rng: np.random.Generator, background: float = 40,
            foreground: float = 200, blur: float = 1.0, noise_sd: float = 12) -> np.ndarray:
    """Bright vessels on a dark background, blurred, with Gaussian noise, as uint8."""
    v = mask.astype(float) * (foreground - background) + background
    v = ndi.gaussian_filter(v, blur) + rng.normal(0, noise_sd, mask.shape)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def vessel_tree(size: int, fraction: float = 0.03,
                seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    A branching tree of tubes in a cubic volume, with a fixed seed.

    The root is near the middle of the first face along axis 0, and the first segment points
    along axis 0. Each segment ends in two children that turn by a random angle. The radius is
    4 voxels at the root and shrinks by a factor 0.85 per level, down to 1.5 voxels. Segments
    are 12 to 30 voxels long. A segment stops at the volume border and its children turn back
    into the volume. Branches are added level by
    level until the vessel fraction reaches `fraction`, so the result is at or a little above it.

    Args:
        size: Edge length of the volume in voxels, at least 32.
        fraction: Target fraction of vessel voxels, in (0, 0.5].
        seed: Seed for the tree and for the noise.

    Returns:
        volume: uint8 image, vessels about 200 on a background of about 40 (background='black').
        mask: bool vessel mask, the same shape.
        root: index of the root voxel, which is in the mask.

    Raises:
        ValueError: If size or fraction is out of range.
        RuntimeError: If the tree stops growing before it reaches `fraction`.
    """
    if size < 32:
        raise ValueError(f"size must be at least 32, got {size}")
    if not 0 < fraction <= 0.5:
        raise ValueError(f"fraction must be in (0, 0.5], got {fraction}")

    rng = np.random.default_rng(seed)
    mask = np.zeros((size,) * 3, dtype=bool)
    target = fraction * mask.size
    low, high = ROOT_RADIUS + 1, size - 2 - ROOT_RADIUS
    start = np.array([low, (size - 1) / 2, (size - 1) / 2])
    root = tuple(int(i) for i in np.rint(start))

    count = 0
    level = [(start, np.array([1.0, 0.0, 0.0]))]
    depth = 0
    segments = 0
    while count < target:
        radius = max(MIN_RADIUS, ROOT_RADIUS * RADIUS_FACTOR ** depth)
        children = []
        for p, direction in level:
            q, direction = _end_point(p, direction, rng.uniform(*SEGMENT_LENGTH), low, high)
            if np.allclose(p, q):
                continue
            count += _add_capsule(mask, p, q, radius)
            segments += 1
            for _ in range(2):
                turned = direction + rng.normal(0, TURN_SD, 3)
                children.append((q, turned / np.linalg.norm(turned)))
            if count >= target:
                break
        if segments > MAX_SEGMENTS or not children:
            raise RuntimeError(f"no tree reached {fraction} vessels in {size}^3")
        level = children
        depth += 1

    return _render(mask, rng), mask, root
