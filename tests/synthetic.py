"""
Small synthetic volumes for tests.

All volumes are built from fixed seeds, so they can be recreated exactly. They are
8-bit, like the rescaled LSFM data, and bright vessels lie on a dark background (background='black').
Use ``255 - volume`` for the white-background case (KESM).
"""
import numpy as np
import scipy.ndimage as ndi


def capsule(shape, p, q, radius):
    """Boolean mask of all voxels within `radius` of the segment p-q (array index order)."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).astype(float)
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    t = np.clip(((grid - p) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(grid - p - t[..., None] * d, axis=-1) <= radius


def render(mask, seed, background=40, foreground=200, blur=1.0, noise_sd=12):
    """Turn a binary mask into a blurred, noisy uint8 image."""
    rng = np.random.default_rng(seed)
    v = mask.astype(float) * (foreground - background) + background
    v = ndi.gaussian_filter(v, blur) + rng.normal(0, noise_sd, mask.shape)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def tube_mask():
    """A straight, oblique tube of radius 3 in a 24^3 volume."""
    return capsule((24, 24, 24), (3, 5, 2), (20, 18, 21), 3.0)


def bifurcation_mask():
    """A Y-shaped bifurcation (radius 3 trunk, radius 2.5 branches) in a 28^3 volume."""
    shape = (28, 28, 28)
    return (capsule(shape, (2, 14, 14), (14, 14, 14), 3.0)
            | capsule(shape, (14, 14, 14), (25, 5, 6), 2.5)
            | capsule(shape, (14, 14, 14), (25, 23, 22), 2.5))


def tube():
    return render(tube_mask(), seed=1)


def bifurcation():
    return render(bifurcation_mask(), seed=2)


def noise():
    """Uniform 8-bit noise in a 16^3 volume, with no vessel at all."""
    return np.random.default_rng(3).integers(0, 256, (16, 16, 16), dtype=np.uint8)


VOLUMES = {
    "tube": tube,
    "bifurcation": bifurcation,
    "noise": noise,
}
