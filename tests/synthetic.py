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


# centerlines of the tube and bifurcation, as (start, end) segments in array index order
TUBE_AXIS = [((3, 5, 2), (20, 18, 21))]
BIFURCATION_AXIS = [((2, 14, 14), (14, 14, 14)), ((14, 14, 14), (25, 5, 6)), ((14, 14, 14), (25, 23, 22))]


def tube_mask():
    """A straight, oblique tube of radius 3 in a 24^3 volume."""
    return capsule((24, 24, 24), *TUBE_AXIS[0], 3.0)


def bifurcation_mask():
    """A Y-shaped bifurcation (radius 3 trunk, radius 2.5 branches) in a 28^3 volume."""
    shape = (28, 28, 28)
    (trunk, branch1, branch2) = BIFURCATION_AXIS
    return (capsule(shape, *trunk, 3.0)
            | capsule(shape, *branch1, 2.5)
            | capsule(shape, *branch2, 2.5))


def torus_mask(shape=(14, 32, 32), major=9.0, minor=3.0):
    """A torus around axis 0, centered in the volume, with one hole."""
    z, y, x = np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij")
    return np.hypot(np.hypot(y, x) - major, z) <= minor


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


# Small networks for the NWT and OBJ readers

# vertex positions (x, y, z)
NETWORK_VERTICES = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (15.0, 5.0, 0.0), (15.0, -5.0, 2.0)]
# (start vertex, end vertex, interior points (x, y, z, radius))
NETWORK_EDGES = [
    (0, 1, [(3.0, 0.5, 0.0, 1.5), (6.0, -0.5, 0.0, 1.25)]),
    (1, 2, [(12.0, 2.0, 0.5, 1.0)]),
    (1, 3, []),
]


def write_obj(path, vertices=NETWORK_VERTICES, edges=NETWORK_EDGES):
    """Write a network as OBJ: one `v` line per point and one `l` polyline per edge (1-based)."""
    points = [tuple(v) for v in vertices]
    lines = []
    for v0, v1, interior in edges:
        ids = [v0 + 1]
        for p in interior:
            points.append(tuple(p[:3]))
            ids.append(len(points))
        lines.append(ids + [v1 + 1])
    with open(path, "w") as f:
        f.writelines(f"v {x} {y} {z}\n" for x, y, z in points)
        f.writelines("l " + " ".join(map(str, ids)) + "\n" for ids in lines)


def write_nwt(path, vertices=NETWORK_VERTICES, edges=NETWORK_EDGES, desc="synthetic network"):
    """
    Write a network in the binary NWT layout read by manage_data.NWT: a 14-byte header,
    a 58-byte description, vertex and edge counts (uint32), then for each vertex its position
    (3 float32), outgoing and incoming edge counts and indices (uint32), and for each edge its
    two vertex indices, point count (uint32) and points as (x, y, z, radius) float32.
    """
    e_out = [[i for i, e in enumerate(edges) if e[0] == v] for v in range(len(vertices))]
    e_in = [[i for i, e in enumerate(edges) if e[1] == v] for v in range(len(vertices))]
    with open(path, "wb") as f:
        f.write(b"nwtfileformat ")
        f.write(desc.encode("utf-8").ljust(58)[:58])
        np.array([len(vertices), len(edges)], np.uint32).tofile(f)
        for v, p in enumerate(vertices):
            np.array(p, np.float32).tofile(f)
            np.array([len(e_out[v]), len(e_in[v]), *e_out[v], *e_in[v]], np.uint32).tofile(f)
        for v0, v1, interior in edges:
            np.array([v0, v1, len(interior)], np.uint32).tofile(f)
            np.array(interior, np.float32).reshape(-1, 4).tofile(f)
