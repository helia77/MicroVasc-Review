# Metrics

## Segmentation: `microvasc/metrics/geometry.py`

Voxel-wise comparison of a binary prediction with a binary ground truth.

```python
from microvasc.metrics.geometry import metric

m = metric(ground_truth, prediction)
m.jaccard(), m.dice(), m.precision(), m.recall()
```

| Method | Formula |
|---|---|
| `jaccard()` | TP / (TP + FP + FN) |
| `dice()`, `f_score()` | 2TP / (2TP + FP + FN) |
| `precision()` | TP / (TP + FP) |
| `recall()`, `TPR()` | TP / (TP + FN) |

When both masks are empty, Jaccard, Dice, precision and recall return 1.0. Any other zero
denominator gives 0.0. The F-score and the Jaccard index are related by J = F / (2 - F).

## Skeletons: NetMets, `microvasc/metrics/network.py`

The NetMets geometry metric of D. Mayerich, C. Bjornsson, J. Taylor and B. Roysam, NetMets:
software for quantifying and visualizing errors in biological network segmentation, BMC
Bioinformatics 13(Suppl 8):S7, 2012. For two sets of curves A and B,

    M(A, B) = (1 / L) sum_{a in A} integral exp(-d(a(t), B)^2 / (2 sigma^2)) dt

where L is the total length of A, d(x, B) is the distance from x to the closest point on B and
sigma is the sensitivity. For a test skeleton T and a ground truth G, precision is M(T, G), the
fraction of T close to G, and recall is M(G, T), the fraction of G found in T. The paper uses
sigma = 1 um.

The integral uses the midpoint rule with steps of at most sigma / 10, and d is the exact distance
to the nearest segment of B. The NetMets software measures d between sample points, so values may
differ slightly from it. When a network has no length, M is 1.0 if the other has no length either
and 0.0 otherwise.

### Inputs

A network can be:

* a `microvasc.data.NWT` object, loaded from a `.nwt` or `.obj` file,
* an array of line segments of shape (N, 2, 3),
* a voxel skeleton turned into segments with `voxel_skeleton_to_segments(mask, voxel_size)`.

### Scales

Coordinates and sigma must use the same unit. Each input has its own scale factor to that unit, a
number or one value per axis, so a voxel skeleton and a ground truth stored in different units can
be compared directly:

```python
from microvasc.data import NWT
from microvasc.metrics.network import netmets, voxel_skeleton_to_segments

test = voxel_skeleton_to_segments(skeleton)        # voxel units, array axes 0, 1, 2
truth = NWT("path/to/ground_truth.obj")
P, R = netmets(test, truth, sigma=1.0,
               test_scale=voxel_size_um,           # number, or one value per axis
               truth_scale=truth_unit_um)
```

`netmets_similarity(A, B, sigma, scale_a, scale_b)` computes M(A, B) alone.

Give the voxel size either to `voxel_skeleton_to_segments` or as `test_scale`, not both. Both
inputs must use the same axis order: `voxel_skeleton_to_segments` returns array axes 0, 1 and 2.

### Voxel skeletons

`voxel_skeleton_to_segments` joins every pair of 26-adjacent skeleton voxels with a segment. Where
three or more voxels touch each other (junctions, some diagonal steps), all pairs are joined, so
small triangles add some length. The paper does not describe how its voxel skeletons were
converted, so values from this conversion are not expected to equal the published ones.
