# Skeletonization

![Skeletonization](skeleton.gif)

`skeleton` implements four of the skeletonization methods from the paper: Lee thinning, Palágyi
thinning, Kline's fast marching method, and a wrapper around the external Kerautret program.

`lee`, `palagyi` and `kline` read a binarized `.npy` volume (any nonzero voxel is foreground) and
write a uint8 `.npy` skeleton of 0 and 1 with the same shape.

## Lee: `lee`

Lee thinning with `skimage.morphology.skeletonize(volume, method='lee')`
([scikit-image source](https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/_skeletonize.py)).

```bash
skeleton lee --input path/to/binary.npy --output path/to/skeleton.npy
```

## Palágyi: `palagyi`

The 12-subiteration curve thinning of K. Palagyi and A. Kuba, A Parallel 3D 12-Subiteration
Thinning Algorithm, Graphical Models and Image Processing 61, 199-221, 1999
([doi:10.1006/gmip.1999.0498](https://doi.org/10.1006/gmip.1999.0498)). The deletion templates are
stored as a lookup table in `microvasc/PK12.npz`, which is installed with the package and
used by default. `--templates` selects another table.

```bash
skeleton palagyi --input path/to/binary.npy --output path/to/skeleton.npy
```

Axis 0 points east, axis 1 north and axis 2 up. Border points are found once per iteration and
reused in all 12 subiterations. The paper checks every point again in each subiteration.

## Kline: `kline`

Kline's fast marching centerline extraction. It needs the `kline` extra
(`pip install -e ".[kline]"`). `--root` is the index of the root voxel along array axes 0, 1 and 2
and must be a vessel voxel. `--dist-map-weight` is the distance map weight (default 6) and
`--min-branch-to-root` the minimum distance from an endpoint to the root, in voxels (default 10). These are the settings used
for the KESM volume in the paper:

```bash
skeleton kline \
  --input path/to/binary.npy \
  --output path/to/centerline.npy \
  --root 95 76 35 \
  --dist-map-weight 19.25 \
  --min-branch-to-root 10.11
```

With a distance map weight of 19.25 the result depends on floating point rounding, so the skeleton
can differ between machines or library builds. Smaller weights, like the default 6, are stable.

The method follows the code used for the paper, which is based on
[poreture](https://github.com/TLKline/poreture). It differs from poreture's `kline_vessel` in two
ways: both fast marches use an all-ones speed array over the bounding box (poreture uses the vessel
mask), and the `cluster_graph_weight` and `min_branch_length` options are not parameters here
(poreture reads them but does not use them).

## Kerautret: `kerautret`

`skeleton kerautret` runs the `centerLineGeodesicGraph` program from
[CDCVAM](https://github.com/kerautret/CDCVAM), which must be built separately. The input is a
surface mesh of the binarized volume in OFF format.

Installation:

```bash
# Requirements: CMake, a C++ compiler, Boost, and DGtal (https://github.com/DGtal-team/DGtal)
git clone https://github.com/kerautret/CDCVAM.git
cd CDCVAM
mkdir build && cd build
cmake ..
make
# The program is build/bin/centerLineGeodesicGraph
```

Usage:

The input meshes in the paper were `binary_<name>.off`, made from the binarized volumes. This
example uses the KESM settings from the table below:

```bash
# Make the OFF mesh from a binarized volume (marching cubes)
python -c "from microvasc.data import npy2off; npy2off('path/to/binary_kesm.npy', 'path/to/binary_kesm.off')"

skeleton kerautret \
  --input path/to/binary_kesm.off \
  --output path/to/out/tree \
  --exe-path path/to/CDCVAM/build/bin/centerLineGeodesicGraph \
  --dilate-dist 1.0 --delta-g 3.3 --radius 1.1 --threshold 0.5
# Writes treeVertex.sdp, treeEdges.sdp, treeRadius.sdp and tree.obj in path/to/out/
```

`--radius` is the accumulation distance. The paper sets it slightly larger than the radius of the
largest vessels.

Settings used in the paper:

| Dataset  | `--dilate-dist` | `--delta-g` | `--radius` | `--threshold` |
|----------|-----------------|-------------|------------|---------------|
| KESM     | 1.0             | 3.3         | 1.1        | 0.5           |
| LSFM     | 1.0             | 3.1         | 0.7        | 0.5           |
| micro-CT | 1.1             | 2.75        | 0.8        | 0.6           |

Values are as used in the paper, in the coordinate units of the input mesh. Without the options,
the defaults are 2.0, 3.0, 10.0 and 0.5, as in CDCVAM.

## Python

```python
from microvasc.skeleton import Skeletonizer

ske = Skeletonizer()
skeleton = ske.lee(mask)
skeleton = ske.palagyi(mask)
centerline = ske.kline(mask, (95, 76, 35), dist_map_weight=19.25, min_branch_to_root=10.11)
obj_path = ske.kerautret("path/to/centerLineGeodesicGraph", "path/to/mesh.off", "path/to/out/tree")
```

## External skeletonization methods

The paper also covers methods that are not implemented here:

* **Mean curvature flow (Tagliasacchi)**: [https://github.com/taiya/starlab-mcfskel](https://github.com/taiya/starlab-mcfskel).
  `microvasc.data.cg2obj` converts its CG output to OBJ (see [Data conversion](data.md)).
* **Voronoi-based (Antiga)**: see the paper.
* **3D Slicer**: [https://www.slicer.org](https://www.slicer.org)

Skeletons are evaluated with NetMets, see [Metrics](metrics.md).
