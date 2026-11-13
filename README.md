# microvasc

Segmentation, skeletonization, and evaluation of 3D microvascular networks in Python.

[![PyPI](https://img.shields.io/pypi/v/microvasc)](https://pypi.org/project/microvasc/)
[![Downloads](https://static.pepy.tech/badge/microvasc)](https://pepy.tech/project/microvasc)
[![tests](https://github.com/helia77/MicroVasc-Review/actions/workflows/ci.yml/badge.svg)](https://github.com/helia77/MicroVasc-Review/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23933580.svg)](https://doi.org/10.5281/zenodo.23933580)

Companion code for [Segmentation and modeling of large-scale microvascular networks: a survey](https://doi.org/10.3389/fbinf.2025.1645520) (Frontiers in Bioinformatics, 2025).

<p align="center">
  <img src="docs/overview.png" alt="Overview of the binarization, skeletonization and evaluation steps" width="100%" />
</p>

## What it does

`microvasc` covers three steps of 3D microvascular image processing:

| Step | Methods |
|---|---|
| Binarization | Otsu (2D and 3D), Frangi, Beyond Frangi |
| Skeletonization | Lee, Palágyi, Kline, Kerautret |
| Evaluation | Jaccard, Dice, precision, recall, NetMets |

Kerautret calls the external CDCVAM program, which must be built separately. See [Skeletonization](docs/skeleton.md).

The paper evaluates these methods on KESM, LSFM, and micro-CT data.

The package includes a subset of the methods compared in the paper. Optimally Oriented Flux, U-Net, nnU-Net, Tagliasacchi, and Antiga are not included. The documentation lists them, with links to their implementations where available.

## Installation

Requires Python 3.9 or newer. Tests run on Python 3.12 and 3.14.

```bash
pip install microvasc
```

The core dependencies are NumPy, SciPy, and scikit-image.

Optional dependencies are available as extras:

| Extra | Installs | Needed for |
|---|---|---|
| `kline` | scikit-fmm | `skeleton kline` |
| `io` | opencv-python, pynrrd, SimpleITK | Image stacks, NRRD files, and resampling in `microvasc.data` |
| `test` | pytest | Running tests |
| `all` | All of the above | |

```bash
pip install "microvasc[kline]"
pip install "microvasc[all]"
```

On some platforms pip builds scikit-fmm from source, which needs a compiler and Python development headers.

To work on the code, install from a clone:

```bash
git clone https://github.com/helia77/MicroVasc-Review.git
cd MicroVasc-Review
pip install -e .
```

From a clone, use `pip install -e ".[kline]"` or `pip install -e ".[all]"` for the extras.

## Quick start

Binarize, skeletonize and evaluate a synthetic tube:

```python
import numpy as np
from scipy import ndimage

from microvasc.binarize import Binarizer
from microvasc.skeleton import Skeletonizer
from microvasc.metrics.geometry import SegmentationMetrics
from microvasc.metrics.network import netmets, voxel_skeleton_to_segments

# a bright tube along axis 0 on a dark, noisy background
_, y, x = np.mgrid[:32, :32, :32]
truth = (y - 16) ** 2 + (x - 16) ** 2 <= 4 ** 2
rng = np.random.default_rng(0)
volume = truth + 0.2 * rng.standard_normal(truth.shape)

mask = Binarizer(volume, background="black").otsu3d()

# keep the largest connected component, then fill the holes left by the noise
labels, _ = ndimage.label(mask)
mask = ndimage.binary_fill_holes(labels == np.bincount(labels.ravel())[1:].argmax() + 1)
print("Jaccard", SegmentationMetrics(truth, mask).jaccard())

skeleton = Skeletonizer().lee(mask)
centerline = [[[0, 16, 16], [31, 16, 16]]]      # the true centerline as one segment
precision, recall = netmets(centerline, voxel_skeleton_to_segments(skeleton), sigma=1.0)
print("NetMets precision", precision, "recall", recall)
```

The same steps from the command line, on `.npy` files:

```bash
binarize otsu3d --input volume.npy --output mask.npy
skeleton lee --input mask.npy --output skeleton.npy
```

## Documentation

The documentation is also published at https://www.helia-gohar.com/MicroVasc-Review/.

| Page | Contents |
|---|---|
| [Binarization](docs/binarize.md) | Otsu, Frangi, Beyond Frangi, CLI options, and external methods |
| [Skeletonization](docs/skeleton.md) | Lee, Palágyi, Kline, Kerautret, CLI options, and paper settings |
| [Metrics](docs/metrics.md) | Jaccard, Dice, precision, recall, and NetMets |
| [Parameter optimization](docs/optimization.md) | Slurm sweeps for the Frangi and Beyond Frangi parameters |
| [Data conversion](docs/data.md) | NWT, OBJ, OFF, NRRD, and image stack utilities |

For development setup and tests, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Data availability

The imaging data and annotations used in the paper are not in this repository. They are available from the authors on request.

## Citation

If you use `microvasc`, please cite the [paper](https://doi.org/10.3389/fbinf.2025.1645520) and the software through its Zenodo DOI, [10.5281/zenodo.23933580](https://doi.org/10.5281/zenodo.23933580).

```bibtex
@article{goharbavang2025segmentation,
  title={Segmentation and modeling of large-scale microvascular networks: a survey},
  author={Goharbavang, Helya and Ashitkov, Artem T and Pillai, Athira and Wythe, Joshua D and Chen, Guoning and Mayerich, David},
  journal={Frontiers in Bioinformatics},
  volume={5},
  pages={1645520},
  year={2025},
  publisher={Frontiers Media SA},
  doi={10.3389/fbinf.2025.1645520}
}
```

Citation metadata is also in [CITATION.cff](CITATION.cff), which GitHub shows as "Cite this repository".

## License

MIT License. See [LICENSE](LICENSE).

`Skeletonizer.kline` is derived from poreture (BSD 3-clause). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
