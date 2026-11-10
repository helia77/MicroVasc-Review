# microvasc

Python code to binarize 3D images of microvascular networks, skeletonize the binary volumes, and
evaluate both against a ground truth. It is the code for the survey paper cited below.

```bash
pip install microvasc
```

| Page | Contents |
|---|---|
| [Binarization](binarize.md) | Otsu, Frangi and Beyond Frangi, `binarize` options, external methods |
| [Skeletonization](skeleton.md) | Lee, Palágyi, Kline and Kerautret, `skeleton` options, settings used in the paper |
| [Metrics](metrics.md) | Jaccard, Dice, precision, recall, and NetMets with per-input scales |
| [Parameter optimization](optimization.md) | Slurm sweeps for alpha, c, beta and tau, and J = F / (2 - F) |
| [Data conversion](data.md) | NWT, OBJ, OFF, NRRD and image stack helpers in `microvasc.data` |

## Citation

Paper: Goharbavang et al., "Segmentation and modeling of large-scale microvascular networks: a
survey", Frontiers in Bioinformatics, 2025
([doi:10.3389/fbinf.2025.1645520](https://doi.org/10.3389/fbinf.2025.1645520)).

Paper project page: [helia-gohar.com/project/microvascular-modeling-review](https://www.helia-gohar.com/project/microvascular-modeling-review/).

Software, all versions: [doi:10.5281/zenodo.23933580](https://doi.org/10.5281/zenodo.23933580).
