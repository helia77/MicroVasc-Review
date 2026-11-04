# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] - 2026-11-03

### Added

- Test suite (pytest) and a GitHub Actions workflow for Python 3.12 and 3.14, with and without
  scikit-fmm.
- NetMets precision and recall for skeletons in `microvasc.metrics.network`:
  `netmets(ground_truth, test)`, which takes the ground truth first like
  `SegmentationMetrics(ground_truth, predicted)`, `netmets_similarity`,
  `voxel_skeleton_to_segments` and `as_segments`.
- `recall()` for the segmentation metrics.
- `microvasc.data.npy2off` writes an OFF surface mesh with marching cubes, the input of
  `skeleton kerautret`.
- `skeleton kline` options `--dist-map-weight` and `--min-branch-to-root`.
- `microvasc/optimization/submit.sh` submits a parameter sweep as a Slurm job array sized from the
  sweep. `parameter_fscore` gains `--scale` (required), `--background`, `--alpha`, `--c` and `--jobs`.
- `microvasc.__version__`.
- Documentation pages in `docs/` and a quick start example in the README.
- `THIRD_PARTY_NOTICES.md` with the poreture license (BSD 3-clause) for the Kline code.

### Changed

- The distribution is `microvasc` (was `microvasc_review`) and all code is in one package:
  `microvasc.binarize`, `microvasc.skeleton`, `microvasc.data`, `microvasc.metrics.geometry`,
  `microvasc.metrics.network` and `microvasc.optimization`. The old `implementations`,
  `manage_data`, `metrics` and `optimization` imports are gone. The `binarize` and `skeleton`
  commands keep their names.
- Packaging uses `pyproject.toml`. Only NumPy, SciPy and scikit-image are required. scikit-fmm
  (`kline` extra) and OpenCV, pynrrd and SimpleITK (`io` extra) are optional and imported only by
  the functions that use them.
- Public names follow PEP 8:
  - `Skeletonize` is `Skeletonizer`, with `kline(volume, root, dist_map_weight, min_branch_to_root)`
    and `kerautret(exe_path, off_path, output_prefix, dilate_dist, delta_g, radius, threshold)`.
  - `metric` is `SegmentationMetrics(ground_truth, predicted)`, with counts `tp`, `tn`, `fp`, `fn`.
  - `Binarizer.frangi` and `Binarizer.bfrangi` take `scales` (was `scale_range`).
  - `netmets_similarity(a, b)`.
  - `microvasc.data`: `Vertex`, `Edge`, `LineSegment`, `Vertex.e_out` and `Vertex.e_in`,
    `NWT.vertices`, `NWT.edges`, `NWT.line_segments()`, `NWT.point_cloud()`,
    `load_images(image_range=...)`, `save_slices(num_slices=...)`, `npy2obj(volume, ...)` and
    `npy2off(volume, ...)`.
  - `microvasc.optimization.parameter_fscore`: `frangi_fscore`, `bfrangi_fscore`, and `num_values`
    for `alpha_c_ranges`, `beta_range`, `tau_range` and `job_count`.
- `binarize frangi` takes `--alpha`, `--beta` and `--c`, and `binarize bfrangi` takes `--tau` (both
  were `--params`). `--th` is `--threshold`.
- `skeleton kerautret` takes `--dilate-dist`, `--delta-g`, `--radius` and `--threshold` (was
  `--params`).
- `skeleton palagyi --templates` is optional and defaults to the lookup table installed with the
  package. The table is stored as packed bits (`PK12.npz`, 16 KB, was a 64 MB `PK12.npy`). A `.npy`
  table still works.
- `palagyi` is a new implementation of Palágyi and Kuba (1999), with a bundled lookup table
  (`PK12.npz`).
- `lee`, `palagyi` and `kline` treat any nonzero voxel as foreground and return uint8 arrays of 0
  and 1.
- The Frangi and Beyond Frangi filters are faster, with the same results.
- The optimization scripts run as `python -m microvasc.optimization.parameter_fscore` and
  `python -m microvasc.optimization.parameter_concat`, and their options use dashes
  (`--input-volume`, `--ground-truth`, `--output-dir`, `--array-id`, `--step-size`, `--input-dir`,
  and `--num-values`, was `--ranges`). `parameter.sbatch` takes
  `PARAMETER INPUT_VOLUME GROUND_TRUTH OUTPUT_DIR` followed by options, and is submitted with
  `submit.sh`.
- Importing the package no longer configures logging. The commands still do.
- The README is shorter. Details are in the documentation pages.

### Fixed

- `lee` and `palagyi` could not run. `lee` uses `skimage.morphology.skeletonize(method="lee")`.
- `kline` crashed before its first fast march, and `skeleton kline` did not save its result. It now
  follows the code used for the paper and reports a root outside the volume or on background.
- `kerautret` passed its arguments in the wrong order and did not handle the output prefix. A
  missing executable gives a clear error.
- `Binarizer.frangi` and `Binarizer.bfrangi` changed the volume and background stored in the
  object, so later calls on the same object gave different results.
- `binarize` accepted the misspelled threshold `otus2d` and then returned the unthresholded filter
  output.
- `f_score()` crashed. Jaccard, Dice, precision and recall return 1.0 when both masks are empty and
  0.0 for any other zero denominator. Jaccard returned 1.0 for an empty ground truth even with
  false positives.
- `NWT` read the edge points of `.nwt` files as pairs instead of (x, y, z, radius), so
  `line_segments()` failed.
- The optimization scripts could not run, and the beta sweep started more jobs than it has values.
  `parameter_concat` sizes its output from the result files found.
- SciPy imports use `scipy.ndimage` instead of the deprecated `scipy.ndimage.filters` and
  `scipy.ndimage.morphology`.
- The LICENSE holder name matches the paper.
- README: the quick start command, and descriptions of `manage_data` and the optimization scripts
  that did not match the code.

### Removed

- `setup.py`, replaced by `pyproject.toml`.
- `TPR()` and `FPR()` of the segmentation metrics. Use `recall()` and `fall_out()`.
- `--ranges` of `parameter_concat`. Array sizes come from the result files.

[Unreleased]: https://github.com/helia77/MicroVasc-Review/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/helia77/MicroVasc-Review/releases/tag/v0.2.0
