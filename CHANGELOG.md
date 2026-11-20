# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Documentation site at https://www.helia-gohar.com/MicroVasc-Review/, built with MkDocs and
  deployed from `main` by GitHub Actions.

### Fixed

- `Binarizer.frangi` and `Binarizer.bfrangi` raise `ValueError` for a `threshold` other than
  `None`, `'otsu2d'` or `'otsu3d'`. Before, an unknown value was ignored and the vesselness was
  returned.
- `SegmentationMetrics` raises `ValueError` when the ground truth and the prediction have
  different shapes. Before, shapes that broadcast were accepted and could give wrong values, such
  as an accuracy above 1.
- `netmets` and `netmets_similarity` used a lot of memory when parts of the two networks were far
  apart, for example with a wrong scale. The distance search now stops at 40 sigma, where the
  weight is already exactly 0, so the results are unchanged.
- `netmets`, `netmets_similarity`, `as_segments` and `voxel_skeleton_to_segments` check their
  inputs and raise `ValueError` with the name of the argument: sigma must be a finite number
  greater than 0, coordinates must be finite, and a scale or voxel size must be one finite number
  or 3 finite values. Before, `sigma=inf` returned 0.0, NaN coordinates gave NaN or a SciPy error,
  and a scale of 2 values gave a NumPy broadcast error.
- `load_images` reads the files in natural sort order of their names, so img_2 comes before
  img_10. Before, it used the unsorted order of `os.listdir`, and slices could come back in any
  order. `save_slices` now writes zero-padded file names (img_00 to img_11 for 12 slices), so
  plain sorting also keeps the order. Folders with the old unpadded names still load in order.
- `save_slices` no longer deletes its output folder. Before, an existing folder was removed with
  everything in it. The default behavior changed: a folder that already holds files now raises
  `FileExistsError`. With the new `overwrite=True`, only earlier slice files of the same format
  (`img_<number>.<file_format>`) are removed, and all other files are kept. A missing folder is
  still created.
- `save_slices` converts the values with one rule for the whole volume. Before, each slice was
  converted on its own: a slice whose maximum was 1 was multiplied by 255 and the others were
  not, so in a 0..11 volume the slice of 1s was saved as 255 and the slice of 2s as 2. Now values
  in [0, 1] are multiplied by 255 and rounded (a 0/1 mask is still saved as 0/255), and integers
  in 0..255 are saved unchanged. The saved values can differ from before: 0..255 volumes with
  slices of only 0 and 1 keep those slices as 0 and 1, and float volumes in [0, 1] are no longer
  saved as zeros. Values above 255, negative values and NaN now raise `ValueError` instead of
  being wrapped.
- `cg2obj` replaced every 'e' in the file with 'l', so numbers such as `1e-05` became `1l-05`.
  It now replaces only the leading 'e' of edge lines and copies everything else unchanged. The
  output is the input path with its `.cg` suffix (any letter case) replaced by `.obj`, and a name
  without a `.cg` suffix raises `ValueError`. Before, such a name made `cg2obj` overwrite its
  input file, and a `.cg` elsewhere in the path, as in `x.cgdir/a.cg`, changed the output folder.
  `cg2obj` also accepts a `Path`.
- `npy2obj` accepts the path of the `.npy` file as a `Path`, like `npy2off`. Before, only a
  string was read, and a `Path` raised `ValueError`.
- `NWT` closes the `.nwt` or `.obj` file after reading it, also when the file cannot be parsed.
  Before, `load_nwt` and `load_obj` never closed it, which gave a `ResourceWarning`.

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
