# Data conversion: `microvasc/data.py`

Helpers to read networks and convert volumes and meshes between the formats used by the methods.
Functions marked with `io` need the `io` extra (`pip install -e ".[io]"`). The others need only the
core install.

## Networks

| Name | Purpose |
|---|---|
| `NWT(filename)` | Reads a network from a `.nwt` or `.obj` file. `vertices` and `edges` hold the graph, `line_segments()` returns its line segments, `point_cloud(spacing)` samples points along them, `save_obj(output_name)` writes it as OBJ. |
| `sdp2obj(input_file)` | Converts the Kerautret output `<prefix>Vertex.sdp` and `<prefix>Edges.sdp` to `<prefix>.obj`. |
| `cg2obj(input_file)` | Converts a CG file from StarLab (Tagliasacchi method) to OBJ. |

NWT and OBJ networks can be passed to the NetMets metric, see [Metrics](metrics.md).

## Volumes and meshes

| Name | Purpose |
|---|---|
| `npy2obj(volume, output_name, level=0.0)` | Surface mesh of a volume (array or `.npy` path) with marching cubes, written as OBJ. |
| `npy2off(volume, output_name, level=0.5)` | Surface mesh of a binarized volume (array or `.npy` path) with marching cubes, written as OFF (input of `skeleton kerautret`). |
| `nrrd2npy(nrrd_path)` (`io`) | Reads an NRRD file into a NumPy array. |
| `npy2nrrd(arr, filename)` (`io`) | Writes a NumPy array as NRRD. |
| `resample(data, spacing, output_spacing, interpolator=None)` (`io`) | Resamples a volume to a new spacing with SimpleITK (linear by default). |

## Image stacks

| Name | Purpose |
|---|---|
| `load_images(folder_path, ...)` (`io`) | Loads the `.bmp`, `.jpg`, `.tif` and `.png` images of a folder, optionally cropped, as a list or a stacked array. |
| `save_slices(volume, folder_path, num_slices='all', file_format='bmp')` (`io`) | Saves the slices of a volume as images. |
| `adjust_brightness_contrast(image, clip_hist_percent=1)` (`io`) | Adjusts the brightness and contrast of an image by clipping its histogram. |
