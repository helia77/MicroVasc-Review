# Binarization

<div>
<model-viewer id="viewer-segmentation" src="https://www.helia-gohar.com/data/mesh/ground_segmentation.glb" alt="3D model of a segmented microvascular network" camera-controls shadow-intensity="1" scale="-1 1 -1" style="width: 100%; height: 400px"></model-viewer>
</div>

`binarize` implements five of the binarization methods from the paper: Otsu's thresholding in
2D and 3D, Frangi vesselness, Beyond Frangi and optimally oriented flux (OOF). All subcommands
read a 3D `.npy` volume and write a `.npy` file.

| Option | Meaning |
|---|---|
| `--input` | input volume (`.npy`) |
| `--output` | output file (`.npy`) |
| `--background` | `black` (default) for bright vessels on a dark background, `white` for dark vessels on a bright background |

## Otsu: `otsu3d` and `otsu2d`

Otsu thresholding with 256 histogram bins, over the whole volume (`otsu3d`) or for each slice
along axis 0 (`otsu2d`). The output is a uint8 mask of 0 and 1.

```bash
binarize otsu3d --input path/to/volume.npy --output path/to/mask.npy --background black
```

## Frangi: `frangi`

Frangi vesselness. `--alpha`, `--beta` and `--c` set the three Frangi parameters and `--scale` the
Gaussian scales (in voxels). The response is the maximum over the scales, in [0, 1]. With
`--threshold otsu2d` or `--threshold otsu3d` the response is thresholded with Otsu and the output is
a uint8 mask.

```bash
binarize frangi \
  --input path/to/volume.npy \
  --output path/to/out.npy \
  --background white \
  --alpha 0.5 --beta 0.5 --c 3 \
  --scale 1 2 3 4 \
  --threshold otsu3d
```

The paper's baseline Frangi settings use c equal to half of the maximum Hessian norm. This code
takes c as an explicit value and does not compute it.

## Beyond Frangi: `bfrangi`

Beyond Frangi (Jerman et al.). `--tau` sets the regularization parameter. `--scale` and
`--threshold` work as for `frangi`.

```bash
binarize bfrangi \
  --input path/to/volume.npy \
  --output path/to/out.npy \
  --tau 0.75 \
  --scale 1 2 3 \
  --threshold otsu3d
```

The regularization uses tau times the largest |lambda3| at each scale, taken before the sign is
flipped for the background. Jerman's reference implementation
([JermanEnhancementFilter](https://github.com/timjerman/JermanEnhancementFilter)) uses the signed
maximum after the flip, so results can differ between the two.

## Optimally oriented flux: `oof`

Optimally oriented flux (Law and Chung, ECCV 2008), ported from Law's MATLAB `oof3response`. At
each radius the response is the eigenvalue of the oriented flux matrix with the largest
magnitude. Radii are compared with the curvilinear normalization of Law et al. (ECCV 2012), and
each voxel keeps the signed response with the largest magnitude over the radii. Vessels are
positive for either background: for `white` the volume is negated first.

| Option | Meaning |
|---|---|
| `--radii` | radii, in the units of `--spacing` (voxels with the default) |
| `--sigma` | Gaussian smoothing, in the units of `--spacing` (default 1) |
| `--spacing` | voxel size along axes 0, 1 and 2, as one number or 3 values (default 1) |
| `--threshold` | `otsu2d`, `otsu3d` or a number t (output > t). Without it the float response is saved |
| `--no-pad` | do not pad the volume before the FFT |

* The normalization needs min(radii) >= sigma. Otherwise the surface area normalization of Law
  and Chung (2008) is used, with a warning, as in Law's code.
* Masks are uint8. Otsu treats the response as bright on black for either background.
* Each side is padded by ceil((max(radii) + 3 sigma) / spacing) voxels, mirrored, so the FFT
  does not wrap around at the border. `--no-pad` turns this off.

```bash
binarize oof \
  --input path/to/volume.npy \
  --output path/to/response.npy \
  --radii 1 1.5 2 2.5 3

binarize oof \
  --input path/to/volume.npy \
  --output path/to/mask.npy \
  --radii 1 2 3 \
  --spacing 1 1 0.75 \
  --threshold 2
```

In Python, the options are the arguments of `Binarizer.oof`: `radii`, `sigma`, `spacing`,
`threshold` (None, `'otsu2d'`, `'otsu3d'` or a number) and `pad`.

References:

* M.W.K. Law and A.C.S. Chung, "Three Dimensional Curvilinear Structure Detection using
  Optimally Oriented Flux", ECCV 2008, pp. 368-382.
* M.W.K. Law et al., "Dilated Divergence based Scale-Space Representation for Curve Analysis",
  ECCV 2012, pp. 557-571.

## Python

```python
from microvasc.binarize import Binarizer

binarizer = Binarizer(volume, background="black")
mask = binarizer.otsu3d()
mask2d = binarizer.otsu2d()
vesselness = binarizer.frangi(alpha=0.5, beta=0.5, c=3, scales=[1, 2, 3])
mask = binarizer.bfrangi(tau=0.75, scales=[1, 2, 3], threshold="otsu3d")
response = binarizer.oof(radii=[1, 2, 3])
```

All arguments and return values are listed in the [API reference](api/binarize.md). The
parameters can be tuned with the scripts described in [Parameter optimization](optimization.md).

## External binarization methods

The paper also covers methods that are not implemented here:

* **U-Net**: [https://github.com/MrMras/CNN](https://github.com/MrMras/CNN)
* **nnU-Net**: [https://github.com/MIC-DKFZ/nnunet](https://github.com/MIC-DKFZ/nnunet)

### U-Net

Installation:

```bash
# Requirements
# Python 3.12+
# PyTorch, NumPy, OpenCV, tqdm, scipy, matplotlib
git clone https://github.com/MrMras/CNN.git
cd CNN

conda env create -n cnn -f environment.yml
conda activate cnn
```

Usage:

```bash
# Upload .npy files under   ./unprocessed_data/{name_of_dataset}
# Change the path to these files in ./create_volume_from_npy.py
python ./create_volume_from_npy.py
# Should create a folder ./data/{name_of_dataset} and add 2 files in there.

cd ./model/       # change the paths in ./model.py to ./data/{name_of_dataset}/{files}
python ./model.py EPOCHS -c
#  Model will save the model under ./saved_models/{name_of_dataset}/model_for_vasc3d{random id}.pth.

cd ../            # specify the path for the trained model and the path to the target volume (line 64)
python ./run_model.py
# Binary and probability maps are saved in ./processed_npy/ and ./probability_npy/
```

### nnU-Net

Installation:

```bash
conda create -n nnunet python=3.9.*
conda activate nnunet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install nnunetv2

# on Windows, use: set nnUNet_raw=path\to\nnUNet_raw (and the same for the other two)
export nnUNet_raw=path/to/nnUNet_raw
export nnUNet_preprocessed=path/to/nnUNet_preprocessed
export nnUNet_results=path/to/nnUNet_results
```

Usage:

```bash
# Use the new presets
nnUNetv2_plan_experiment -d DATASET_ID -pl nnUNetPlannerResEncL
# Extract fingerprint
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
# Train
nnUNetv2_train DATASET_ID 3d_fullres 0/1/2/3/4 -p nnUNetResEncUNetLPlans
# Predict
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -p nnUNetResEncUNetLPlans --save_probabilities
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```
