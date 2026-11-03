# Binarization

<p align="center">
  <img src="segmentation.gif" alt="Segmentation" width="100%" />
</p>

`binarize` implements four of the binarization methods from the paper: Otsu's thresholding in
2D and 3D, Frangi vesselness and Beyond Frangi. All subcommands read a 3D `.npy` volume and
write a `.npy` file.

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

Frangi vesselness. `--params alpha beta c` sets the three Frangi parameters and `--scale` the
Gaussian scales (in voxels). The response is the maximum over the scales, in [0, 1]. With
`--th otsu2d` or `--th otsu3d` the response is thresholded with Otsu and the output is a uint8 mask.

```bash
binarize frangi \
  --input path/to/volume.npy \
  --output path/to/out.npy \
  --background white \
  --params 0.5 0.5 3 \
  --scale 1 2 3 4 \
  --th otsu3d
```

The paper's baseline Frangi settings use c equal to half of the maximum Hessian norm. This code
takes c as an explicit value and does not compute it.

## Beyond Frangi: `bfrangi`

Beyond Frangi (Jerman et al.). `--params tau` sets the regularization parameter. `--scale` and
`--th` work as for `frangi`.

```bash
binarize bfrangi \
  --input path/to/volume.npy \
  --output path/to/out.npy \
  --params 0.75 \
  --scale 1 2 3 \
  --th otsu3d
```

The regularization uses tau times the largest |lambda3| at each scale, taken before the sign is
flipped for the background. Jerman's reference implementation
([JermanEnhancementFilter](https://github.com/timjerman/JermanEnhancementFilter)) uses the signed
maximum after the flip, so results can differ between the two.

## Python

```python
from microvasc.binarize import Binarizer

binarizer = Binarizer(volume, background="black")
mask = binarizer.otsu3d()
mask2d = binarizer.otsu2d()
vesselness = binarizer.frangi(alpha=0.5, beta=0.5, c=3, scale_range=[1, 2, 3])
mask = binarizer.bfrangi(tau=0.75, scale_range=[1, 2, 3], threshold="otsu3d")
```

The parameters can be tuned with the scripts described in [Parameter optimization](optimization.md).

## External binarization methods

The paper also covers methods that are not implemented here:

* **Optimally Oriented Flux**: [https://www.mathworks.com/matlabcentral/fileexchange/41612-optimally-oriented-flux-oof-for-3d-curvilinear-structure](https://www.mathworks.com/matlabcentral/fileexchange/41612-optimally-oriented-flux-oof-for-3d-curvilinear-structure)
* **U-Net**: [https://github.com/MrMras/CNN](https://github.com/MrMras/CNN)
* **nnU-Net**: [https://github.com/MIC-DKFZ/nnunet](https://github.com/MIC-DKFZ/nnunet)

### U-Net

* Installation:

  ```bash
  # Requirements
  # Python 3.12+
  # PyTorch, NumPy, OpenCV, tqdm, scipy, matplotlib
  git clone https://github.com/MrMras/CNN.git
  cd CNN

  conda env create -n cnn -f environment.yml
  conda activate cnn
  ```
* Usage:

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

* Installation:

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
* Usage:

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
