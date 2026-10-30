# MicroVasc-Review

**Review of Segmentation and Skeletonization Methods for Large-Scale Microvascular Networks**

This repository contains code and resources associated with the paper  
"*Segmentation and Modeling of Large-Scale Microvascular Networks: A Survey*" by Goharbavang *et al*.

📄 **[Read the Full Paper](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1645520/full)**

<p align="center">
  <img src="docs/overview.png" alt="Overview Diagram" width="100%" />
</p>

## Citation
If you use this code or the paper, please cite:

```bibtex
@article{goharbavang2025segmentation,
  title={Segmentation and modeling of large-scale microvascular networks: a survey},
  author={Goharbavang, Helya and Ashitkov, Artem T and Pillai, Athira and Wythe, Joshua D and Chen, Guoning and Mayerich, David},
  journal={Frontiers in Bioinformatics},
  volume={5},
  pages={1645520},
  year={2025},
  publisher={Frontiers Media SA}
}
```

## Repository Content
The paper covers the following methods:

* **Binarization**: Otsu’s thresholding; Frangi vesselness; Beyond Frangi; Optimally Oriented Flux; U‑Net; nnU‑Net
* **Skeletonization**: Lee thinning; Palágyi thinning; Surface normal accumulation (Kerautret); Fast Marching (Kline); Mean curvature flow (Tagliasacchi); Voronoi‑based (Antiga)

This repository includes two CLI modules under 'implementations/' for a subset of these methods. For the rest, please see the External Resources section below for links to community or official implementations.

 - `Binarize.py`  
  Provides `otsu2d`, `otsu3d`, `frangi`, and `bfrangi`.

- `Skeleton.py`  
  Provides `lee`, `palagyi`, `kerautret`, and `kline` subcommands.

Other directories:
* `manage_data/` Scripts for preprocessing and converting raw imaging data (e.g., Numpy → NWT).

* `metrics/` Tools to compute quantitative evaluation metrics:

  * **Segmentation**: Jaccard index, Dice coefficient, precision, recall
  * **Skeletonization**: NetMets precision and recall

* `optimization/` Parameter-tuning and sensitivity-analysis scripts (grid search for scale, shape, and threshold parameters).

* `LICENSE` MIT License.

## Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/helia77/MicroVasc-Review.git
   cd MicroVasc-Review
   ```

2. **Install dependencies** (e.g., via Conda environment)

   ```bash
   # conda create --name microvasc
   # conda activate microvasc
   pip install -e .
   ```

3. **Verify CLI tools**

   ```bash
   binarize --help
   skeleton --help
   ```

4. **Run an algorithm**

   ```bash
   binarize frangi \
   --input path/to/volume.npy \
   --output path/to/out.npy \
   --background white \
   --params 0.5 0.5 3 \
   --scale 1 2 3 4
   --th otsu3d
   ```

   Kline centerline extraction (needs `pip install -e ".[kline]"`), with the root given as array indices along axes 0, 1 and 2. These are the settings used for the KESM volume in the paper:

   ```bash
   skeleton kline \
   --input path/to/binary.npy \
   --output path/to/centerline.npy \
   --root 95 76 35 \
   --dmw 19.25 \
   --mbtr 10.11
   ```
<!---
5. **Evaluate results**

   ```bash
   cd metrics
   python evaluate.py --pred ../results/frangi --gt ../data/lsfm/gt
   ```

6. **Optimize parameters**

   ```bash
   cd optimization
   python optimize_frangi.py --modality KESM --scales 1 2 3 4
   ```
--->
## External Resources & Implementations
* **Beyond Frangi**: [https://github.com/timjerman/JermanEnhancementFilter](https://github.com/timjerman/JermanEnhancementFilter)
* **Optimally Oriented Flux**: [https://matlab.mathworks.com/open/fileexchange/v1?id=41612](https://www.mathworks.com/matlabcentral/fileexchange/41612-optimally-oriented-flux-oof-for-3d-curvilinear-structure)
* **U-Net**: [https://github.com/MrMras/CNN](https://github.com/MrMras/CNN)
* **nnU-Net**: [https://github.com/MIC-DKFZ/nnunet](https://github.com/MIC-DKFZ/nnunet)
* **Lee Thinning**: [https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/_skeletonize.py](https://github.com/scikit-image)
* **Palagyi Thinning**: K. Palagyi and A. Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm, Graphical Models and Image Processing 61, 199-221, 1999. [https://doi.org/10.1006/gmip.1999.0498](https://doi.org/10.1006/gmip.1999.0498)
* **Kerautret Centerline**: [https://github.com/kerautret/CDCVAM](https://github.com/kerautret/CDCVAM)
* **Kline Centerline**: [https://github.com/TLKline/poreture](https://github.com/TLKline/poreture)
* **Starlab MCF Skeletonization**: [https://github.com/taiya/starlab-mcfskel](https://github.com/taiya/starlab-mcfskel)
* **3D Slicer**: [https://www.slicer.org](https://www.slicer.org)

Refer to the paper for detailed methodology, evaluation results, and discussions.

## Running External Codes

### 1. **U-Net**
   
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
      # Binary and probability maps will be generated in ./processed_npy/ and ./probability_npy/
    ```
    
### 2. **nnU-Net**
   
  * Installation:
    
    ```bash
    conda create -n nnunet python=3.9.*
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install nnunetv2
    
    set nnUNet_raw = path/to/nnUNet_raw
    set nnUNet_preprocessed = path/to/nnUNet_preprocessed
    set nnUNet_results = path/to/nnUNet_results
    ```
  * Usage:
    
    ```bash
    # Use the new presets
    nnUNetv2_plan_experiment -d DATASET_ID -pl nnUNetPlannerResEncL
    # Extract fingerprint
    nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
    # Train
    nnUNetv2_train DATASET__ID 3d_fullres 0/1/2/3/4 -p nnUNetResEncUNetLPlans
    # Predict
    nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -p nnUNetResEncUNetLPlans --save_probabilities
    nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
    ```

### 3. **Kerautret (external)**

  `skeleton kerautret` runs the `centerLineGeodesicGraph` program from [CDCVAM](https://github.com/kerautret/CDCVAM), which must be built separately. The input is a surface mesh of the binarized volume in OFF format.

  * Installation:

    ```bash
    # Requirements: CMake, a C++ compiler, Boost, and DGtal (https://github.com/DGtal-team/DGtal)
    git clone https://github.com/kerautret/CDCVAM.git
    cd CDCVAM
    mkdir build && cd build
    cmake ..
    make
    # The program is build/bin/centerLineGeodesicGraph
    ```
  * Usage:

    The input meshes in the paper were `binary_<name>.off`, made from the binarized volumes. This example uses the KESM settings from the table below:

    ```bash
    # Make the OFF mesh from a binarized volume (marching cubes)
    python -c "from manage_data.manage_data import npy2off; npy2off('path/to/binary_kesm.npy', 'path/to/binary_kesm.off')"

    skeleton kerautret \
      --input path/to/binary_kesm.off \
      --output path/to/out/tree \
      --exe-path path/to/CDCVAM/build/bin/centerLineGeodesicGraph \
      --dilate-dist 1.0 --delta-g 3.3 --radius 1.1 --threshold 0.5
    # Writes treeVertex.sdp, treeEdges.sdp, treeRadius.sdp and tree.obj in path/to/out/
    ```
    `--radius` is the accumulation distance. The paper sets it slightly larger than the radius of the largest vessels.

    Settings used in the paper:

    | Dataset  | `--dilate-dist` | `--delta-g` | `--radius` | `--threshold` |
    |----------|-----------------|-------------|------------|---------------|
    | KESM     | 1.0             | 3.3         | 1.1        | 0.5           |
    | LSFM     | 1.0             | 3.1         | 0.7        | 0.5           |
    | micro-CT | 1.1             | 2.75        | 0.8        | 0.6           |

    Values are as used in the paper, in the coordinate units of the input mesh.
