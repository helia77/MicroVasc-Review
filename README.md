# MicroVasc-Review

**Review of Segmentation and Skeletonization Methods for Large-Scale Microvascular Networks**

This repository contains code and resources associated with the paper  
"*Segmentation and Modeling of Large-Scale Microvascular Networks: A Survey*" by Goharbavang *et al*.

<p align="center">
  <img src="docs/overview.png" alt="Overview Diagram" width="100%" />
</p>

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

<!---
2. **Install dependencies** (e.g., via `requirements.txt` or a Conda environment)

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**

   ```bash
   cd manage_data
   python download_and_convert.py --modality LSFM --output ../data/lsfm
   ```

4. **Run an algorithm**

   ```bash
   cd implementations/frangi
   python run_frangi.py --input ../../data/lsfm --output ../../results/frangi
   ```

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

* **Optimally Oriented Flux (OOF)**: [https://matlab.mathworks.com/open/fileexchange/v1?id=41612](https://www.mathworks.com/matlabcentral/fileexchange/41612-optimally-oriented-flux-oof-for-3d-curvilinear-structure)

* **U-Net**: [https://github.com/milesial/Pytorch-UNet](https://github.com/MrMras/CNN)

* **nnU-Net**: [https://github.com/MIC-DKFZ/nnunet](https://github.com/MIC-DKFZ/nnunet)

* **Starlab MCF Skeletonization**: [https://github.com/taiya/starlab-mcfskel](https://github.com/taiya/starlab-mcfskel)

* **3D Slicer**: [https://www.slicer.org](https://www.slicer.org)

Refer to the paper for detailed methodology, evaluation results, and discussions.
