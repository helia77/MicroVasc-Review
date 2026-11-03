# Parameter optimization

These scripts sweep the Frangi parameters (alpha and c, then beta) and the Beyond Frangi
parameter tau on a volume with a ground truth segmentation. Each job runs the filter, thresholds
the result with 3D Otsu and scores it against the ground truth. The sweeps run as Slurm job
arrays, one job per parameter value (or per block of the alpha/c grid).

## Installation

From the repository root, in the Python environment used on the cluster:

```bash
pip install -e .
```

`parameter.sbatch` loads `python/3.9` with `module load`. Change that line, the job name and the
resources (`-t`, `--mem-per-cpu`) to fit your cluster. Run all commands below from the
repository root.

## Sweeps

| Sweep | Values (N = `--ranges`, default 100) | Fixed | Jobs |
|---|---|---|---|
| `alpha` | alpha: `linspace(0, 2, N)`, c: `linspace(0, 50, N)` | beta = 1 | `ceil(N / step)^2` (625 for N = 100, `--step_size 4`) |
| `beta` | N values in [0, 1], then 200 values in (1, 15] | alpha, c (`--alpha`, `--c`) | N + 200 (300 for N = 100, 500 for `--ranges 300`) |
| `tau` | `linspace(0, 1, N)` | | N |

The value lists are defined in `optimization/parameter_fscore.py` (`alpha_c_ranges`,
`beta_range`, `tau_range`).

Submit each sweep with `submit.sh`. It asks `parameter_fscore` for the number of jobs and passes
`--array=0-<jobs - 1>` to `sbatch`, so the array always matches the sweep. Options after the
four paths go to `parameter_fscore`. `--scale` (the filter scales) is required. Set `SCALES` to the
scales chosen for the data set, and add `--background white` for dark vessels on a bright
background.

```bash
SCALES="..."   # filter scales for the data set, for example "1 2 3"

# 1. alpha and c
optimization/submit.sh alpha data/volume.npy data/ground_truth.npy results/alpha \
    --scale $SCALES --ranges 100 --step_size 4

# 2. beta, with the best alpha and c from step 1 (defaults: --alpha 0.005 --c 22.73)
optimization/submit.sh beta data/volume.npy data/ground_truth.npy results/beta \
    --scale $SCALES --ranges 100 --alpha <alpha> --c <c>

# 3. tau (Beyond Frangi)
optimization/submit.sh tau data/volume.npy data/ground_truth.npy results/tau \
    --scale $SCALES --ranges 100
```

To see the number of jobs without submitting:

```bash
python -m optimization.parameter_fscore --parameter beta --ranges 100 --jobs
```

Without Slurm, run the jobs in a loop, with the same options:

```bash
JOBS=$(python -m optimization.parameter_fscore --parameter tau --ranges 100 --jobs)
for i in $(seq 0 $((JOBS - 1))); do
    python -m optimization.parameter_fscore --parameter tau --input_volume data/volume.npy \
        --ground_truth data/ground_truth.npy --output_dir results/tau --array_id $i \
        --scale $SCALES --ranges 100
done
```

Each alpha/c job writes `Fscore_<c block>_<alpha block>_.npy`, a block of F-scores (rows c,
columns alpha). Each beta or tau job writes `Fscore_<id>_.npy`, `Precision_<id>_.npy` and
`Recall_<id>_.npy`.

## Concatenation

```bash
python -m optimization.parameter_concat --input_dir results/alpha --output_dir results --parameter alpha --step_size 4
python -m optimization.parameter_concat --input_dir results/beta --output_dir results --parameter beta
python -m optimization.parameter_concat --input_dir results/tau --output_dir results --parameter tau
```

This writes `alpha_c_micro_fscores.npy` (rows c, columns alpha), `beta_fscore.npy`,
`beta_precision.npy`, `beta_recall.npy` and the same three files for tau. The array sizes follow
the files found. Values of jobs that did not finish stay 0, so check the Slurm logs first.

## Picking the best values

Use the same N (`--ranges`) as in the sweep:

```python
import numpy as np
from optimization.parameter_fscore import alpha_c_ranges, beta_range, tau_range

N = 100
F = np.load("results/alpha_c_micro_fscores.npy")
i, j = np.unravel_index(np.argmax(F), F.shape)
alpha_range, c_range = alpha_c_ranges(N)
alpha, c = alpha_range[j], c_range[i]

beta = beta_range(N)[np.argmax(np.load("results/beta_fscore.npy"))]
tau = tau_range(N)[np.argmax(np.load("results/tau_fscore.npy"))]
```

## F-score and Jaccard index

The scripts record the F-score, which equals the Dice coefficient. The paper reports the
Jaccard index J. The two are related by

    J = F / (2 - F)

J increases with F, so the parameters with the highest F-score also have the highest Jaccard
index. To plot Jaccard, convert the stored values with `F / (2 - F)`.
