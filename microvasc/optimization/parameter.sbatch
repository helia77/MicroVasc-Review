#!/bin/bash
#SBATCH -J Name
#SBATCH -N 1                                        # Total # of nodes
#SBATCH -n 1                                        # Total # of precessors
#SBATCH -t 00:40:00                                 # Run time (hh:mm:ss)
#SBATCH --mem-per-cpu=2GB

# One job of a parameter sweep. Submit with microvasc/optimization/submit.sh, which sets --array
# from the length of the sweep. See docs/optimization.md.

module load python/3.9

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Not running as a Slurm array job. Submit with microvasc/optimization/submit.sh." >&2
    exit 1
fi

# command-line arguments
PARAMETER=$1            # Parameter to process (alpha, beta, or tau)
INPUT_VOLUME=$2         # Path to input volume
GROUND_TRUTH=$3         # Path to ground truth
OUTPUT_DIR=$4           # Directory to save output files
shift 4                 # the rest (--scale, --num-values, --step-size, ...) goes to parameter_fscore

python -m microvasc.optimization.parameter_fscore \
    --parameter "$PARAMETER" \
    --input-volume "$INPUT_VOLUME" \
    --ground-truth "$GROUND_TRUTH" \
    --output-dir "$OUTPUT_DIR" \
    --array-id "$SLURM_ARRAY_TASK_ID" \
    "$@"
