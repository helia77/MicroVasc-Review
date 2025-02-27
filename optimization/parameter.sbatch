#!/bin/bash
#SBATCH -J Name
#SBATCH --array=0-624                               # in case of N=100 and step_size=4 ((100/4)*(100/4))
#SBATCH -N 1                                        # Total # of nodes
#SBATCH -n 1                                        # Total # of precessors
#SBATCH -t 00:40:00                                 # Run time (hh:mm:ss)
#SBATCH --mem-per-cpu=2GB

module load python/3.9

# command-line arguments
PARAMETER=$1            # Parameter to process (alpha/c, beta, or tau)
INPUT_VOLUME=$2         # Path to input volume
GROUND_TRUTH=$3         # Path to ground truth
OUTPUT_DIR=$4           # Directory to save output files
STEP_SIZE=$5            # Step size for parameter ranges
RANGES=$6               # Number of parameter values to process

# array range based on parameter
if [ "$PARAMETER" == "alpha" ]; then
    # Default for alpha/c requires 625 jobs (100 of each with step size 4 => 25x25 grid)
    ARRAY_RANGE="0-624"
elif [ "$PARAMETER" == "beta" ]; then
    # Default for beta parameter requires 500 jobs
    ARRAY_RANGE="0-499"
elif [ "$PARAMETER" == "tau" ]; then
    # Default for tau requires 100 jobs
    ARRAY_RANGE="0-99"
else
    echo "Invalid parameter. Choose 'alpha', 'beta', or 'tau'."
    exit 1
fi

# update the job array range
#SBATCH --array=$ARRAY_RANGE



python parameter_fscore.py \
    --input_volume $INPUT_VOLUME \
    --ground_truth $GROUND_TRUTH \
    --output_dir $OUTPUT_DIR \
    --parameter $PARAMETER \
    --step_size $STEP_SIZE \
    --ranges $RANGES \
    --array_id $SLURM_ARRAY_TASK_ID