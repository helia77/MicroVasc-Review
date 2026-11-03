#!/bin/bash
# Submit a parameter sweep as a Slurm job array, one job per value (or alpha/c block).
#
# Usage: microvasc/optimization/submit.sh PARAMETER INPUT_VOLUME GROUND_TRUTH OUTPUT_DIR [options]
#
# PARAMETER is alpha, beta or tau. Options are passed to microvasc.optimization.parameter_fscore
# (--scale is required). The array size is computed by parameter_fscore from the same
# options, so it always matches the length of the sweep.
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 PARAMETER INPUT_VOLUME GROUND_TRUTH OUTPUT_DIR [options]" >&2
    exit 1
fi

PARAMETER=$1
INPUT_VOLUME=$2
GROUND_TRUTH=$3
OUTPUT_DIR=$4
shift 4

JOBS=$(python -m microvasc.optimization.parameter_fscore --parameter "$PARAMETER" --jobs "$@")
echo "Submitting $JOBS jobs for $PARAMETER"
sbatch --array=0-$((JOBS - 1)) "$(dirname "$0")/parameter.sbatch" \
    "$PARAMETER" "$INPUT_VOLUME" "$GROUND_TRUTH" "$OUTPUT_DIR" "$@"
