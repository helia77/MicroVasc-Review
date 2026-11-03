import os
import math
import argparse
import warnings
import numpy as np
from microvasc.binarize import Binarizer
from microvasc.metrics.geometry import SegmentationMetrics


def alpha_c_ranges(num_values):
    alpha_range = np.linspace(0, 2, num_values)     # used 100 in the paper  TODO(Helia): confirm
    c_range = np.linspace(0, 50, num_values)        # used 100 in the paper  TODO(Helia): confirm
    return alpha_range, c_range

def beta_range(num_values):
    beta_range = np.linspace(0, 1, num_values)                                  # 500 in the paper  TODO(Helia): confirm
    beta_range = np.concatenate((beta_range, np.linspace(1, 15, 201)[1:]))      # total of 500 elements  TODO(Helia): confirm
    return beta_range

def tau_range(num_values):
    return np.linspace(0, 1.0, num_values)                                      # used 100 in the paper  TODO(Helia): confirm

def job_count(parameter, num_values, step):
    """
    Number of Slurm array jobs for a sweep: one per (c block, alpha block) of size step x step
    for alpha/c, one per value for beta and tau.
    """
    if parameter == 'alpha':
        alpha, c = alpha_c_ranges(num_values)
        return math.ceil(len(c) / step) * math.ceil(len(alpha) / step)
    if parameter == 'beta':
        return len(beta_range(num_values))
    return len(tau_range(num_values))

def frangi_fscore(volume, ground_truth, background, alpha, beta, c, scales):
    # calculating the vesselness function
    binarizer = Binarizer(volume, background)
    output = binarizer.frangi(alpha=alpha, beta=beta, c=c, scales=scales, threshold='otsu3d')
    met = SegmentationMetrics(ground_truth, output)
    return met.f_score(), met.precision(), met.recall()

def bfrangi_fscore(volume, ground_truth, background, tau, scales):
    # calculating the vesselness function
    binarizer = Binarizer(volume, background)
    output = binarizer.bfrangi(tau=tau, scales=scales, threshold='otsu3d')
    met = SegmentationMetrics(ground_truth, output)
    return met.f_score(), met.precision(), met.recall()

def main(args):
    sample_vol = np.load(args.input_volume)
    sample_gr = np.load(args.ground_truth)
    num_values = args.num_values
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.parameter == 'alpha':
        # parameter ranges
        alpha_range, c_range = alpha_c_ranges(num_values)
        b = 1                                   # doesn't matter in this scope

        # step sizes
        step = args.step_size
        blocks = math.ceil(len(alpha_range) / step)
        step2 = args.array_id // blocks
        step3 = args.array_id % blocks

        # extract alpha and c values
        alpha = alpha_range[step3 * step:(step3 + 1) * step]
        c = c_range[step2 * step:(step2 + 1) * step]

        A, C = np.meshgrid(alpha, c)
        F_score = np.zeros(A.shape)

        # compute F_score for each combination of alpha and c
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                F_score[i, j] = frangi_fscore(sample_vol, sample_gr, args.background,
                                              A[i, j], b, C[i, j], args.scale)[0]

        # save
        np.save(os.path.join(output_dir, f'Fscore_{step2}_{step3}_.npy'), F_score)

    elif args.parameter == 'beta':
        # alpha and c come from the alpha/c optimization (--alpha, --c)
        fscore, pre, rec = frangi_fscore(sample_vol, sample_gr, args.background, args.alpha,
                                         beta_range(num_values)[args.array_id], args.c, args.scale)

        np.save(os.path.join(output_dir, f'Fscore_{args.array_id}_.npy'), [fscore])
        np.save(os.path.join(output_dir, f'Precision_{args.array_id}_.npy'), [pre])
        np.save(os.path.join(output_dir, f'Recall_{args.array_id}_.npy'), [rec])

    elif args.parameter == 'tau':
        fscore, pre, rec = bfrangi_fscore(sample_vol, sample_gr, args.background,
                                          tau_range(num_values)[args.array_id], args.scale)

        np.save(os.path.join(output_dir, f'Fscore_{args.array_id}_.npy'), [fscore])
        np.save(os.path.join(output_dir, f'Precision_{args.array_id}_.npy'), [pre])
        np.save(os.path.join(output_dir, f'Recall_{args.array_id}_.npy'), [rec])

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute F-scores for one job of a Frangi (alpha/c, beta) or Beyond Frangi (tau) parameter sweep.")
    parser.add_argument('--parameter', type=str, required=True, choices=['alpha', 'beta', 'tau'], help='Parameter to process.')
    parser.add_argument('--step-size', type=int, default=4, help='Block size of the alpha/c grid handled by one job.')
    parser.add_argument('--num-values', type=int, default=100, help='Number of each parameter to process.')
    parser.add_argument('--jobs', action='store_true', help='Print the number of array jobs for this sweep and exit.')
    parser.add_argument('--input-volume', type=str, help='Path to the input volume file.')
    parser.add_argument('--ground-truth', type=str, help='Path to the ground truth file.')
    parser.add_argument('--output-dir', type=str, help='Directory to save the output files.')
    parser.add_argument('--array-id', type=int, help='Array ID for parallel processing.')
    parser.add_argument('--scale', nargs='+', type=float, help='Scales for the Frangi and Beyond Frangi filters.')
    parser.add_argument('--background', choices=['black', 'white'], default='black', help='Background of the input volume.')
    parser.add_argument('--alpha', type=float, default=0.005, help='Frangi alpha for the beta sweep (from the alpha/c optimization).')
    parser.add_argument('--c', type=float, default=22.73, help='Frangi c for the beta sweep (from the alpha/c optimization).')

    args = parser.parse_args(argv)
    if args.jobs:
        return args
    missing = ['--' + k.replace('_', '-') for k in ('input_volume', 'ground_truth', 'output_dir', 'array_id', 'scale')
               if getattr(args, k) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    jobs = job_count(args.parameter, args.num_values, args.step_size)
    if not 0 <= args.array_id < jobs:
        parser.error(f"--array-id must be in 0-{jobs - 1} for this sweep")
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.jobs:
        print(job_count(args.parameter, args.num_values, args.step_size))
    else:
        # avoid saving warning files
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        main(args)
