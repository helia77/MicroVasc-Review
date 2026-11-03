import os
import math
import argparse
import warnings
import numpy as np
from implementations.Binarize import Binarizer
from metrics.geometry_metric import metric


def alpha_c_ranges(N):
    alpha_range = np.linspace(0, 2, N)      # used 100 in the paper  TODO(Helia): confirm
    c_range = np.linspace(0, 50, N)         # used 100 in the paper  TODO(Helia): confirm
    return alpha_range, c_range

def beta_range(N):
    beta_range = np.linspace(0, 1, N)                                           # 500 in the paper  TODO(Helia): confirm
    beta_range = np.concatenate((beta_range, np.linspace(1, 15, 201)[1:]))      # total of 500 elements  TODO(Helia): confirm
    return beta_range

def tau_range(N):
    return np.linspace(0, 1.0, N)                                               # used 100 in the paper  TODO(Helia): confirm

def job_count(parameter, N, step):
    """
    Number of Slurm array jobs for a sweep: one per (c block, alpha block) of size step x step
    for alpha/c, one per value for beta and tau.
    """
    if parameter == 'alpha':
        alpha, c = alpha_c_ranges(N)
        return math.ceil(len(c) / step) * math.ceil(len(alpha) / step)
    if parameter == 'beta':
        return len(beta_range(N))
    return len(tau_range(N))

def return_frangi_fscore(vol, gr_truth, background, A, B, C, scale_range):
    # calculating the vesselness function
    volume = Binarizer(vol, background)
    output = volume.frangi(alpha=A, beta=B, c=C, scale_range=scale_range, threshold='otsu3d')
    met = metric(gr_truth, output)
    return met.f_score(), met.precision(), met.recall()

def return_bfrangi_fscore(vol, gr_truth, background, tau, scale_range):
    # calculating the vesselness function
    volume = Binarizer(vol, background)
    output = volume.bfrangi(tau=tau, scale_range=scale_range, threshold='otsu3d')
    met = metric(gr_truth, output)
    return met.f_score(), met.precision(), met.recall()

def main(args):
    sample_vol = np.load(args.input_volume)
    sample_gr = np.load(args.ground_truth)
    N = args.ranges
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.parameter == 'alpha':
        # parameter ranges
        alpha_range, c_range = alpha_c_ranges(N)
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
                F_score[i, j] = return_frangi_fscore(sample_vol, sample_gr, args.background,
                                                     A[i, j], b, C[i, j], args.scale)[0]

        # save
        np.save(os.path.join(output_dir, f'Fscore_{step2}_{step3}_.npy'), F_score)

    elif args.parameter == 'beta':
        # alpha and c come from the alpha/c optimization (--alpha, --c)
        fscore, pre, rec = return_frangi_fscore(sample_vol, sample_gr, args.background, args.alpha,
                                                beta_range(N)[args.array_id], args.c, args.scale)

        np.save(os.path.join(output_dir, f'Fscore_{args.array_id}_.npy'), [fscore])
        np.save(os.path.join(output_dir, f'Precision_{args.array_id}_.npy'), [pre])
        np.save(os.path.join(output_dir, f'Recall_{args.array_id}_.npy'), [rec])

    elif args.parameter == 'tau':
        fscore, pre, rec = return_bfrangi_fscore(sample_vol, sample_gr, args.background,
                                                 tau_range(N)[args.array_id], args.scale)

        np.save(os.path.join(output_dir, f'Fscore_{args.array_id}_.npy'), [fscore])
        np.save(os.path.join(output_dir, f'Precision_{args.array_id}_.npy'), [pre])
        np.save(os.path.join(output_dir, f'Recall_{args.array_id}_.npy'), [rec])

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute F-scores for one job of a Frangi (alpha/c, beta) or Beyond Frangi (tau) parameter sweep.")
    parser.add_argument('--parameter', type=str, required=True, choices=['alpha', 'beta', 'tau'], help='Parameter to process.')
    parser.add_argument('--step_size', type=int, default=4, help='Block size of the alpha/c grid handled by one job.')
    parser.add_argument('--ranges', type=int, default=100, help='Number of each parameter to process.')
    parser.add_argument('--jobs', action='store_true', help='Print the number of array jobs for this sweep and exit.')
    parser.add_argument('--input_volume', type=str, help='Path to the input volume file.')
    parser.add_argument('--ground_truth', type=str, help='Path to the ground truth file.')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output files.')
    parser.add_argument('--array_id', type=int, help='Array ID for parallel processing.')
    parser.add_argument('--scale', nargs='+', type=float, help='Scales for the Frangi and Beyond Frangi filters.')
    parser.add_argument('--background', choices=['black', 'white'], default='black', help='Background of the input volume.')
    parser.add_argument('--alpha', type=float, default=0.005, help='Frangi alpha for the beta sweep (from the alpha/c optimization).')
    parser.add_argument('--c', type=float, default=22.73, help='Frangi c for the beta sweep (from the alpha/c optimization).')

    args = parser.parse_args(argv)
    if args.jobs:
        return args
    missing = [f'--{k}' for k in ('input_volume', 'ground_truth', 'output_dir', 'array_id', 'scale')
               if getattr(args, k) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    jobs = job_count(args.parameter, args.ranges, args.step_size)
    if not 0 <= args.array_id < jobs:
        parser.error(f"--array_id must be in 0-{jobs - 1} for this sweep")
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.jobs:
        print(job_count(args.parameter, args.ranges, args.step_size))
    else:
        # avoid saving warning files
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        main(args)
