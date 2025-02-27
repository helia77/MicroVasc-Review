import os
import argparse
import warnings
import Binarize
import numpy as np
import geometry_metric as metric

# avoid saving warning files
warnings.filterwarnings('ignore', category=RuntimeWarning)


def return_frangi_fscore(vol, gr_truth, background, A, B, C, scale_range):
    # calculating the vesselness function
    volume = Binarize(vol, background)
    output = volume.frangi_filter(A, B, C, scale_range, threshold='otsu3d')
    met = metric(gr_truth, output)
    return met.f_score(), met.precision(), met.recall()

def return_bfrangi_fscore(vol, gr_truth, background, tau, scale_range):
    # calculating the vesselness function
    volume = Binarize(vol, background)
    output = volume.bfrangi_filter(tau, scale_range, threshold='otsu3d')
    met = metric(gr_truth, output)
    return met.f_score(), met.precision(), met.recall()

def main(args):
    sample_vol = np.load(args.input_volume)
    sample_gr = np.load(args.ground_truth)
    N = args.ranges
    
    if args.parameter == 'alpha':
        # parameter ranges
        alpha_range = np.linspace(0, 2, N)      # used 100 in the paper
        c_range = np.linspace(0, 50, N)         # used 100 in the paper
        b = 1                                   # doesn't matter in this scope
    
        # step sizes
        step = args.step_size
        arr_num = args.array_id
        step2 = arr_num // 25
        step3 = arr_num % 25
    
        # extract alpha and c values
        alpha = alpha_range[step3 * step:(step3 + 1) * step]
        c = c_range[step2 * step:(step2 + 1) * step]
    
        A, C = np.meshgrid(alpha, c)
        F_score = np.zeros(A.shape)
    
        # compute F_score for each combination of alpha and c
        for i in range(step):
            for j in range(step):
                F_score[i, j] = return_frangi_fscore(sample_vol, sample_gr, A[i, j], b, C[i, j])

        # save
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'Fscore_{step2}_{step3}_.npy'), F_score)
        
    elif args.parameter == 'beta':
        alpha = 0.005                                                               # change it based on the results from alpha/c optimization
        c = 22.73                                                                   # change it based on the results from alpha/c optimization
        beta_range = np.linspace(0, 1, N)                                           # 500 in the paper
        beta_range = np.concatenate((beta_range, np.linspace(1, 15, 201)[1:]))      # total of 500 elements

        fscore, pre, rec = return_frangi_fscore(sample_vol, sample_gr, alpha, beta_range[args.array_id], c)

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'Fscore_{args.array_id}_.npy'), [fscore])
        np.save(os.path.join(output_dir, f'Precision_{args.array_id}_.npy'), [pre])
        np.save(os.path.join(output_dir, f'Recall_{args.array_id}_.npy'), [rec])
    
    elif args.parameter == 'tau':
        tau_range = np.linspace(0, 1.0, N)                                          # used 100 in the paper

        fscore, pre, rec = return_bfrangi_fscore(sample_vol, sample_gr, tau_range[args.array_id])

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'Fscore_{args.array_id}_.npy'), [fscore])
        np.save(os.path.join(output_dir, f'Precision_{args.array_id}_.npy'), [pre])
        np.save(os.path.join(output_dir, f'Recall_{args.array_id}_.npy'), [rec])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute F-scores for a range of alpha and c values.")
    parser.add_argument('--input_volume', type=str, required=True, help='Path to the input volume file.')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to the ground truth file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files.')
    parser.add_argument('--parameter', type=str, required=True, choices=['alpha', 'beta', 'tau'], help='Parameter to process.')
    parser.add_argument('--step_size', type=int, default=4, help='Step size for parameter ranges.')
    parser.add_argument('--ranges', type=int, default=100, help='Number of each parameter to process.')
    parser.add_argument('--array_id', type=int, required=True, help='Array ID for parallel processing.')

    args = parser.parse_args()
    main(args)

