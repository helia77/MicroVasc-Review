import os
import argparse
import numpy as np


def main(args):
    path = args.input_dir
    N = args.ranges
    files = os.listdir(path)
    step = args.step_size

    if args.parameter == 'alpha':
        fscore = np.zeros((N, N))
        for name in files:
            value = np.load(os.path.join(path, name))[0]
            step2 = int(name.split('_')[1])
            step3 = int(name.split('_')[2])
            fscore[step2 * step:(step2 + 1) * step, step3 * step:(step3 + 1) * step] = np.load(os.path.join(path, name))
        np.save(os.path.join(args.output_dir, 'alpha_c_micro_fscores.npy'), fscore)
    
    elif args.parameter in ['beta', 'tau']:
        fscore = np.zeros((N,))
        precision = np.zeros((N,))
        recall = np.zeros((N,))
        for name in files:
            value = np.load(os.path.join(path, name))[0]
            typ = name.split('_')[0]
            num = int(name.split('_')[1])
            if typ == 'Fscore':
                fscore[num] = value
            elif typ == 'Precision':
                precision[num] = value
            elif typ == 'Recall':
                recall[num] = value
        np.save(os.path.join(args.output_dir, f'{args.parameter}_fscore.npy'), fscore)
        np.save(os.path.join(args.output_dir, f'{args.parameter}_precision.npy'), precision)
        np.save(os.path.join(args.output_dir, f'{args.parameter}_recall.npy'), recall)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate F-score results from saved files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the F-score files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the concatenated F-scores.')
    parser.add_argument('--parameter', type=str, required=True, choices=['alpha', 'beta', 'tau'], help='Parameter to process.')
    parser.add_argument('--step_size', type=int, default=4, help='Step size used in the parameter ranges.')
    parser.add_argument('--ranges', type=int, default=100, help='Number of each parameter used in the process.')

    args = parser.parse_args()
    main(args)