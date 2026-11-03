import os
import re
import argparse
import numpy as np


def main(args):
    path = args.input_dir
    files = os.listdir(path)
    step = args.step_size
    os.makedirs(args.output_dir, exist_ok=True)

    if args.parameter == 'alpha':
        # Fscore_<c block>_<alpha block>_.npy, each a (c, alpha) block of at most step x step
        blocks = {}
        for name in files:
            match = re.fullmatch(r'Fscore_(\d+)_(\d+)_\.npy', name)
            if match:
                blocks[int(match[1]), int(match[2])] = np.load(os.path.join(path, name))
        if not blocks:
            raise SystemExit(f"No Fscore_<i>_<j>_.npy files found in {path}")
        rows = max(step2 * step + b.shape[0] for (step2, _), b in blocks.items())
        cols = max(step3 * step + b.shape[1] for (_, step3), b in blocks.items())
        fscore = np.zeros((rows, cols))
        for (step2, step3), block in blocks.items():
            fscore[step2 * step:step2 * step + block.shape[0], step3 * step:step3 * step + block.shape[1]] = block
        np.save(os.path.join(args.output_dir, 'alpha_c_micro_fscores.npy'), fscore)

    elif args.parameter in ['beta', 'tau']:
        # <Fscore|Precision|Recall>_<array id>_.npy, each holding one value
        values = {}
        for name in files:
            match = re.fullmatch(r'(Fscore|Precision|Recall)_(\d+)_\.npy', name)
            if match:
                values[match[1], int(match[2])] = np.load(os.path.join(path, name))[0]
        if not values:
            raise SystemExit(f"No Fscore_<i>_.npy, Precision_<i>_.npy or Recall_<i>_.npy files found in {path}")
        N = max(num for _, num in values) + 1
        fscore = np.zeros((N,))
        precision = np.zeros((N,))
        recall = np.zeros((N,))
        for (typ, num), value in values.items():
            if typ == 'Fscore':
                fscore[num] = value
            elif typ == 'Precision':
                precision[num] = value
            elif typ == 'Recall':
                recall[num] = value
        np.save(os.path.join(args.output_dir, f'{args.parameter}_fscore.npy'), fscore)
        np.save(os.path.join(args.output_dir, f'{args.parameter}_precision.npy'), precision)
        np.save(os.path.join(args.output_dir, f'{args.parameter}_recall.npy'), recall)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Concatenate F-score results from saved files. Array sizes follow the files found; values of missing jobs stay 0.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the F-score files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the concatenated F-scores.')
    parser.add_argument('--parameter', type=str, required=True, choices=['alpha', 'beta', 'tau'], help='Parameter to process.')
    parser.add_argument('--step_size', type=int, default=4, help='Step size used in the parameter ranges.')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
