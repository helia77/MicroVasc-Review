"""
Regenerate the reference outputs used by test_kline.py. Needs scikit-fmm.

Only run this when the Kline behavior is meant to change:

    python tests/capture_kline_golden.py --force

Outputs are stored as packed bits to keep the file small.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

import synthetic
from implementations.Skeleton import Skeletonize

DATA = Path(__file__).parent / "data" / "kline_golden.npz"
SOURCE = Path(__file__).parents[1] / "implementations" / "Skeleton.py"

# mask and a root voxel on the vessel, as array indices along axes 0, 1, 2
CASES = {
    "tube": (synthetic.tube_mask, (3, 5, 2)),
    "bifurcation": (synthetic.bifurcation_mask, (2, 14, 14)),
    "torus": (synthetic.torus_mask, (7, 24, 16)),
}
# (dmw, mbtr): the defaults and the weights used for the KESM volume in the paper
WEIGHTS = {"defaults": (6, 10), "kesm": (19.25, 10.11)}


def case_key(name, weights):
    return f"kline__{name}__{weights}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite the existing reference file.")
    args = parser.parse_args()
    if DATA.exists() and not args.force:
        parser.error(f"{DATA} exists. Use --force only if a change in behavior is intended.")

    import skfmm

    arrays = {}
    for name, (make, root) in CASES.items():
        mask = make().astype(np.uint8)
        for weights, (dmw, mbtr) in WEIGHTS.items():
            skel = Skeletonize().kline(mask, root, dmw=dmw, mbtr=mbtr)
            arrays[case_key(name, weights)] = np.packbits(skel.ravel())
    arrays["meta__skeleton_sha256"] = np.array(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    arrays["meta__versions"] = np.array(f"numpy {np.__version__}, scikit-fmm {skfmm.__version__}")

    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, **arrays)
    print(f"Saved {len(arrays)} arrays to {DATA} ({DATA.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
