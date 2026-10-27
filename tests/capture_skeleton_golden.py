"""
Regenerate the reference outputs used by test_palagyi.py.

Only run this when the thinning behavior is meant to change:

    python tests/capture_skeleton_golden.py --force

Outputs are stored as packed bits to keep the file small.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

import synthetic
from implementations.Skeleton import Skeletonize

DATA = Path(__file__).parent / "data" / "skeleton_golden.npz"
SOURCE = Path(__file__).parents[1] / "implementations" / "Skeleton.py"

MASKS = {
    "tube": synthetic.tube_mask,
    "bifurcation": synthetic.bifurcation_mask,
    "torus": synthetic.torus_mask,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite the existing reference file.")
    args = parser.parse_args()
    if DATA.exists() and not args.force:
        parser.error(f"{DATA} exists. Use --force only if a change in behavior is intended.")

    arrays = {}
    for name, make in MASKS.items():
        skel = Skeletonize().palagyi(make().astype(np.uint8))
        arrays[f"palagyi__{name}"] = np.packbits(skel.ravel())
    arrays["meta__skeleton_sha256"] = np.array(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    arrays["meta__numpy"] = np.array(np.__version__)

    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, **arrays)
    print(f"Saved {len(arrays)} arrays to {DATA} ({DATA.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
