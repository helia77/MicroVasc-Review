"""
Regenerate the reference outputs used by test_golden_binarize.py.

Only run this when the filter behavior is meant to change:

    python tests/capture_golden.py --force

The parameters are test settings for small synthetic volumes, not the values used in the paper.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import scipy
import skimage

import synthetic
from implementations.Binarize import Binarizer

DATA = Path(__file__).parent / "data" / "binarize_golden.npz"
SOURCE = Path(__file__).parents[1] / "implementations" / "Binarize.py"

SCALES = [1.0, 2.0, 3.0]
FRANGI = dict(alpha=0.5, beta=0.5, c=25.0)
TAU = 0.5

BACKGROUNDS = ("black", "white")
# (method, threshold) pairs; otsu methods take no threshold argument
METHODS = [
    ("otsu2d", None), ("otsu3d", None),
    ("frangi", None), ("frangi", "otsu2d"), ("frangi", "otsu3d"),
    ("bfrangi", None), ("bfrangi", "otsu2d"), ("bfrangi", "otsu3d"),
]


def case_key(name, background, method, threshold):
    return f"{name}__{background}__{method}__{threshold or 'none'}"


def prepare(volume, background):
    """The white-background case is the inverted volume (dark vessels on bright tissue)."""
    return volume if background == "black" else 255 - volume


def run_case(volume, background, method, threshold):
    """Run one Binarizer call on a fresh object, so no state carries over between calls."""
    binzr = Binarizer(prepare(volume, background), background=background)
    if method in ("otsu2d", "otsu3d"):
        return getattr(binzr, method)()
    if method == "frangi":
        return binzr.frangi(scale_range=SCALES, threshold=threshold, **FRANGI)
    if method == "bfrangi":
        return binzr.bfrangi(tau=TAU, scale_range=SCALES, threshold=threshold)
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite the existing reference file.")
    args = parser.parse_args()
    if DATA.exists() and not args.force:
        parser.error(f"{DATA} exists. Use --force only if a change in behavior is intended.")

    arrays = {}
    for name, make in synthetic.VOLUMES.items():
        volume = make()
        arrays[f"input__{name}"] = volume
        for background in BACKGROUNDS:
            for method, threshold in METHODS:
                arrays[case_key(name, background, method, threshold)] = run_case(volume, background, method, threshold)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=SOURCE.parent).stdout.strip()
    arrays["meta__binarize_sha256"] = np.array(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    arrays["meta__commit"] = np.array(commit)
    arrays["meta__versions"] = np.array(
        f"numpy {np.__version__}, scipy {scipy.__version__}, scikit-image {skimage.__version__}")

    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, **arrays)
    print(f"Saved {len(arrays)} arrays to {DATA} ({DATA.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
