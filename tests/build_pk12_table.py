"""
Build the Palagyi lookup table from the templates T1 to T14 and compare it with the
table that Skeletonize.palagyi loads by default.

Entry i is True when the 3x3x3 neighborhood of a black point matches at least one
template of T_US, where bit k of i is the value of the neighbor at BIT_OFFSETS[k].
All 2^26 neighborhoods are checked.

    python tests/build_pk12_table.py                  # compare only
    python tests/build_pk12_table.py --output implementations/PK12.npz

The table is saved as packed bits (np.packbits) under "pk12" in a compressed .npz file.

Exits with status 1 if any entry differs from the bundled table.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from palagyi_templates import BIT_OFFSETS, TEMPLATES, offset

N_BITS = len(BIT_OFFSETS)
BIT = {index: BIT_OFFSETS.index(offset(index)) for index in range(2, 28)}


def _mask(positions):
    return np.uint32(sum(1 << BIT[p] for p in positions))


def matches(template, idx):
    """Boolean array: which neighborhoods in idx (uint32 bit patterns) the template matches."""
    black, white = _mask(template["black"]), _mask(template["white"])
    ok = ((idx & black) == black) & ((idx & white) == 0)
    for group in template["groups"]:
        if group[0] == "x":
            ok &= (idx & _mask(group[1])) != 0
        elif group[0] == "v":
            m = _mask(group[1])
            ok &= (idx & m) != m
        elif group[0] == "z":
            a, b = np.uint32(BIT[group[1]]), np.uint32(BIT[group[2]])
            ok &= (((idx >> a) ^ (idx >> b)) & np.uint32(1)) == 1
        else:
            raise ValueError(f"Unknown template group {group[0]!r}")
    return ok


def build_table(chunk_bits=22):
    """Lookup table of length 2^26, built in chunks of 2^chunk_bits neighborhoods."""
    table = np.zeros(1 << N_BITS, dtype=bool)
    step = 1 << chunk_bits
    for start in range(0, table.size, step):
        idx = np.arange(start, start + step, dtype=np.uint32)
        for template in TEMPLATES.values():
            table[start:start + step] |= matches(template, idx)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="Save the built table to this .npz file.")
    args = parser.parse_args()

    from implementations.Skeleton import Skeletonize

    table = build_table()
    print(f"built:   {table.size} entries, {int(table.sum())} deletable, "
          f"data sha256 {hashlib.sha256(table.tobytes()).hexdigest()}")
    if args.output is not None:
        np.savez_compressed(args.output, pk12=np.packbits(table))
        print(f"saved to {args.output}")

    bundled = Skeletonize()._load_table(Skeletonize.DEFAULT_TEMPLATES)
    print(f"bundled: {bundled.size} entries, {int(bundled.sum())} deletable, "
          f"data sha256 {hashlib.sha256(bundled.tobytes()).hexdigest()} ({Skeletonize.DEFAULT_TEMPLATES})")
    if bundled.dtype != table.dtype or bundled.shape != table.shape:
        print(f"bundled table has dtype {bundled.dtype} and shape {bundled.shape}, "
              f"expected {table.dtype} and {table.shape}")
        sys.exit(1)
    differ = int(np.count_nonzero(table != bundled))
    print(f"differing entries: {differ}")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
