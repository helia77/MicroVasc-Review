"""
Regenerate the reference outputs used by test_skeleton_metric.py.

Only run this when the NetMets or network reader behavior is meant to change:

    python tests/capture_netmets_golden.py --force

Stores Lee skeletons of the synthetic tube and bifurcation as inputs (packed bits), so the
NetMets values do not depend on the thinning code, and the outputs of the NWT/OBJ reader for
the synthetic network in synthetic.py.
"""
import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np

import synthetic
from implementations.Skeleton import Skeletonize
from manage_data.manage_data import NWT
from metrics.skeleton_metric import netmets, voxel_skeleton_to_segments

DATA = Path(__file__).parent / "data" / "netmets_golden.npz"
SOURCES = [Path(__file__).parents[1] / "metrics" / "skeleton_metric.py",
           Path(__file__).parents[1] / "manage_data" / "manage_data.py"]

SKELETONS = {
    "tube": (synthetic.tube_mask, synthetic.TUBE_AXIS),
    "bifurcation": (synthetic.bifurcation_mask, synthetic.BIFURCATION_AXIS),
}
VOXEL_SIZES = {"iso": (1.0, 1.0, 1.0), "aniso": (0.75, 1.0, 1.0)}
SIGMA = 1.0


def case_key(name, spacing):
    return f"netmets__{name}__{spacing}"


def netmets_case(skeleton, axis, voxel_size):
    """(P, R) of a voxel skeleton against its analytic centerline, both scaled to um."""
    test = voxel_skeleton_to_segments(skeleton, voxel_size)
    truth = np.asarray(axis, dtype=float) * voxel_size
    return np.array(netmets(test, truth, SIGMA))


def reader_outputs():
    """What NWT reads from the synthetic network written as OBJ and as NWT."""
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for kind, write in (("obj", synthetic.write_obj), ("nwt", synthetic.write_nwt)):
            path = Path(d) / f"network.{kind}"
            write(path)
            n = NWT(str(path))
            out[f"{kind}__vertices"] = np.array([v.p for v in n.v], dtype=float)
            out[f"{kind}__edge_v"] = np.array([e.v for e in n.e], dtype=int)
            out[f"{kind}__eout"] = np.array([str([int(i) for i in v.Eout]) for v in n.v])
            out[f"{kind}__ein"] = np.array([str([int(i) for i in v.Ein]) for v in n.v])
            out[f"{kind}__segments"] = np.array([s.p for s in n.linesegments()], dtype=float)
        out["obj__pointcloud"] = np.array(NWT(str(Path(d) / "network.obj")).pointcloud(0.5), dtype=float)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite the existing reference file.")
    args = parser.parse_args()
    if DATA.exists() and not args.force:
        parser.error(f"{DATA} exists. Use --force only if a change in behavior is intended.")

    arrays = reader_outputs()
    for name, (make, axis) in SKELETONS.items():
        mask = make()
        skeleton = Skeletonize().lee(mask.astype(np.uint8))
        arrays[f"skeleton__{name}"] = np.packbits(skeleton.ravel())
        arrays[f"shape__{name}"] = np.array(mask.shape)
        for spacing, voxel_size in VOXEL_SIZES.items():
            arrays[case_key(name, spacing)] = netmets_case(skeleton, axis, voxel_size)
    for source in SOURCES:
        arrays[f"meta__{source.stem}_sha256"] = np.array(hashlib.sha256(source.read_bytes()).hexdigest())
    arrays["meta__numpy"] = np.array(np.__version__)

    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, **arrays)
    print(f"Saved {len(arrays)} arrays to {DATA} ({DATA.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
