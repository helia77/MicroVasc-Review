"""Smoke tests: each command runs from the command line and writes its output."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import synthetic
from implementations.Skeleton import Skeletonize

ROOT = Path(__file__).parents[1]

# imports only the core dependencies; optional packages are blocked
NO_OPTIONAL = ("import sys\n"
               "for m in ('skfmm', 'cv2', 'nrrd', 'SimpleITK'):\n"
               "    sys.modules[m] = None\n")


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_skeleton_help_without_optional_packages():
    result = run("-c", NO_OPTIONAL + "sys.argv = ['skeleton', '--help']\n"
                 "from implementations.Skeleton import main\nmain()")
    assert result.returncode == 0, result.stderr
    for cmd in ("lee", "palagyi", "kerautret", "kline"):
        assert cmd in result.stdout


def test_skeleton_lee(tmp_path):
    mask = synthetic.tube_mask().astype(np.uint8)
    np.save(tmp_path / "in.npy", mask)
    result = run("-m", "implementations.Skeleton", "lee",
                 "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy"))
    assert result.returncode == 0, result.stderr
    out = np.load(tmp_path / "out.npy")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, Skeletonize().lee(mask))


@pytest.mark.parametrize("templates", [False, True], ids=["bundled", "explicit"])
def test_skeleton_palagyi(tmp_path, templates):
    mask = synthetic.tube_mask().astype(np.uint8)
    np.save(tmp_path / "in.npy", mask)
    extra = ["--templates", str(Skeletonize.DEFAULT_TEMPLATES)] if templates else []
    result = run("-m", "implementations.Skeleton", "palagyi",
                 "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy"), *extra)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), Skeletonize().palagyi(mask))
