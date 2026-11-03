"""Smoke tests: each command runs from the command line and writes its output."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import synthetic
from microvasc.skeleton import Skeletonizer

ROOT = Path(__file__).parents[1]

# imports only the core dependencies; optional packages are blocked
NO_OPTIONAL = ("import sys\n"
               "for m in ('skfmm', 'cv2', 'nrrd', 'SimpleITK'):\n"
               "    sys.modules[m] = None\n")


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_skeleton_help_without_optional_packages():
    result = run("-c", NO_OPTIONAL + "sys.argv = ['skeleton', '--help']\n"
                 "from microvasc.skeleton import main\nmain()")
    assert result.returncode == 0, result.stderr
    for cmd in ("lee", "palagyi", "kerautret", "kline"):
        assert cmd in result.stdout


def test_skeleton_lee(tmp_path):
    mask = synthetic.tube_mask().astype(np.uint8)
    np.save(tmp_path / "in.npy", mask)
    result = run("-m", "microvasc.skeleton", "lee",
                 "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy"))
    assert result.returncode == 0, result.stderr
    out = np.load(tmp_path / "out.npy")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, Skeletonizer().lee(mask))


@pytest.mark.parametrize("templates", [False, True], ids=["bundled", "explicit"])
def test_skeleton_palagyi(tmp_path, templates):
    mask = synthetic.tube_mask().astype(np.uint8)
    np.save(tmp_path / "in.npy", mask)
    extra = ["--templates", str(Skeletonizer.DEFAULT_TEMPLATES)] if templates else []
    result = run("-m", "microvasc.skeleton", "palagyi",
                 "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy"), *extra)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), Skeletonizer().palagyi(mask))


def test_skeleton_kline(tmp_path):
    pytest.importorskip("skfmm")
    mask = synthetic.tube_mask().astype(np.uint8)
    np.save(tmp_path / "in.npy", mask)
    result = run("-m", "microvasc.skeleton", "kline",
                 "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy"),
                 "--root", "3", "5", "2", "--dist-map-weight", "6", "--min-branch-to-root", "10")
    assert result.returncode == 0, result.stderr
    out = np.load(tmp_path / "out.npy")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, Skeletonizer().kline(mask, (3, 5, 2), dist_map_weight=6, min_branch_to_root=10))


def test_skeleton_kline_bad_root(tmp_path):
    np.save(tmp_path / "in.npy", synthetic.tube_mask().astype(np.uint8))
    result = run("-m", "microvasc.skeleton", "kline",
                 "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy"),
                 "--root", "0", "23", "0")
    assert result.returncode != 0
    assert "not on a vessel voxel" in result.stderr
    assert not (tmp_path / "out.npy").exists()


def test_skeleton_kerautret(tmp_path, fake_cdcvam):
    exe, read_log = fake_cdcvam
    (tmp_path / "mesh.off").write_text("OFF\n0 0 0\n")
    result = run("-m", "microvasc.skeleton", "kerautret",
                 "--input", str(tmp_path / "mesh.off"), "--output", str(tmp_path / "tree"),
                 "--exe-path", str(exe), "--dilate-dist", "1.5", "--delta-g", "2",
                 "--radius", "12", "--threshold", "0.25")
    assert result.returncode == 0, result.stderr
    assert read_log()["argv"] == [str(exe), "-i", str(tmp_path / "mesh.off"), "-o", str(tmp_path / "tree"),
                                  "--dilateDist", "1.5", "-g", "2.0", "-R", "12.0", "-t", "0.25"]
    assert (tmp_path / "tree.obj").exists()
