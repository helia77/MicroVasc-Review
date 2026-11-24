"""Smoke tests: each command runs from the command line and writes its output."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import synthetic
from microvasc.binarize import Binarizer
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
    assert result.stdout.startswith("usage: skeleton ")
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


def binarize_oof(tmp_path, volume, *args):
    np.save(tmp_path / "in.npy", volume)
    return run("-m", "microvasc.binarize", "oof",
               "--input", str(tmp_path / "in.npy"), "--output", str(tmp_path / "out.npy"), *args)


@pytest.mark.parametrize("args, kwargs", [
    (["--radii", "1", "2", "3"], {"radii": [1.0, 2.0, 3.0]}),
    (["--radii", "1.5", "3", "--sigma", "0.75", "--spacing", "1", "1", "0.5", "--no-pad", "--background", "white"],
     {"radii": [1.5, 3.0], "sigma": 0.75, "spacing": [1.0, 1.0, 0.5], "pad": False, "background": "white"}),
    (["--radii", "2", "--spacing", "0.5"], {"radii": [2.0], "spacing": 0.5}),
], ids=["defaults", "all-options", "one-spacing"])
def test_binarize_oof_response(tmp_path, args, kwargs):
    background = kwargs.pop("background", "black")
    volume = synthetic.tube() if background == "black" else 255 - synthetic.tube()
    result = binarize_oof(tmp_path, volume, *args)
    assert result.returncode == 0, result.stderr
    out = np.load(tmp_path / "out.npy")
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, Binarizer(volume, background).oof(**kwargs))


@pytest.mark.parametrize("text, threshold", [("otsu2d", "otsu2d"), ("otsu3d", "otsu3d"), ("2", 2.0), ("-1.5", -1.5)])
def test_binarize_oof_threshold(tmp_path, text, threshold):
    volume = synthetic.tube()
    result = binarize_oof(tmp_path, volume, "--radii", "1", "2", "3", "--threshold", text)
    assert result.returncode == 0, result.stderr
    out = np.load(tmp_path / "out.npy")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, Binarizer(volume).oof([1, 2, 3], threshold=threshold))


@pytest.mark.parametrize("args, message", [
    (["--radii", "1", "--threshold", "otsu"], "--threshold: must be otsu2d, otsu3d or a finite number"),
    (["--radii", "1", "--threshold", "nan"], "--threshold: must be otsu2d, otsu3d or a finite number"),
    (["--radii", "1", "--threshold", "inf"], "--threshold: must be otsu2d, otsu3d or a finite number"),
    (["--radii", "1", "--sigma", "x"], "--sigma: invalid float value"),
    (["--threshold", "otsu3d"], "the following arguments are required: --radii"),
    (["--radii"], "--radii: expected at least one argument"),
    (["--radii", "-1", "2"], "radii"),
    (["--radii", "1", "--sigma", "0"], "sigma"),
    (["--radii", "1", "--spacing", "1", "1"], "spacing must be one number or 3 values"),
])
def test_binarize_oof_bad_input(tmp_path, args, message):
    result = binarize_oof(tmp_path, synthetic.tube(), *args)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "out.npy").exists()


def test_binarize_oof_2d_volume(tmp_path):
    result = binarize_oof(tmp_path, np.zeros((8, 8)), "--radii", "1")
    assert result.returncode != 0
    assert "3D" in result.stderr
    assert not (tmp_path / "out.npy").exists()
