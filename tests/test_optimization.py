"""
Tests for the parameter sweep scripts in microvasc/optimization/.

These scripts could not run before (wrong imports and filter calls), so there are no earlier
outputs to compare with; results are compared with direct Binarizer and SegmentationMetrics calls instead.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import synthetic
from microvasc.binarize import Binarizer
from microvasc.metrics.geometry import SegmentationMetrics
from microvasc.optimization import parameter_concat, parameter_fscore as pf

ROOT = Path(__file__).parents[1]
SCALE = [1.0]
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
# the sweeps start at alpha, beta, c = 0, where Frangi divides by the smallest float and the
# division overflows to -inf; exp(-inf) = 0 is the intended result, so the overflow is not reported
sweep_overflow = np.errstate(over="ignore")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("inputs")
    np.save(d / "vol.npy", synthetic.tube())
    np.save(d / "gt.npy", synthetic.tube_mask().astype(np.uint8))
    return d / "vol.npy", d / "gt.npy"


def run_job(inputs, out, parameter, array_id, *extra):
    vol, gt = inputs
    pf.main(pf.parse_args(["--parameter", parameter, "--input-volume", str(vol), "--ground-truth", str(gt),
                           "--output-dir", str(out), "--array-id", str(array_id),
                           "--scale", *map(str, SCALE), *extra]))


def concat(src, dst, parameter, *extra):
    parameter_concat.main(parameter_concat.parse_args(
        ["--input-dir", str(src), "--output-dir", str(dst), "--parameter", parameter, *extra]))


def frangi_scores(alpha, beta, c):
    out = Binarizer(synthetic.tube()).frangi(alpha=alpha, beta=beta, c=c, scales=SCALE, threshold="otsu3d")
    m = SegmentationMetrics(synthetic.tube_mask(), out)
    return m.f_score(), m.precision(), m.recall()


@pytest.mark.parametrize("parameter,N,step,jobs", [
    ("alpha", 100, 4, 625), ("alpha", 10, 4, 9), ("beta", 100, 4, 300), ("beta", 300, 4, 500),
    ("tau", 100, 4, 100),
])
def test_job_count_follows_sweep_length(parameter, N, step, jobs):
    assert pf.job_count(parameter, N, step) == jobs


@sweep_overflow
def test_alpha_c_grid_with_uneven_blocks(inputs, tmp_path):
    # N = 3 with blocks of 2: four jobs with blocks of 2x2, 2x1, 1x2 and 1x1
    N, step = 3, 2
    jobs = pf.job_count("alpha", N, step)
    assert jobs == 4
    for i in range(jobs):
        run_job(inputs, tmp_path / "jobs", "alpha", i, "--num-values", str(N), "--step-size", str(step))
    concat(tmp_path / "jobs", tmp_path, "alpha", "--step-size", str(step))
    grid = np.load(tmp_path / "alpha_c_micro_fscores.npy")
    assert grid.shape == (N, N)
    alpha_range, c_range = pf.alpha_c_ranges(N)
    for i, c in enumerate(c_range):          # rows are c, columns alpha
        for j, alpha in enumerate(alpha_range):
            assert grid[i, j] == frangi_scores(alpha, 1, c)[0]


@sweep_overflow
def test_beta_sweep(inputs, tmp_path):
    N = 2
    last = pf.job_count("beta", N, 4) - 1
    assert last == N + 199
    for i in (0, 1, last):
        run_job(inputs, tmp_path / "jobs", "beta", i, "--num-values", str(N), "--alpha", "0.5", "--c", "25")
    concat(tmp_path / "jobs", tmp_path, "beta")
    for name, k in (("fscore", 0), ("precision", 1), ("recall", 2)):
        values = np.load(tmp_path / f"beta_{name}.npy")
        assert values.shape == (last + 1,)    # sized from the files found
        for i in (0, 1, last):
            assert values[i] == frangi_scores(0.5, pf.beta_range(N)[i], 25)[k]
    assert pf.beta_range(N)[last] == 15


def test_tau_sweep(inputs, tmp_path):
    N = 2
    for i in range(N):
        run_job(inputs, tmp_path / "jobs", "tau", i, "--num-values", str(N), "--background", "black")
    concat(tmp_path / "jobs", tmp_path, "tau")
    fscore = np.load(tmp_path / "tau_fscore.npy")
    assert fscore.shape == (N,)
    for i, tau in enumerate(pf.tau_range(N)):
        out = Binarizer(synthetic.tube()).bfrangi(tau=tau, scales=SCALE, threshold="otsu3d")
        assert fscore[i] == SegmentationMetrics(synthetic.tube_mask(), out).f_score()


def test_bad_arguments(inputs, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_job(inputs, tmp_path, "tau", 2, "--num-values", "2")      # only jobs 0 and 1 exist
    with pytest.raises(SystemExit):
        pf.parse_args(["--parameter", "tau", "--array-id", "0"])     # missing inputs and --scale
    assert "--input-volume, --ground-truth, --output-dir, --scale" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        concat(tmp_path, tmp_path, "tau")                             # no result files


# shell scripts, run with fake sbatch, module and python commands

def fake_command(bin_dir, name, log):
    path = bin_dir / name
    path.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{log}"\n')
    path.chmod(0o755)


@needs_bash
def test_scripts_parse():
    for script in ("parameter.sbatch", "submit.sh"):
        result = subprocess.run(["bash", "-n", str(ROOT / "microvasc" / "optimization" / script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@needs_bash
@pytest.mark.parametrize("parameter,ranges,array", [("beta", "100", "0-299"), ("beta", "300", "0-499"),
                                                    ("alpha", "100", "0-624"), ("tau", "100", "0-99")])
def test_submit_sets_array_from_sweep_length(tmp_path, parameter, ranges, array):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    fake_command(bin_dir, "sbatch", tmp_path / "sbatch.log")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(["bash", "microvasc/optimization/submit.sh", parameter, "vol.npy", "gt.npy", "out",
                             "--scale", "1", "2", "--num-values", ranges],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "sbatch.log").read_text().splitlines()
    assert argv == [f"--array={array}", "microvasc/optimization/parameter.sbatch", parameter, "vol.npy", "gt.npy", "out",
                    "--scale", "1", "2", "--num-values", ranges]


@needs_bash
def test_sbatch_job_passes_arguments(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_command(bin_dir, "python", tmp_path / "python.log")
    fake_command(bin_dir, "module", tmp_path / "module.log")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "SLURM_ARRAY_TASK_ID": "7"}
    args = ["bash", "microvasc/optimization/parameter.sbatch", "tau", "vol.npy", "gt.npy", "out", "--scale", "1", "2"]
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "python.log").read_text().splitlines() == [
        "-m", "microvasc.optimization.parameter_fscore", "--parameter", "tau", "--input-volume", "vol.npy",
        "--ground-truth", "gt.npy", "--output-dir", "out", "--array-id", "7", "--scale", "1", "2"]

    del env["SLURM_ARRAY_TASK_ID"]
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "submit.sh" in result.stderr
