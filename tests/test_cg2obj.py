"""
cg2obj (microvasc/data.py): CG skeletons from starlab-mcfskel to OBJ.

The CG lines follow the writer curveskel_io_cg::save of starlab-mcfskel: a "# D:3 NV:n NE:m"
header, "v x y z" vertex lines with numbers from QString::arg (so 1e-05 can appear) and
"e i j" edge lines with 1-based indices.
"""
from pathlib import Path

import numpy as np
import pytest

from microvasc.data import NWT, cg2obj
from microvasc.metrics.network import as_segments

CG = ("# D:3 NV:4 NE:3\n"
      "v 0 0 0\n"
      "v 1e-05 1 0\n"
      "v 2.5e+03 1 -3.25e-07\n"
      "v 3 2 1\n"
      "e 1 2\n"
      "e 2 3\n"
      "e 2 4\n")
OBJ = CG.replace("\ne ", "\nl ")
VERTICES = [(0, 0, 0), (1e-05, 1, 0), (2.5e+03, 1, -3.25e-07), (3, 2, 1)]


def write(path, text=CG):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())
    return path


def test_small_file_end_to_end(tmp_path):
    cg = write(tmp_path / "skeleton.cg")
    cg2obj(str(cg))
    assert (tmp_path / "skeleton.obj").read_bytes() == OBJ.encode()
    assert cg.read_bytes() == CG.encode()


def test_load_obj_reads_the_result(tmp_path):
    cg2obj(write(tmp_path / "skeleton.cg"))
    network = NWT(tmp_path / "skeleton.obj")
    assert len(network.vertices) == 4 and len(network.edges) == 3
    expected = [(VERTICES[i - 1], VERTICES[j - 1]) for i, j in [(1, 2), (2, 3), (2, 4)]]
    np.testing.assert_array_equal(as_segments(network), expected)


@pytest.mark.parametrize("number", ["1e-05", "2.5e+03", "-3.25e-07", "1e+20"])
def test_numbers_with_an_exponent_are_kept(tmp_path, number):
    cg2obj(write(tmp_path / "a.cg", f"v {number} {number} 0\nv 1 2 3\ne 1 2\n"))
    assert (tmp_path / "a.obj").read_text() == f"v {number} {number} 0\nv 1 2 3\nl 1 2\n"


@pytest.mark.parametrize("name", ["skeleton", "skeleton.obj", "skeleton.cgx", "skeleton.cg.txt", ".cg"])
def test_name_without_cg_suffix_raises_and_keeps_the_input(tmp_path, name):
    path = write(tmp_path / name)
    with pytest.raises(ValueError, match="expected a .cg file"):
        cg2obj(path)
    assert path.read_bytes() == CG.encode()
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


def test_cg_elsewhere_in_the_path_does_not_matter(tmp_path):
    cg2obj(write(tmp_path / "x.cgdir" / "a.cg"))
    assert sorted(p.name for p in (tmp_path / "x.cgdir").iterdir()) == ["a.cg", "a.obj"]
    assert (tmp_path / "x.cgdir" / "a.obj").read_bytes() == OBJ.encode()


@pytest.mark.parametrize("as_type", [str, Path])
@pytest.mark.parametrize("suffix", [".cg", ".CG", ".Cg"])
def test_str_and_path_input_and_any_suffix_case(tmp_path, as_type, suffix):
    cg2obj(as_type(write(tmp_path / f"skeleton{suffix}")))
    assert (tmp_path / "skeleton.obj").read_bytes() == OBJ.encode()


def test_line_endings_are_kept(tmp_path):
    cg2obj(write(tmp_path / "a.cg", CG.replace("\n", "\r\n")))
    assert (tmp_path / "a.obj").read_bytes() == OBJ.replace("\n", "\r\n").encode()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cg2obj(tmp_path / "missing.cg")
