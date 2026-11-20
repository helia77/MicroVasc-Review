"""
NWT.load_obj (microvasc/data.py) with spaces and tabs between fields, and the file extension
in any letter case.
"""
from pathlib import Path

import numpy as np
import pytest

import synthetic
from microvasc.data import NWT


def assert_same(a, b):
    """Same vertex positions, edge ends, edge points and line segments."""
    np.testing.assert_array_equal([v.p for v in a.vertices], [v.p for v in b.vertices])
    assert [e.v for e in a.edges] == [e.v for e in b.edges]
    for ea, eb in zip(a.edges, b.edges):
        np.testing.assert_array_equal(np.array(ea.p, float), np.array(eb.p, float))
    np.testing.assert_array_equal([s.p for s in a.line_segments()], [s.p for s in b.line_segments()])


@pytest.fixture
def reference(tmp_path):
    synthetic.write_obj(tmp_path / "network.obj")
    return tmp_path / "network.obj", NWT(tmp_path / "network.obj")


@pytest.mark.parametrize("change", [
    lambda line: line.replace(" ", "  "),               # two spaces
    lambda line: line.replace(" ", "\t"),               # tab
    lambda line: line.replace(" ", " \t "),             # spaces and a tab
    lambda line: line.replace("\n", " \n"),             # trailing space
    lambda line: line.replace("\n", "\r\n"),            # CRLF
], ids=["two spaces", "tab", "spaces and tab", "trailing space", "crlf"])
def test_fields_split_on_any_whitespace(tmp_path, reference, change):
    path, expected = reference
    text = "".join(change(line) for line in path.read_text().splitlines(keepends=True))
    (tmp_path / "changed.obj").write_bytes(text.encode())
    assert_same(NWT(tmp_path / "changed.obj"), expected)


def test_empty_and_other_lines_are_skipped(tmp_path, reference):
    path, expected = reference
    text = "# network\n\n   \no network\n" + path.read_text().replace("\nl ", "\n\nvn 0 0 1\nl ", 1)
    (tmp_path / "changed.obj").write_text(text)
    assert_same(NWT(tmp_path / "changed.obj"), expected)


@pytest.mark.parametrize("as_type", [str, Path])
@pytest.mark.parametrize("suffix", [".OBJ", ".Obj"])
def test_obj_extension_in_any_case(tmp_path, reference, as_type, suffix):
    path, expected = reference
    (tmp_path / f"upper{suffix}").write_bytes(path.read_bytes())
    assert_same(NWT(as_type(tmp_path / f"upper{suffix}")), expected)


def test_nwt_extension_in_any_case(tmp_path):
    synthetic.write_nwt(tmp_path / "network.nwt")
    synthetic.write_nwt(tmp_path / "upper.NWT")
    assert_same(NWT(tmp_path / "upper.NWT"), NWT(tmp_path / "network.nwt"))


@pytest.mark.parametrize("name", ["network.txt", "network.objx", "network.obj.bak", "network"])
def test_other_extensions_raise(tmp_path, reference, name):
    (tmp_path / name).write_bytes(reference[0].read_bytes())
    with pytest.raises(ValueError, match="Expected '.nwt' or '.obj'"):
        NWT(tmp_path / name)
