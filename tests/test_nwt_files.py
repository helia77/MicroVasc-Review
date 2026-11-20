"""
NWT.load_nwt and NWT.load_obj (microvasc/data.py) close the file they read, after a successful
read and after an error in the file.
"""
import builtins
import gc
import struct
import warnings

import pytest

import synthetic
from microvasc import data as md
from microvasc.data import NWT

WRITERS = {"nwt": synthetic.write_nwt, "obj": synthetic.write_obj}


@pytest.fixture
def opened(monkeypatch):
    """Files opened by microvasc.data, through an `open` that records them."""
    files = []

    def recording_open(*args, **kwargs):
        f = builtins.open(*args, **kwargs)
        files.append(f)
        return f

    monkeypatch.setattr(md, "open", recording_open, raising=False)
    return files


def broken_file(kind, path):
    if kind == "nwt":
        synthetic.write_nwt(path)
        path.write_bytes(path.read_bytes()[:76])  # header and vertex count, no edge count
        return struct.error
    path.write_text("v 0 0 0\nv 1 x 0\nl 1 2\n")
    return ValueError


@pytest.mark.parametrize("kind", ["nwt", "obj"])
def test_file_is_closed_after_reading(tmp_path, opened, kind):
    WRITERS[kind](tmp_path / f"network.{kind}")
    network = NWT(tmp_path / f"network.{kind}")
    assert len(network.edges) == len(synthetic.NETWORK_EDGES)
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize("kind", ["nwt", "obj"])
def test_file_is_closed_after_an_error(tmp_path, opened, kind):
    error = broken_file(kind, tmp_path / f"network.{kind}")
    with pytest.raises(error):
        NWT(tmp_path / f"network.{kind}")
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize("kind", ["nwt", "obj"])
@pytest.mark.parametrize("broken", [False, True], ids=["valid", "broken"])
def test_no_resource_warning(tmp_path, kind, broken):
    path = tmp_path / f"network.{kind}"
    error = broken_file(kind, path) if broken else WRITERS[kind](path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        try:
            NWT(path)
        except Exception as e:
            assert broken and isinstance(e, error)
        gc.collect()
    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []
