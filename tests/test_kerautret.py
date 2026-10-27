from pathlib import Path

import pytest

from implementations.Skeleton import Skeletonize
from manage_data import manage_data as md

EXPECTED_OBJ = """\
# vertices
v 1 2 3
v 4 5 6
v 7 8 9

# edges
l 1 2
l 2 3
"""


@pytest.fixture
def mesh(tmp_path):
    off = tmp_path / "mesh.off"
    off.write_text("OFF\n0 0 0\n")
    return off


def test_command_line(tmp_path, mesh, fake_cdcvam):
    exe, read_log = fake_cdcvam
    prefix = tmp_path / "out" / "tree"
    prefix.parent.mkdir()
    Skeletonize().kerautret(exe, mesh, prefix, dilateDist=1.5, deltaG=2.0, radius=12.0, threshold=0.25)
    assert read_log()["argv"] == [str(exe), "-i", str(mesh), "-o", str(prefix),
                                  "--dilateDist", "1.5", "-g", "2.0", "-R", "12.0", "-t", "0.25"]


def test_default_parameters(tmp_path, mesh, fake_cdcvam):
    exe, read_log = fake_cdcvam
    Skeletonize().kerautret(exe, mesh, tmp_path / "tree")
    assert read_log()["argv"][5:] == ["--dilateDist", "2.0", "-g", "3.0", "-R", "10.0", "-t", "0.5"]


def test_writes_obj(tmp_path, mesh, fake_cdcvam):
    exe, _ = fake_cdcvam
    obj = Skeletonize().kerautret(exe, mesh, tmp_path / "tree")
    assert obj == tmp_path / "tree.obj"
    assert obj.read_text() == EXPECTED_OBJ


def test_removes_only_old_files_of_this_prefix(tmp_path, mesh, fake_cdcvam):
    exe, read_log = fake_cdcvam
    prefix = tmp_path / "tree"
    own = [tmp_path / f"tree{s}" for s in ("Vertex.sdp", "Edges.sdp", "Radius.sdp")]
    others = [tmp_path / "otherVertex.sdp", tmp_path / "tree2Vertex.sdp",
              tmp_path / "treeVertex.sdp.bak", tmp_path / "tree.obj.keep"]
    for f in own + others:
        f.write_text("stale\n")

    Skeletonize().kerautret(exe, mesh, prefix)

    assert read_log()["existing"] == []
    # the fake writes Vertex and Edges again, but not Radius
    assert not own[2].exists()
    assert all(f.read_text() == "stale\n" for f in others)


def test_missing_executable(tmp_path, mesh, fake_cdcvam):
    _, read_log = fake_cdcvam
    with pytest.raises(FileNotFoundError, match="github.com/kerautret/CDCVAM"):
        Skeletonize().kerautret(tmp_path / "missing", mesh, tmp_path / "tree")
    assert read_log() is None


def test_executable_without_execute_permission(tmp_path, mesh, fake_cdcvam):
    exe, read_log = fake_cdcvam
    exe.chmod(0o644)
    with pytest.raises(FileNotFoundError, match="not executable"):
        Skeletonize().kerautret(exe, mesh, tmp_path / "tree")
    assert read_log() is None


def test_missing_mesh(tmp_path, fake_cdcvam):
    exe, read_log = fake_cdcvam
    with pytest.raises(FileNotFoundError, match="mesh"):
        Skeletonize().kerautret(exe, tmp_path / "missing.off", tmp_path / "tree")
    assert read_log() is None


@pytest.mark.parametrize("as_path", [False, True], ids=["str", "Path"])
def test_sdp2obj(tmp_path, as_path):
    prefix = tmp_path / "graph"
    Path(f"{prefix}Vertex.sdp").write_text("1 2 3\n4 5 6\n7 8 9\n\n")
    Path(f"{prefix}Edges.sdp").write_text("0 1\n1 2\n")
    obj = md.sdp2obj(prefix if as_path else str(prefix))
    assert obj == f"{prefix}.obj"
    assert Path(obj).read_text() == EXPECTED_OBJ


def test_sdp2obj_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        md.sdp2obj(tmp_path / "nothing")
