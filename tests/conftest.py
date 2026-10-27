import json
import stat
import sys

import pytest

# Stands in for CDCVAM's centerLineGeodesicGraph: records its arguments, notes which
# output files already existed, and writes a three-vertex graph to <prefix>Vertex.sdp
# and <prefix>Edges.sdp.
FAKE_CDCVAM = """\
import json, os, sys
args = sys.argv[1:]
prefix = args[args.index("-o") + 1]
existing = [s for s in ("Vertex.sdp", "Edges.sdp", "Radius.sdp") if os.path.exists(prefix + s)]
with open(os.environ["FAKE_CDCVAM_LOG"], "w") as f:
    json.dump({"argv": sys.argv, "existing": existing}, f)
with open(prefix + "Vertex.sdp", "w") as f:
    f.write("1 2 3\\n4 5 6\\n7 8 9\\n")
with open(prefix + "Edges.sdp", "w") as f:
    f.write("0 1\\n1 2\\n")
"""


@pytest.fixture
def fake_cdcvam(tmp_path, monkeypatch):
    """Path to a fake centerLineGeodesicGraph and a function that reads its log."""
    exe = tmp_path / "bin" / "centerLineGeodesicGraph"
    exe.parent.mkdir()
    exe.write_text(f"#!{sys.executable}\n{FAKE_CDCVAM}")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "fake_cdcvam.json"
    monkeypatch.setenv("FAKE_CDCVAM_LOG", str(log))

    def read_log():
        return json.loads(log.read_text()) if log.exists() else None

    return exe, read_log
