import hashlib

from build_pk12_table import build_table
from implementations.Skeleton import Skeletonize

# SHA-256 of the unpacked lookup table (2^26 bool entries) in implementations/PK12.npz
PK12_DATA_SHA256 = "5eeaee537c85ca95643cbf61c5e28e5023ff59f72ac46128453d82c1f80fc1a8"


def test_bundled_table_data_is_unchanged():
    table = Skeletonize()._load_table(Skeletonize.DEFAULT_TEMPLATES)
    assert hashlib.sha256(table.tobytes()).hexdigest() == PK12_DATA_SHA256


def test_templates_rebuild_bundled_table():
    bundled = Skeletonize()._load_table(Skeletonize.DEFAULT_TEMPLATES)
    table = build_table()
    assert table.dtype == bundled.dtype and table.shape == bundled.shape
    assert (table != bundled).sum() == 0
