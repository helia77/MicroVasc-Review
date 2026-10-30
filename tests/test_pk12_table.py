import hashlib

import numpy as np

from build_pk12_table import build_table
from implementations.Skeleton import Skeletonize

# SHA-256 of the bundled implementations/PK12.npy file
PK12_SHA256 = "279a101101729ba3480fb67ecd15a0d6d3aaf000248440a681d7c9fad278054a"


def test_bundled_table_file_is_unchanged():
    assert hashlib.sha256(Skeletonize.DEFAULT_TEMPLATES.read_bytes()).hexdigest() == PK12_SHA256


def test_templates_rebuild_bundled_table():
    bundled = np.load(Skeletonize.DEFAULT_TEMPLATES)
    table = build_table()
    assert table.dtype == bundled.dtype and table.shape == bundled.shape
    assert np.count_nonzero(table != bundled) == 0
