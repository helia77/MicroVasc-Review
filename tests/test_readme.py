"""The quick-start example in README.md runs and gives values in the expected ranges."""
import re
from pathlib import Path

import numpy as np

README = Path(__file__).parents[1] / "README.md"


def quick_start_code():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quick start", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    assert match, "no python block in the Quick start section of README.md"
    return match.group(1)


def test_readme_quick_start():
    namespace = {}
    exec(compile(quick_start_code(), str(README), "exec"), namespace)

    mask, skeleton, truth = namespace["mask"], namespace["skeleton"], namespace["truth"]
    assert mask.shape == truth.shape and set(np.unique(mask)) <= {0, 1}
    assert np.all(skeleton <= mask)
    assert 0 < namespace["precision"] <= 1 and 0 < namespace["recall"] <= 1
