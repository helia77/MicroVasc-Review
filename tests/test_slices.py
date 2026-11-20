"""
Slice order of save_slices and load_images (microvasc/data.py).

Each test runs twice: with OpenCV, skipped when it is not installed (it is in the `io` extra),
and with a small stand-in for cv2 that stores each image as .npy data under the image file
name, so the file order logic is tested without the `io` extra.
"""
import os
import sys
import types

import numpy as np
import pytest

from microvasc.data import load_images, save_slices


def _fake_cv2():
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR = 0, 1

    def imwrite(filename, img):
        with open(filename, "wb") as f:
            np.save(f, img)
        return True

    def imread(filename, flags):
        with open(filename, "rb") as f:
            return np.load(f)

    cv2.imwrite, cv2.imread = imwrite, imread
    return cv2


@pytest.fixture(params=["opencv", "stand-in"])
def cv2(request, monkeypatch):
    if request.param == "opencv":
        return pytest.importorskip("cv2")
    fake = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


def volume(n):
    # every slice has a value above 1, so save_slices writes it unchanged
    v = np.random.default_rng(n).integers(0, 256, (n, 5, 7), dtype=np.uint8)
    v[:, 0, 0] = 200
    return v


def list_in_order(monkeypatch, folder, names):
    """Make os.listdir return `names` for `folder`, like a file system that does not sort."""
    real = os.listdir
    monkeypatch.setattr(os, "listdir",
                        lambda path: list(names) if os.fspath(path) == os.fspath(folder) else real(path))


def test_round_trip_keeps_order(cv2, tmp_path):
    v = volume(12)
    save_slices(v, tmp_path / "slices")
    assert sorted(os.listdir(tmp_path / "slices")) == [f"img_{i:02d}.bmp" for i in range(12)]
    np.testing.assert_array_equal(load_images(tmp_path / "slices", stack=True, grayscale=True), v)


@pytest.mark.parametrize("n,first,last", [(1, "img_0", "img_0"), (10, "img_0", "img_9"),
                                          (11, "img_00", "img_10"), (101, "img_000", "img_100")])
def test_names_are_zero_padded_to_the_last_index(cv2, tmp_path, n, first, last):
    save_slices(np.full((n, 2, 2), 7, np.uint8), tmp_path / "slices", file_format="png")
    names = sorted(os.listdir(tmp_path / "slices"))
    assert len(names) == n and names[0] == f"{first}.png" and names[-1] == f"{last}.png"


@pytest.mark.parametrize("pattern", ["img_{i}.bmp", "img_{i:02d}.bmp", "slice{i}_z.png"])
def test_files_written_in_shuffled_order_load_in_order(cv2, tmp_path, monkeypatch, pattern):
    # "img_{i}.bmp" is the naming of save_slices before zero padding
    v = volume(12)
    order = np.random.default_rng(0).permutation(12)
    names = [pattern.format(i=i) for i in order]
    for i, name in zip(order, names):
        cv2.imwrite(str(tmp_path / name), v[i])
    list_in_order(monkeypatch, tmp_path, names)
    np.testing.assert_array_equal(load_images(tmp_path, stack=True, grayscale=True), v)
    np.testing.assert_array_equal(load_images(tmp_path, image_range=[3, 7], stack=True, grayscale=True), v[3:7])
    np.testing.assert_array_equal(load_images(tmp_path, image_range=4, stack=True, grayscale=True), v[:4])
