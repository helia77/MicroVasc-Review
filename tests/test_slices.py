"""
Slice order and output folder handling of save_slices and load_images (microvasc/data.py).

Each test runs twice: with OpenCV, skipped when it is not installed (it is in the `io` extra),
and with a small stand-in for cv2 that stores each image as .npy data under the image file
name, so the file order logic is tested without the `io` extra.
"""
import os
import re
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


# output folder

def write_files(folder, contents):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        (folder / name).write_text(text)


def folder_contents(folder):
    return {p.name: p.read_text() for p in folder.iterdir() if p.is_file() and p.suffix == ".txt"}


def test_missing_folder_is_created(cv2, tmp_path):
    save_slices(volume(3), tmp_path / "a" / "b")
    assert sorted(os.listdir(tmp_path / "a" / "b")) == ["img_0.bmp", "img_1.bmp", "img_2.bmp"]


def test_folder_without_files_is_used(cv2, tmp_path):
    (tmp_path / "out" / "subfolder").mkdir(parents=True)
    save_slices(volume(2), tmp_path / "out")
    assert sorted(os.listdir(tmp_path / "out")) == ["img_0.bmp", "img_1.bmp", "subfolder"]


@pytest.mark.parametrize("existing", [{"notes.txt": "keep"}, {"img_0.bmp": "old slice"}])
def test_folder_with_files_raises_and_keeps_them(cv2, tmp_path, existing):
    write_files(tmp_path / "out", existing)
    with pytest.raises(FileExistsError, match=re.escape(f"output folder {str(tmp_path / 'out')!r}")):
        save_slices(volume(3), tmp_path / "out")
    assert {p.name: p.read_text() for p in (tmp_path / "out").iterdir()} == existing


def test_overwrite_replaces_only_slice_files(cv2, tmp_path):
    out = tmp_path / "out"
    save_slices(volume(12), out)                               # img_00 to img_11
    for i in range(12):
        cv2.imwrite(str(out / f"img_{i}.bmp"), volume(12)[i])  # unpadded names of older versions
    others = {"notes.txt": "keep", "img_3.png.txt": "keep", "img_final.bmp.txt": "keep"}
    write_files(out, others)
    for name in ["img_3.png", "img_final.bmp", "image_3.bmp", "img_3.bmp.bak"]:
        (out / name).write_bytes(b"not a slice of this format")
    (out / "img_7.bmp.d").mkdir()
    (out / "img_7.bmp.d" / "inside.txt").write_text("keep")

    new = volume(5) + 1
    save_slices(new, out, overwrite=True)
    kept = ["img_3.bmp.bak", "img_3.png", "img_3.png.txt", "img_7.bmp.d", "img_final.bmp",
            "img_final.bmp.txt", "image_3.bmp", "notes.txt"]
    assert sorted(os.listdir(out)) == sorted([f"img_{i}.bmp" for i in range(5)] + kept)
    assert folder_contents(out) == others
    assert (out / "img_7.bmp.d" / "inside.txt").read_text() == "keep"
    assert (out / "img_final.bmp").read_bytes() == b"not a slice of this format"
    for i in range(5):
        np.testing.assert_array_equal(cv2.imread(str(out / f"img_{i}.bmp"), cv2.IMREAD_GRAYSCALE), new[i])


def test_overwrite_keeps_slices_of_another_format(cv2, tmp_path):
    out = tmp_path / "out"
    save_slices(volume(3), out, file_format="png")
    save_slices(volume(2), out, file_format="bmp", overwrite=True)
    assert sorted(os.listdir(out)) == ["img_0.bmp", "img_0.png", "img_1.bmp", "img_1.png", "img_2.png"]


def test_folder_path_that_is_a_file_raises(cv2, tmp_path):
    (tmp_path / "out").write_text("a file")
    with pytest.raises(FileExistsError):
        save_slices(volume(2), tmp_path / "out", overwrite=True)
    assert (tmp_path / "out").read_text() == "a file"
