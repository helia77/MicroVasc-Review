"""
Slice order, file extensions and output folder handling of save_slices and load_images
(microvasc/data.py).

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
    """Single images are stored with np.save, multi-page files with np.savez (one array per page)."""
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_ANYDEPTH = 0, 1, 2

    def imwrite(filename, img):
        with open(filename, "wb") as f:
            np.save(f, img)
        return True

    def imwritemulti(filename, imgs):
        with open(filename, "wb") as f:
            np.savez(f, *imgs)
        return True

    def _pages(filename):
        try:
            with open(filename, "rb") as f:
                data = np.load(f)
                return [data[k] for k in data.files] if isinstance(data, np.lib.npyio.NpzFile) else [data]
        except (OSError, ValueError):
            return []

    def imread(filename, flags):
        pages = _pages(filename)
        return pages[0] if pages else None

    def imreadmulti(filename, flags=1):
        pages = _pages(filename)
        return bool(pages), tuple(pages)

    def imcount(filename):
        return len(_pages(filename))

    cv2.imwrite, cv2.imwritemulti, cv2.imread, cv2.imreadmulti, cv2.imcount = (
        imwrite, imwritemulti, imread, imreadmulti, imcount)
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


# file extensions

def write_and_read(cv2, folder, names, v):
    """Write v[i] to names[i] and return what OpenCV reads back (JPEG is lossy)."""
    for name, image in zip(names, v):
        cv2.imwrite(str(folder / name), image)
    return np.stack([cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE) for name in names])


@pytest.mark.parametrize("ext", [".bmp", ".jpg", ".tif", ".png", ".tiff", ".BMP", ".JPG", ".TIF",
                                 ".TIFF", ".PNG", ".Tif", ".Tiff", ".JpG"])
def test_extension_in_any_case_is_loaded(cv2, tmp_path, ext):
    v = volume(3)
    expected = write_and_read(cv2, tmp_path, [f"img_{i}{ext}" for i in range(3)], v)
    np.testing.assert_array_equal(load_images(tmp_path, stack=True, grayscale=True), expected)


def test_mixed_extensions_load_in_name_order(cv2, tmp_path):
    names = ["s_0.tif", "s_1.TIF", "s_2.tiff", "s_3.TIFF", "s_4.png", "s_5.PNG", "s_6.bmp",
             "s_7.BMP", "s_8.jpg", "s_9.JPG", "s_10.Tif"]
    v = volume(len(names))
    expected = write_and_read(cv2, tmp_path, names, v)
    np.testing.assert_array_equal(load_images(tmp_path, stack=True, grayscale=True), expected)


def test_other_files_are_skipped(cv2, tmp_path):
    v = volume(2)
    expected = write_and_read(cv2, tmp_path, ["img_0.tif", "img_1.TIFF"], v)
    for name in ["notes.txt", "img_2.npy", "img_3.tif.bak", "img_4tif", "img_5.tiff.txt", "img_6.TI"]:
        (tmp_path / name).write_bytes(b"not an image")
    np.testing.assert_array_equal(load_images(tmp_path, stack=True, grayscale=True), expected)


# multi-page TIFF

@pytest.mark.parametrize("name", ["stack.tif", "stack.TIFF"])
def test_multi_page_tiff_pages_are_the_slices(cv2, tmp_path, name):
    v = volume(6)
    cv2.imwritemulti(str(tmp_path / name), list(v))
    np.testing.assert_array_equal(load_images(tmp_path, stack=True, grayscale=True), v)
    np.testing.assert_array_equal(load_images(tmp_path, image_range=[2, 5], stack=True, grayscale=True), v[2:5])
    np.testing.assert_array_equal(load_images(tmp_path, image_range=3, stack=True, grayscale=True), v[:3])
    pages = load_images(tmp_path, grayscale=True)
    assert isinstance(pages, list) and len(pages) == 6


def test_multi_page_tiff_in_color_and_cropped(cv2, tmp_path):
    v = volume(4)
    cv2.imwritemulti(str(tmp_path / "stack.tif"), list(v))
    _, pages = cv2.imreadmulti(str(tmp_path / "stack.tif"), flags=cv2.IMREAD_COLOR)
    expected = np.stack(pages)[:, 1:4, 2:6]
    loaded = load_images(tmp_path, stack=True, crop_size=(3, 4), crop_location=(1, 2))
    np.testing.assert_array_equal(loaded, expected)
    assert loaded.shape[:3] == (4, 3, 4)


@pytest.mark.parametrize("others", [["img_1.png"], ["img_1.tif"], ["img_1.tif", "img_2.bmp"]])
def test_multi_page_tiff_with_other_images_raises(cv2, tmp_path, others):
    v = volume(5)
    cv2.imwritemulti(str(tmp_path / "img_0.tif"), list(v))
    for name in others:
        cv2.imwrite(str(tmp_path / name), v[0])
    with pytest.raises(ValueError, match=r"'img_0.tif' has 5 pages\. A multi-page TIFF must be the only image"):
        load_images(tmp_path, stack=True, grayscale=True)


def test_two_multi_page_tiffs_raise(cv2, tmp_path):
    v = volume(3)
    cv2.imwritemulti(str(tmp_path / "a.tif"), list(v))
    cv2.imwritemulti(str(tmp_path / "b.tif"), list(v))
    with pytest.raises(ValueError, match="'a.tif' has 3 pages"):
        load_images(tmp_path, grayscale=True)


def test_single_page_tiffs_and_unreadable_file(cv2, tmp_path, caplog):
    # unchanged behavior: one slice per file, a file OpenCV cannot read is skipped with a warning
    v = volume(3)
    expected = write_and_read(cv2, tmp_path, ["img_0.tif", "img_2.tiff"], v[[0, 2]])
    (tmp_path / "img_1.tif").write_bytes(b"not an image")
    np.testing.assert_array_equal(load_images(tmp_path, stack=True, grayscale=True), expected)
    assert "Failed to load image: img_1.tif" in caplog.text


# bit depth, with OpenCV only: the stand-in keeps every array as it is

def volume16(n):
    v = np.random.default_rng(n).integers(0, 65536, (n, 5, 7), dtype=np.uint16)
    v[0, 0, :3] = [57000, 256, 255]
    return v


@pytest.mark.parametrize("ext", [".tif", ".png"])
def test_16_bit_grayscale_keeps_its_values(tmp_path, ext):
    cv2 = pytest.importorskip("cv2")
    v = volume16(4)
    for i in range(4):
        cv2.imwrite(str(tmp_path / f"img_{i}{ext}"), v[i])
    loaded = load_images(tmp_path, stack=True, grayscale=True)
    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, v)
    color = load_images(tmp_path, stack=True)
    assert color.dtype == np.uint16 and color.shape == v.shape + (3,)
    np.testing.assert_array_equal(color, np.repeat(v[..., None], 3, axis=-1))


def test_16_bit_color_keeps_its_values(tmp_path):
    cv2 = pytest.importorskip("cv2")
    v = np.stack([volume16(3), volume16(3) // 2, volume16(3) // 3], axis=-1)
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), v[i])
    loaded = load_images(tmp_path, stack=True)
    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, v)
    gray = load_images(tmp_path, stack=True, grayscale=True)
    assert gray.dtype == np.uint16 and gray.shape == v.shape[:3]


def test_16_bit_multi_page_tiff_keeps_its_values(tmp_path):
    cv2 = pytest.importorskip("cv2")
    v = volume16(5)
    cv2.imwritemulti(str(tmp_path / "stack.tif"), list(v))
    loaded = load_images(tmp_path, stack=True, grayscale=True)
    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, v)


@pytest.mark.parametrize("ext", [".bmp", ".jpg", ".tif", ".png"])
@pytest.mark.parametrize("grayscale", [True, False])
def test_8_bit_images_are_read_as_before(tmp_path, ext, grayscale):
    cv2 = pytest.importorskip("cv2")
    v = volume(3)
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"img_{i}{ext}"), np.stack([v[i], v[i] // 2, 255 - v[i]], axis=-1))
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    expected = np.stack([cv2.imread(str(tmp_path / f"img_{i}{ext}"), flags) for i in range(3)])
    loaded = load_images(tmp_path, stack=True, grayscale=grayscale)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, expected)


# mixed bit depths

MIXED = [(np.uint8, np.uint16), (np.uint16, np.uint8), (np.uint16, np.float32)]


@pytest.mark.parametrize("first,second", MIXED, ids=lambda t: np.dtype(t).name)
@pytest.mark.parametrize("stack", [True, False])
def test_mixed_bit_depths_raise(cv2, tmp_path, first, second, stack):
    # before, a stack was converted to the larger type and a list mixed types
    cv2.imwrite(str(tmp_path / "img_0.tif"), np.full((4, 5), 200, first))
    cv2.imwrite(str(tmp_path / "img_1.tif"), np.full((4, 5), 200, first))
    cv2.imwrite(str(tmp_path / "img_2.tif"), np.full((4, 5), 250, second))
    message = (f"'img_2.tif' is {np.dtype(second).name} and 'img_0.tif' is {np.dtype(first).name}. "
               "All images must have the same bit depth")
    with pytest.raises(ValueError, match=re.escape(message)):
        load_images(tmp_path, stack=stack, grayscale=True)


def test_mixed_bit_depths_outside_image_range_are_not_read(cv2, tmp_path):
    cv2.imwrite(str(tmp_path / "img_0.tif"), np.full((4, 5), 200, np.uint8))
    cv2.imwrite(str(tmp_path / "img_1.tif"), np.full((4, 5), 201, np.uint8))
    cv2.imwrite(str(tmp_path / "img_2.tif"), np.full((4, 5), 50000, np.uint16))
    loaded = load_images(tmp_path, image_range=2, stack=True, grayscale=True)
    assert loaded.dtype == np.uint8 and loaded[:, 0, 0].tolist() == [200, 201]


def test_mixed_bit_depths_in_multi_page_tiff_raise(cv2, tmp_path):
    cv2.imwritemulti(str(tmp_path / "stack.tif"), [np.full((4, 5), 200, np.uint8),
                                                   np.full((4, 5), 50000, np.uint16)])
    if cv2.imreadmulti(str(tmp_path / "stack.tif"), flags=cv2.IMREAD_ANYDEPTH)[1][1].dtype != np.uint16:
        pytest.skip("this OpenCV writes all pages with one bit depth")
    with pytest.raises(ValueError, match=re.escape("'page 1 of stack.tif' is uint16 and 'page 0 of stack.tif' is uint8")):
        load_images(tmp_path, stack=True, grayscale=True)


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


# values: one rule for the whole volume

def save_and_load(volume, folder):
    save_slices(volume, folder)
    return load_images(folder, stack=True, grayscale=True)


def test_0_to_11_volume_is_saved_unchanged(cv2, tmp_path):
    # before, the slice of 1s came back as 255 and the slice of 2s stayed 2
    v = np.broadcast_to(np.arange(12, dtype=np.uint8)[:, None, None], (12, 4, 5)).copy()
    np.testing.assert_array_equal(save_and_load(v, tmp_path / "out"), v)


@pytest.mark.parametrize("dtype", [np.uint8, np.int64, bool, np.float64])
def test_mask_is_saved_as_0_255(cv2, tmp_path, dtype):
    mask = np.random.default_rng(1).random((6, 5, 7)) < 0.4
    mask[3] = False                                             # an empty slice
    expected = mask.astype(np.uint8) * 255
    np.testing.assert_array_equal(save_and_load(mask.astype(dtype), tmp_path / "out"), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32])
def test_0_255_volume_with_dark_slices_is_saved_unchanged(cv2, tmp_path, dtype):
    v = volume(8)                                               # 0..255, 200 in every slice
    v[2] = np.random.default_rng(2).integers(0, 2, v[2].shape)  # a slice of only 0 and 1
    v[5] = 0                                                    # a slice of zeros
    v[6] = 1                                                    # a slice of ones
    np.testing.assert_array_equal(save_and_load(v.astype(dtype), tmp_path / "out"), v)


def test_values_in_0_1_are_multiplied_by_255(cv2, tmp_path):
    # a vesselness volume; before, every slice whose maximum was below 1 was saved as zeros
    v = np.random.default_rng(3).random((5, 4, 6)) * 0.7
    v[4] = 0
    np.testing.assert_array_equal(save_and_load(v, tmp_path / "out"), np.rint(v * 255).astype(np.uint8))


def test_all_zero_volume_is_saved_as_zeros(cv2, tmp_path):
    v = np.zeros((3, 4, 5), np.uint8)
    np.testing.assert_array_equal(save_and_load(v, tmp_path / "out"), v)


@pytest.mark.parametrize("v,message", [
    (np.array([300, 2], np.uint16), "values from 2 to 300"),
    (np.array([-1, 5], np.int64), "values from -1 to 5"),
    (np.array([0.5, 1.5]), "values from 0.5 to 1.5 that are not all integers"),
    (np.array([0.0, np.nan]), "NaN or inf"),
    (np.array([0.0, np.inf]), "NaN or inf"),
])
def test_other_values_raise_and_write_nothing(cv2, tmp_path, v, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        save_slices(np.broadcast_to(v[None, None, :], (3, 2, 2)), tmp_path / "out")
    assert not (tmp_path / "out").exists()
