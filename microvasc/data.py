import os
import re
import struct
import numpy as np
import logging as log
import skimage as ski
from typing import List, Optional, Sequence, Tuple, Union


def _natural_key(name: str) -> Tuple[list, str]:
    """Sort key that compares the digit runs of a file name as numbers: img_2 before img_10."""
    parts = re.split(r'(\d+)', name)
    parts[1::2] = [int(p) for p in parts[1::2]]
    return parts, name


def load_images(folder_path: Union[str, os.PathLike], image_range: Union[str, int, Sequence[int]] = 'all',
                stack: bool = False, grayscale: bool = False, crop_size: Tuple[int, int] = (0, 0),
                crop_location: Tuple[int, int] = (0, 0)) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """
    Load the .bmp, .jpg, .tif, .tiff and .png images of a folder. Needs the `io` extra.
    The extension can be in any letter case, so .TIF and .Tiff files are loaded too.

    The files are read in natural sort order of their names: digit runs compare as numbers,
    so img_2 comes before img_10, and the zero-padded names of `save_slices` sort the same way.
    Names with equal numbers, such as img_01 and img_1, are ordered as plain strings.

    A multi-page TIFF must be the only image in the folder. Its pages are then the images, in
    page order, and `image_range` and the crop apply to the pages.

    Images keep their bit depth: 8-bit files give uint8, 16-bit files give uint16 with the
    values unchanged. Nothing is rescaled. Convert 16-bit data yourself before functions that
    expect 0..255, such as `save_slices`.

    Args:
        folder_path: Folder containing the images.
        image_range: 'all' for all images, an int n for the first n images, or [start, end]
            for the images start to end - 1.
        stack: If True, return one array with the images along axis 0. Else return a list.
        grayscale: If True, load images as grayscale. Else load them as 3-channel BGR, the
            OpenCV channel order. Both keep the bit depth.
        crop_size: Crop size (rows, columns). (0, 0) means no cropping.
        crop_location: Top-left corner of the crop (row, column).

    Returns:
        The images as an array or a list. None, with a logged warning, if the folder has no
            images. Images that OpenCV cannot read are skipped with a warning.

    Raises:
        ValueError: If the folder has a multi-page TIFF and other images.
    """
    import cv2
    extensions = ('.bmp', '.jpg', '.tif', '.tiff', '.png')
    images_list = sorted((f for f in os.listdir(folder_path) if f.lower().endswith(extensions)),
                         key=_natural_key)

    if not images_list:
        log.warning(f"No images found in {folder_path}")
        return None

    # IMREAD_ANYDEPTH keeps 16-bit and float images as they are; without it OpenCV converts to 8-bit
    flags = (cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR) | cv2.IMREAD_ANYDEPTH
    # pages of each TIFF; 0 if OpenCV cannot read the file
    page_counts = {f: cv2.imcount(os.path.join(folder_path, f))
                   for f in images_list if f.lower().endswith(('.tif', '.tiff'))}
    multi_page = [f for f, n in page_counts.items() if n > 1]
    if multi_page and len(images_list) > 1:
        raise ValueError(f"{multi_page[0]!r} has {page_counts[multi_page[0]]} pages. A multi-page "
                         f"TIFF must be the only image in the folder, and {folder_path} has "
                         f"{len(images_list)} images")
    if multi_page:
        _, pages = cv2.imreadmulti(os.path.join(folder_path, multi_page[0]), flags=flags)
        num_images = len(pages)
    else:
        num_images = len(images_list)

    if image_range == 'all':
        img_range = (0, num_images)
    elif isinstance(image_range, int):
        img_range = (0, image_range)
    else:
        img_range = image_range

    images = []
    for i in range(img_range[0], img_range[1]):
        if multi_page:
            img = pages[i]
        else:
            img = cv2.imread(os.path.join(folder_path, images_list[i]), flags)
        if img is None:
            log.warning(f"Failed to load image: {images_list[i]}")
            continue
        if crop_size != (0, 0):
            x, y = crop_location
            img = img[x:x+crop_size[0], y:y+crop_size[1]]
        images.append(img)

    if stack:
        return np.stack(images, axis=0)
    else:
        return images


def _volume_to_uint8(volume: np.ndarray) -> np.ndarray:
    """The volume as uint8, with the one rule for the whole volume described in `save_slices`."""
    v = np.asarray(volume)
    if v.dtype == bool:
        return v.astype(np.uint8) * 255
    if not (np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)):
        raise ValueError(f"save_slices needs an integer, float or bool volume, got dtype {v.dtype}")
    if v.size == 0:
        return v.astype(np.uint8)
    if not np.all(np.isfinite(v)):
        raise ValueError("save_slices writes values 0..255, and the volume holds NaN or inf. "
                         "Convert it to 0..255 first.")
    low, high = v.min(), v.max()
    if low >= 0 and high <= 1:                                      # rule 1
        return np.rint(v * 255.0).astype(np.uint8)
    integers = np.issubdtype(v.dtype, np.integer) or bool(np.all(v == np.rint(v)))
    if low >= 0 and high <= 255 and integers:                       # rule 2
        return v.astype(np.uint8)
    raise ValueError(f"save_slices writes values 0..255, and the volume holds values from {low} to "
                     f"{high}{'' if integers else ' that are not all integers'}. Convert it to "
                     f"0..255 first.")


def save_slices(volume: np.ndarray, folder_path: Union[str, os.PathLike], num_slices: Union[str, int] = 'all',
                file_format: str = 'bmp', overwrite: bool = False) -> None:
    """
    Save the slices of a 3D volume along axis 0 as images. Needs the `io` extra.

    Slice i is written as img_<i>, zero padded to the width of the last index: 10 slices give
    img_0 to img_9, 12 slices give img_00 to img_11. Plain sorting of the names keeps the slice
    order, and `load_images` reads them back in that order.

    The folder is created if it does not exist, and used as is if it holds no files
    (subfolders do not count). A folder that holds files is used only with `overwrite=True`.
    No file other than an earlier slice file is ever deleted.

    The values are converted to uint8 with one rule for the whole volume, decided from all of
    its slices, so a slice is written the same way whether 5 or all slices are saved:

    1. All values in [0, 1], including bool and an all-zero volume: multiplied by 255 and
       rounded. A 0/1 mask is saved as 0/255, a vesselness volume in [0, 1] as 0..255.
    2. Otherwise, all values are integers in 0..255, in any dtype: saved unchanged.
    3. Anything else, such as values above 255, negative values, NaN or inf, or non-integer
       values above 1: `ValueError`. Convert the volume to 0..255 first.

    A 0..255 volume that holds only the values 0 and 1 falls under rule 1 and is saved as
    0/255.

    Args:
        volume: 3D volume.
        folder_path: Output folder.
        num_slices: Number of slices to save, from slice 0. 'all' for all slices.
        file_format: Image file extension, such as 'bmp' or 'png'.
        overwrite: If True, write into a folder that holds files. Its files named
            img_<number>.<file_format>, padded or not, are removed first, and all other files
            are kept.

    Raises:
        ValueError: If the values fit neither rule 1 nor rule 2. Nothing is written.
        FileExistsError: If `folder_path` holds files and `overwrite` is False, or if
            `folder_path` is a file.
    """
    import cv2
    volume = _volume_to_uint8(volume)
    folder = os.fspath(folder_path)
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if files and not overwrite:
        raise FileExistsError(f"output folder {folder!r} already holds {len(files)} file(s). Pass "
                              f"overwrite=True to replace its img_<number>.{file_format} slice "
                              f"files, other files are kept.")
    if overwrite:
        # only the files this function writes: img_ + digits + the same extension
        own = re.compile(rf"img_[0-9]+\.{re.escape(file_format)}")
        for f in files:
            if own.fullmatch(f):
                os.remove(os.path.join(folder, f))

    if num_slices == 'all':
        num_slices = volume.shape[0]
        
    width = len(str(max(num_slices - 1, 0)))
    for i in range(num_slices):
        filename = os.path.join(folder_path, f'img_{i:0{width}d}.{file_format}')
        cv2.imwrite(filename, volume[i])


def nrrd2npy(nrrd_path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Read an NRRD file into a NumPy array with pynrrd. Needs the `io` extra.

    Args:
        nrrd_path: Path to the NRRD file.

    Returns:
        The volume, in the axis order of pynrrd's default reader.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(nrrd_path):
        raise FileNotFoundError(f"NRRD file not found: {nrrd_path}")

    import nrrd
    data, _ = nrrd.read(nrrd_path)
    return data

def npy2nrrd(arr: np.ndarray, filename: str) -> None:
    """
    Write a NumPy array as an NRRD file with pynrrd. Needs the `io` extra.

    `nrrd2npy` reads the file back to the same array.

    Args:
        arr: Volume to save.
        filename: Path of the output NRRD file, as a string.

    Raises:
        ValueError: If `arr` is not a NumPy array.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    import nrrd
    nrrd.write(filename, arr)
    
def resample(data: np.ndarray, spacing: Sequence[float], output_spacing: Sequence[float],
             interpolator: Optional[int] = None) -> np.ndarray:
    """
    Resample a 3D volume to a new spacing with SimpleITK. Needs the `io` extra.

    Spacings are in SimpleITK order (x, y, z), the reverse of the array axes: `spacing[0]`
    applies to array axis 2. The output size along each axis is
    int(size * spacing / output_spacing), and voxels outside the input are 0.

    Args:
        data: 3D volume.
        spacing: Voxel spacing of `data`, (x, y, z).
        output_spacing: Voxel spacing of the output, (x, y, z).
        interpolator: A SimpleITK interpolator such as `sitk.sitkNearestNeighbor`. None means
            `sitk.sitkLinear`.

    Returns:
        The resampled volume, float32.
    """
    import SimpleITK as sitk
    if interpolator is None:
        interpolator = sitk.sitkLinear
    log.info(f"Resampling volume to spacing {output_spacing}")
    image = sitk.GetImageFromArray(data)
    image = sitk.Cast(image, sitk.sitkFloat32)
    image.SetSpacing(spacing)

    size = image.GetSize()
    size_scaling = np.array(spacing) / np.array(output_spacing)
    output_size = tuple(int(s * sc) for s, sc in zip(size, size_scaling))

    resampled = sitk.Resample(
        image,
        output_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        output_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )

    return sitk.GetArrayFromImage(resampled)

def npy2obj(volume: Union[np.ndarray, str, os.PathLike], output_name: Union[str, os.PathLike],
            level: float = 0.0) -> None:
    """
    Write the surface mesh of a volume, from marching cubes, as an OBJ file.

    Vertex coordinates are in voxels, in array axis order.

    Args:
        volume: Volume, or path of a .npy file.
        output_name: Path of the output OBJ file.
        level: Level for marching cubes.

    Raises:
        ValueError: If `volume` is not an array or a path.
    """
    if isinstance(volume, (str, os.PathLike)):
        volume = np.load(volume)
    elif not isinstance(volume, np.ndarray):
        raise ValueError("Input must be a numpy array or file path.")
        
    # marching cubes
    verts, faces, _, _ = ski.measure.marching_cubes(volume, level=level)
    
    with open(output_name, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

def npy2off(volume: Union[np.ndarray, str, os.PathLike], output_name: Union[str, os.PathLike],
            level: float = 0.5) -> Tuple[int, int]:
    """
    Write the surface mesh of a binarized volume as an OFF file, the input of the Kerautret
    method.

    The mesh comes from marching cubes on the binarized volume, as described in the paper.
    The paper does not give the exact mesh settings. Vertex coordinates are in voxels, in
    array axis order.

    Args:
        volume: Volume, or path of a .npy file.
        output_name: Path of the output OFF file.
        level: Level for marching cubes. 0.5 (default) for 0/1 volumes.

    Returns:
        Number of vertices and number of faces written.

    Raises:
        ValueError: If `volume` is not an array or a path.

    Examples:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from microvasc.data import npy2off
        >>> volume = np.zeros((6, 6, 6), dtype=np.uint8)
        >>> volume[2:4, 2:4, 2:4] = 1
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     npy2off(volume, os.path.join(folder, "cube.off"))
        (24, 44)
    """
    if isinstance(volume, (str, os.PathLike)):
        volume = np.load(volume)
    elif not isinstance(volume, np.ndarray):
        raise ValueError("Input must be a numpy array or file path.")

    # marching cubes
    verts, faces, _, _ = ski.measure.marching_cubes(volume, level=level)

    # OFF face indices are 0-based
    with open(output_name, 'w') as f:
        f.write('OFF\n')
        f.write(f'{len(verts)} {len(faces)} 0\n')
        for v in verts:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
    return len(verts), len(faces)

# convert SDP files (created by skelet_kerautret function) to one OBJ file for visualization 
def sdp2obj(input_file: Union[str, os.PathLike]) -> str:
    """
    Convert the Kerautret output <prefix>Vertex.sdp and <prefix>Edges.sdp to <prefix>.obj.

    Each vertex line becomes a `v` line and each edge a two-point `l` line.

    Args:
        input_file: The prefix of the SDP files.

    Returns:
        Path of the OBJ file, <prefix>.obj.

    Raises:
        FileNotFoundError: If one of the two SDP files does not exist.
    """
    input_file = os.fspath(input_file)
    vertex_file = input_file + 'Vertex.sdp'
    edges_file = input_file + 'Edges.sdp'
    if not os.path.exists(vertex_file) or not os.path.exists(edges_file):
        raise FileNotFoundError(f"SDP files not found: {vertex_file}, {edges_file}")

    with open(vertex_file, 'r') as vfile, open(edges_file, 'r') as efile:
        vertices = [line for line in vfile if line.strip()]
        edges = [line for line in efile if line.strip()]

    obj_file = input_file + '.obj'
    with open(obj_file, 'w') as objfile:
        objfile.write('# vertices\n')
        for vertex in vertices:
            objfile.write('v '+ vertex.strip() + '\n')

        objfile.write('\n# edges\n')
        for edge in edges:
            v1, v2 = edge.strip().split()
            objfile.write(f'l {int(v1) + 1} {int(v2) + 1}\n')
    return obj_file


def cg2obj(input_file: Union[str, os.PathLike]) -> None:
    """
    Convert a CG file from StarLab (Tagliasacchi method) to an OBJ file.

    The CG writer of starlab-mcfskel emits a `# D:3 NV:<n> NE:<m>` header, `v <x> <y> <z>`
    vertex lines and `e <i> <j>` edge lines with 1-based indices, as in OBJ. Each edge line
    becomes an OBJ line `l <i> <j>`: only the leading 'e' is replaced. All other lines, numbers
    such as 1e-05 and line endings are copied unchanged.

    The output is written next to the input, with the .cg suffix replaced by .obj. An existing
    file of that name is overwritten. The input file is never changed.

    Args:
        input_file: Path of the CG file. Its suffix must be .cg, in any letter case.

    Raises:
        ValueError: If the suffix is not .cg.
        FileNotFoundError: If the CG file does not exist.
        OSError: If the file cannot be read or written.
    """
    root, suffix = os.path.splitext(os.fspath(input_file))
    if suffix.lower() != '.cg':
        raise ValueError(f"expected a .cg file, got {os.fspath(input_file)!r}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CG file not found: {input_file}")

    # newline='' keeps the line endings of the input
    with open(input_file, 'r', newline='') as infile:
        content = infile.read()

    # an edge line starts with 'e' and a space or tab
    modified = re.sub(r'^e(?=[ \t])', 'l', content, flags=re.MULTILINE)
    with open(root + '.obj', 'w', newline='') as outfile:
        outfile.write(modified)
        

class Vertex:
    """
    A vertex of a network.

    Args:
        x: First coordinate.
        y: Second coordinate.
        z: Third coordinate.
        e_out: Indices of the edges that start at this vertex.
        e_in: Indices of the edges that end at this vertex.

    Attributes:
        p: Position, np.array([x, y, z]).
        e_out: Indices of the edges that start at this vertex.
        e_in: Indices of the edges that end at this vertex.
    """
    def __init__(self, x: float, y: float, z: float, e_out: Sequence[int], e_in: Sequence[int]) -> None:
        self.p = np.array([x, y, z])
        self.e_out = e_out
        self.e_in = e_in


class Edge:
    """
    An edge of a network, a polyline between two vertices.

    Args:
        v0: Index of the start vertex.
        v1: Index of the end vertex.
        p: Points between the two vertices.

    Attributes:
        v: The vertex indices (v0, v1).
        p: Points between the two vertices, without the vertices themselves. From a .nwt file
            they are (x, y, z, radius) tuples, from an .obj file [x, y, z] arrays.
    """
    def __init__(self, v0: int, v1: int, p: List[Sequence[float]]) -> None:
        self.v = (v0, v1)
        self.p = p
        
    
class LineSegment:
    """
    A straight line segment in 3D.

    Args:
        p0: Start point, an array of 3 coordinates.
        p1: End point, an array of 3 coordinates.

    Attributes:
        p: The end points (p0, p1).
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray) -> None:
        self.p = (p0, p1)

    def point_cloud(self, spacing: float) -> List[np.ndarray]:
        """
        Evenly spaced points along the segment, at most `spacing` apart.

        Args:
            spacing: Largest distance between two neighboring points.

        Returns:
            The points, including both end points. A zero-length segment gives p0 and p1.

        Raises:
            ValueError: If `spacing` is not a finite number greater than 0.

        Examples:
            >>> import numpy as np
            >>> from microvasc.data import LineSegment
            >>> segment = LineSegment(np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
            >>> [float(p[0]) for p in segment.point_cloud(1.0)]
            [0.0, 1.0, 2.0, 3.0]
        """
        if np.ndim(spacing) != 0 or not np.isfinite(spacing) or not spacing > 0:
            raise ValueError(f"spacing must be a finite number greater than 0, got {spacing!r}")
        v = self.p[1] - self.p[0]
        l = np.linalg.norm(v)
        n = int(np.ceil(l / spacing))
        if n == 0:
            return [self.p[0], self.p[1]]

        pc = [self.p[0] + v * (i/n) for i in range(n+1)]
        return pc


class NWT:
    r"""
    A network read from a .nwt or .obj file.

    Args:
        filename: Path of a .nwt or .obj file. The extension can be in any letter case.

    Attributes:
        header: File header. "nwtfileformat " for an OBJ file.
        desc: File description.
        vertices: The vertices, a list of `Vertex`.
        edges: The edges, a list of `Edge`.

    Raises:
        ValueError: If the extension is not .nwt or .obj.

    Examples:
        >>> import os, tempfile
        >>> from microvasc.data import NWT
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "network.obj")
        ...     with open(path, "w") as f:
        ...         _ = f.write("v 0 0 0\nv 1 0 0\nv 2 0 0\nv 2 3 0\nl 1 2 3 4\n")
        ...     network = NWT(path)
        >>> len(network.vertices), len(network.edges), len(network.line_segments())
        (2, 1, 3)
    """
    def __init__(self, filename: Union[str, os.PathLike]) -> None:
        fext = os.path.splitext(filename)[1].lower()                    #get the file extension so that we know the file type
        if fext == ".nwt":                                              #if the file extension is NWT
            self.load_nwt(filename)                                     #load a NWT file
        elif fext == ".obj":                                            #if the file extension is OBJ
            self.load_obj(filename)                                     #load an OBJ file
        else:                                                           #otherwise raise an exception
            raise ValueError("Unsupported file type. Expected '.nwt' or '.obj' file.")
    
    def load_nwt(self, filename: Union[str, os.PathLike]) -> None:
        """
        Read a binary .nwt file into `header`, `desc`, `vertices` and `edges`.

        The file holds a 14-byte header, a 58-byte description, the vertex and edge counts,
        each vertex as its position and the indices of its outgoing and incoming edges, and each
        edge as its two vertex indices and its points as (x, y, z, radius).

        Args:
            filename: Path of the .nwt file.
        """
        with open(filename, "rb") as fid:                                   #open a binary file for reading, closed on return or error
            self.header = fid.read(14).decode("utf-8")                      #load the header
            self.desc = fid.read(58).decode("utf-8")                        #load the description
            nv = struct.unpack("I", fid.read(4))[0]                         #load the number of vertices and edges
            ne = struct.unpack("I", fid.read(4))[0]

            self.vertices = []                                              #create an empty list to store the vertices
            for _ in range(nv):                                             #iterate across all vertices
                p = np.fromfile(fid, np.float32, 3)                         #read the vertex position
                E = np.fromfile(fid, np.uint32, 2)                          #read the number of edges
                e_out = np.fromfile(fid, np.uint32, E[0])                   #read the indices of the outgoing edges
                e_in = np.fromfile(fid, np.uint32, E[1])                    #read the indices of the incoming edges
                self.vertices.append(Vertex(p[0], p[1], p[2], e_out, e_in)) #create a vertex

            self.edges = []                                                 #create an empty array to store the edges
            for _ in range(ne):                                             #iterate over all edges
                v = np.fromfile(fid, np.uint32, 2)                          #load the vertex indices that this edge connects
                npts = struct.unpack("I", fid.read(4))[0]                   #read the number of points defining this edge
                pv = np.fromfile(fid, np.float32, 4*npts)                   #read the array of points
                p = [tuple(pv[i:i+4]) for i in range(0, 4*npts, 4)]         #group the values into (x, y, z, radius) tuples
                self.edges.append(Edge(v[0], v[1], p))                      #create and append the edge to the edge list

    def load_obj(self, filename: Union[str, os.PathLike]) -> None:
        """
        Read an OBJ file into `header`, `desc`, `vertices` and `edges`.

        Only `v` and `l` lines are read. Their values can be separated by any spaces or tabs.
        Each `l` line is an edge: its first and last points become vertices, the points between
        them the edge points.

        Args:
            filename: Path of the .obj file.
        """
        vertices = []                                                   #create an array of vertices
        lines = []                                                      #create an array of lines
        with open(filename, "r") as fid:                                #open the file for reading, closed on return or error
            for line in fid:                                            #for each line in the file
                elements = line.split()                                 #split it on any whitespace
                if not elements:                                        #skip empty lines
                    continue
                if elements[0] == "v":                                  #if the element is a vertex
                    vertices.append([float(i) for i in elements[1:]])   #add the coordinates to the vertex list
                if elements[0] == "l":                                  #if the element is a line
                    lines.append([int(i) for i in elements[1:]])        #add this line to the line list

        self.header = "nwtfileformat "                                  #assign a header and description
        self.desc = "File generated from OBJ"
                                                                        #insert the first and last vertex ID for each line into a set
        vertex_set = set()                                              #create an empty set
        for line in lines:                                              #for each line in the list of lines
            vertex_set.add(line[0])                                     #add the first and last vertex to the vertex set (this will remove redundancies)
            vertex_set.add(line[-1])
        
        
        obj2nwt = {si:vi for vi, si in enumerate(vertex_set)}           #create a mapping between OBJ vertex indices and NWT vertex indices

        #iterate through each line (edge), assigning them to their starting and ending vertices
        v_out = [[] for _ in range(len(vertex_set))]                    #create an array of empty lists storing the inlet and outlet edges for each vertex
        v_in = [[] for _ in range(len(vertex_set))]

        self.edges = []                                                 #create an empty list storing the NWT vertex IDs for each edge (inlet and outlet)
        for li, line in enumerate(lines):                               #for each line
            v0 = obj2nwt[line[0]]                                       #get the NWT index for the starting and ending points (vertices)
            v1 = obj2nwt[line[-1]]
            v_out[v0].append(li)                                        #add the line index to a list of inlet edges
            v_in[v1].append(li)                                         #add the line index to a list of outlet edges
            p = [np.array(vertices[pi - 1]) for pi in line[1:-1]]       #add the coordinates of the point that is not an end point in the NWT graph
            self.edges.append(Edge(v0, v1, p))                          #create an edge, specifying the inlet and outlet vertices and all defining points

        #for each vertex in the set, create a NWT vertex containing all of the necessary edge information
        self.vertices = []                                              #create an empty list to store the vertices
        for si in vertex_set:                                           #for each OBJ vertex in the vertex set
            vi = obj2nwt[si]                                            #calculate the corresponding NWT index
            self.vertices.append(Vertex(vertices[si-1][0], vertices[si-1][1], vertices[si-1][2], v_out[vi], v_in[vi])) #create a vertex object, consisting of a position and attached edges


    def line_segments(self) -> List[LineSegment]:
        """
        The straight segments of all edges, from the start vertex through the edge points to
        the end vertex.

        Returns:
            The segments, edge by edge.
        """
        segments = []                                                   #create an empty list of line segments
        #print('num edges:', len(self.edges))
        for e in self.edges:                                            #for each edge in the graph
            p0 = self.vertices[e.v[0]].p                                #load the first point (from the starting vertex)
            for p in e.p:                                               #for each point in the edge
                p1 = np.array([p[0], p[1], p[2]])                       #get the second point for the line segment
                segments.append(LineSegment(p0, p1))                    #append the line segment to the list of line segments
                p0 = p1                                                 #update the start point for the next segment to the end point of this one
            p1 = self.vertices[e.v[1]].p                                #load the last point (from the ending vertex)
            segments.append(LineSegment(p0, p1))                        #append the last line segment for this edge to the list
        return segments

 
    def point_cloud(self, spacing: float) -> List[np.ndarray]:
        """
        Points along all segments, from `LineSegment.point_cloud` on each segment.

        A point shared by two segments appears once for each of them.

        Args:
            spacing: Largest distance between two neighboring points of a segment.

        Returns:
            The points, segment by segment.

        Raises:
            ValueError: If `spacing` is not a finite number greater than 0.
        """
        segments  = self.line_segments()
        pc = []
        for i, l in enumerate(segments ):
            if i%10000 == 0:
                log.debug(f"Processed {i // 10000} segments.")
            pc.extend(l.point_cloud(spacing))
        return pc
    
    def save_obj(self, output_name: Union[str, os.PathLike]) -> None:
        """
        Write the vertices and edges as an OBJ file.

        Each edge is written as a straight line between its two vertices. The edge points are
        not written.

        Args:
            output_name: Path of the output OBJ file.
        """
        with open(output_name, 'w') as output:
            output.write('# vertices\n')
            for vert in self.vertices:      #vertex variable
                x, y, z = vert.p
                output.write(f'v {x} {y} {z}\n')
            output.write('# edges\n')
            for edg in self.edges:
                v1, v2 = edg.v
                output.write(f'l {v1+1} {v2+1}\n')



def adjust_brightness_contrast(image: np.ndarray, clip_hist_percent: float = 1) -> Tuple[np.ndarray, float, float]:
    """
    Stretch the contrast of an image by clipping its gray-level histogram. Needs the `io` extra.

    The gray levels below and above the clipped part of the histogram are mapped to 0 and 255
    with `cv2.convertScaleAbs(image, alpha=alpha, beta=beta)`. If no gray levels are left
    between the two cuts, as in a constant image, the image is returned unchanged (a copy) with
    alpha 1.0 and beta 0.0.

    Args:
        image: Grayscale (2D) or 3-channel BGR image, uint8.
        clip_hist_percent: Percent of the pixels to clip, half at each end of the histogram.

    Returns:
        The adjusted image, alpha (contrast factor) and beta (brightness offset).
    """
    import cv2
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index-1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut, not below the left cut
    maximum_gray = hist_size -1
    while maximum_gray > minimum_gray and accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # No gray levels left between the cuts, for example a constant image
    if maximum_gray == minimum_gray:
        return (image.copy(), 1.0, 0.0)

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    adjusted_image  = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (adjusted_image, alpha, beta)