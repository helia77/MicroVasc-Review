import os
import shutil
import struct
import numpy as np
import logging as log
import skimage as ski
from typing import List, Optional, Sequence, Tuple, Union


def load_images(folder_path: Union[str, os.PathLike], image_range: Union[str, int, Sequence[int]] = 'all',
                stack: bool = False, grayscale: bool = False, crop_size: Tuple[int, int] = (0, 0),
                crop_location: Tuple[int, int] = (0, 0)) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """
    Load the .bmp, .jpg, .tif and .png images of a folder. Needs the `io` extra.

    The files are taken in the order returned by `os.listdir`, which is not sorted.

    Args:
        folder_path: Folder containing the images.
        image_range: 'all' for all images, an int n for the first n images, or [start, end]
            for the images start to end - 1.
        stack: If True, return one array with the images along axis 0. Else return a list.
        grayscale: If True, load images as grayscale. Else load them as 3-channel BGR, the
            OpenCV channel order.
        crop_size: Crop size (rows, columns). (0, 0) means no cropping.
        crop_location: Top-left corner of the crop (row, column).

    Returns:
        The images as an array or a list. None, with a logged warning, if the folder has no
            images. Images that OpenCV cannot read are skipped with a warning.
    """
    import cv2
    images_list = [f for f in os.listdir(folder_path) if f.endswith('.bmp') or
                   f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.png')]

    if not images_list:
        log.warning(f"No images found in {folder_path}")
        return None

    if image_range == 'all':
        img_range = (0, len(images_list))
    elif isinstance(image_range, int):
        img_range = (0, image_range)
    else:
        img_range = image_range
        
    images = []
    for i in range(img_range[0], img_range[1]):
        img = cv2.imread(os.path.join(folder_path, images_list[i]), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
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


def save_slices(volume: np.ndarray, folder_path: Union[str, os.PathLike], num_slices: Union[str, int] = 'all',
                file_format: str = 'bmp') -> None:
    """
    Save the slices of a 3D volume along axis 0 as images img_0, img_1, ... Needs the `io` extra.

    If `folder_path` exists, it is deleted first with everything in it.

    Each slice is converted to uint8 on its own: a slice whose maximum is 1 is multiplied by
    255, any other slice is cast to uint8 unchanged.

    Args:
        volume: 3D volume.
        folder_path: Output folder.
        num_slices: Number of slices to save, from slice 0. 'all' for all slices.
        file_format: Image file extension, such as 'bmp' or 'png'.
    """
    import cv2
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
        
    if num_slices == 'all':
        num_slices = volume.shape[0]
        
    for i in range(num_slices):
        img = volume[i]
        if np.max(img) == 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        filename = os.path.join(folder_path, f'img_{i}.{file_format}')
        cv2.imwrite(filename, img)


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

def npy2obj(volume: Union[np.ndarray, str], output_name: Union[str, os.PathLike], level: float = 0.0) -> None:
    """
    Write the surface mesh of a volume, from marching cubes, as an OBJ file.

    Vertex coordinates are in voxels, in array axis order.

    Args:
        volume: Volume, or path of a .npy file as a string.
        output_name: Path of the output OBJ file.
        level: Level for marching cubes.

    Raises:
        ValueError: If `volume` is not an array or a string.
    """
    if isinstance(volume, str):
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


def cg2obj(input_file: str) -> None:
    """
    Convert a CG file from StarLab (Tagliasacchi method) to an OBJ file.

    Every character 'e' in the file is replaced with 'l', and the output path is `input_file`
    with '.cg' replaced by '.obj'.

    Args:
        input_file: Path of the CG file, as a string.

    Raises:
        FileNotFoundError: If the CG file does not exist.
        OSError: If the file cannot be read or written.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CG file not found: {input_file}")
        
    with open(input_file, 'r') as infile: 
        content = infile.read()
        
    modified = content.replace('e', 'l')
    output_file = input_file.replace('.cg', '.obj')
    
    with open(output_file, 'w') as outfile:
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

        Examples:
            >>> import numpy as np
            >>> from microvasc.data import LineSegment
            >>> segment = LineSegment(np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
            >>> [float(p[0]) for p in segment.point_cloud(1.0)]
            [0.0, 1.0, 2.0, 3.0]
        """
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
        filename: Path of a .nwt or .obj file. The extension is case sensitive.

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
        _, fext = os.path.splitext(filename)                            #get the file extension so that we know the file type
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
        fid = open(filename, "rb")                                      #open a binary file for reading
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

        Only `v` and `l` lines are read, and their values must be separated by single spaces.
        Each `l` line is an edge: its first and last points become vertices, the points between
        them the edge points.

        Args:
            filename: Path of the .obj file.
        """
        fid = open(filename, "r")                                       #open the file for reading        
        vertices = []                                                   #create an array of vertices
        lines = []                                                      #create an array of lines
        for line in fid:                                                #for each line in the file
            elements = line.split(" ")                                  #split it into token elements          
            if elements[-1] == '\n' and len(elements) != 1:             #make sure the last element is not \n
                elements.pop(-1)
            if elements[0] == "v":                                      #if the element is a vertex
                vertices.append([float(i) for i in elements[1:]])       #add the coordinates to the vertex list            
            if elements[0] == "l":                                      #if the element is a line            
                lines.append([int(i) for i in elements[1:]])            #add this line to the line list

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
    with `cv2.convertScaleAbs(image, alpha=alpha, beta=beta)`.

    Args:
        image: 3-channel BGR image, uint8.
        clip_hist_percent: Percent of the pixels to clip, half at each end of the histogram.

    Returns:
        The adjusted image, alpha (contrast factor) and beta (brightness offset).
    """
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    adjusted_image  = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (adjusted_image, alpha, beta)