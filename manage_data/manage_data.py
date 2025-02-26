import os
import nrrd
import cv2
import shutil
import struct
import numpy as np
import logging as log
import skimage as ski
import SimpleITK as sitk


def load_images(folder_path, num_img_range='all', stack=False, grayscale=False, crop_size=(0,0), crop_location = (0, 0)):
    """
    Load images from a folder into a numpy array or list.

    Args:
        folder_path (str): Path to the folder containing images.
        num_img_range (str or list): Range of images to load. 'all' for all images, or [start, end] for a range.
        stack (bool): If True, return images as a stacked numpy array. Else, return a list of images.
        grayscale (bool): If True, load images as grayscale. Else, load as RGB.
        crop_size (tuple): Size of the crop (height, width). If (0, 0), no cropping is applied.
        crop_location (tuple): Location of the crop (x, y).

    Returns:
        np.ndarray or list: Loaded images as a stacked array or list.
    """
    images_list = [f for f in os.listdir(folder_path) if f.endswith('.bmp') or
                   f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.png')]

    if not images_list:
        log.warning(f"No images found in {folder_path}")
        return None

    if num_img_range == 'all':
        img_range = (0, len(images_list))
    elif isinstance(num_img_range, int):
        img_range = (0, num_img_range)
    else:
        img_range = num_img_range
        
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


def save_slices(volume, folder_path, number='all', file_format='bmp'):
    """
    Save slices of a 3D volume as images.

    Args:
        volume (np.ndarray): 3D volume to save.
        folder_path (str): Path to the output folder.
        number (str or int): Number of slices to save. 'all' for all slices.
        file_format (str): File format for saving images ('bmp' or 'png').
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
        
    if number == 'all':
        number = volume.shape[0]
        
    for i in range(number):
        img = volume[i]
        if np.max(img) == 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        filename = os.path.join(folder_path, f'img_{i}.{file_format}')
        cv2.imwrite(filename, img)


def nrrd2npy(nrrd_path):
    """
    Load a NRRD file and convert it to a ndarray.

    Args:
        nrrd_path (str): Path to the NRRD file.

    Returns:
        np.ndarray: The volume data as a ndarray (to be used in Slicer3D).
    """
    if not os.path.exists(nrrd_path):
        raise FileNotFoundError(f"NRRD file not found: {nrrd_path}")

    data, _ = nrrd.read(nrrd_path)
    return data

def npy2nrrd(arr, filename):
    """
    Save a NumPy array as a NRRD file.

    Args:
        arr (np.ndarray): The volume data to save.
        filename (str): Path to the output NRRD file.
    Returns:
        None, saves the NRRD file.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    nrrd.write(filename, arr)
    
def resample(data, spacing, output_spacing, interpolator=sitk.sitkLinear):
    """
    Resample a 3D volume to a new spacing.

    Args:
        data (np.ndarray): Input 3D volume.
        spacing (tuple): Original spacing.
        output_spacing (tuple): Desired spacing.
        interpolator: Interpolation method (default: sitk.sitkLinear).

    Returns:
        np.ndarray: Resampled volume.
    """
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

def npy2obj(input_file, output_name, level=0.0):
    """
    Convert a numpy array to an OBJ file.

    Args:
        input_file (str or np.ndarray): Input numpy array or file path.
        output_name (str): Output OBJ file path.
        level (float): Level for marching cubes.
    """
    if isinstance(input_file, str):
        volume = np.load(input_file)
    elif isinstance(input_file, np.ndarray):
        volume = input_file
    else:
        raise ValueError("Input must be a numpy array or file path.")
        
    # marching cubes
    verts, faces, _, _ = ski.measure.marching_cubes(volume, level=level)
    
    with open(output_name, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

# convert SDP files (created by skelet_kerautret function) to one OBJ file for visualization 
def sdp2obj(input_file):
    """
    Convert SDP files (Vertex.sdp and Edges.sdp) to an OBJ file. Used for Kerautret method.

    Args:
        input_file (str): Base name of the SDP files (without extensions).
    """
    vertex_file = input_file + 'Vertex.sdp'
    edges_file = input_file + 'Edges.sdp'
    if not os.path.exists(vertex_file) or not os.path.exists(edges_file):
        raise FileNotFoundError(f"SDP files not found: {vertex_file}, {edges_file}")
    
    with open(vertex_file, 'r') as vfile, open(edges_file, 'r') as efile:
        vertices = vfile.readlines()
        edges = efile.readlines()
        
    with open(input_file+'.obj', 'w') as objfile:
        objfile.write('# vertices\n')
        for vertex in vertices:
            objfile.write('v '+ vertex.strip() + '\n')
        
        objfile.write('\n# edges\n')
        for edge in edges:
            v1, v2 = edge.strip().split()
            objfile.write(f'l {int(v1) + 1} {int(v2) + 1}\n')


def cg2obj(input_file):
    """
    Convert a CG file to an OBJ file. Used in StarLab software (Tagliasacchi method).

    Args:
        input_file (str): Path to the CG file.

    Raises:
        FileNotFoundError: If the CG file does not exist.
        IOError: If the file cannot be read or written.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CG file not found: {input_file}")
        
    with open(input_file, 'r') as infile: 
        content = infile.read()
        
    modified = content.replace('e', 'l')
    output_file = input_file.replace('.cg', '.obj')
    
    with open(output_file, 'w') as outfile:
        outfile.write(modified)
        

class vertex:
    """
    A helper class representing a vertex in a 3D network.

    Attributes:
        p (np.ndarray): The position of the vertex as a 3D array [x, y, z].
        Eout (list): List of outgoing edges.
        Ein (list): List of incoming edges.
    """
    def __init__(self, x, y, z, e_out, e_in):
        """
        Args:
            x (float): x coordinate of the vertex.
            y (float): y coordinate of the vertex.
            z (float): z coordinate of the vertex.
            e_out (list): List of outgoing edges.
            e_in (list): List of incoming edges.
        """
        self.p = np.array([x, y, z])
        self.Eout = e_out
        self.Ein = e_in


class edge:
    """
    A helper class representing an edge in a 3D network.

    Attributes:
        v (tuple): A tuple of two vertex indices (v0, v1) that the edge connects.
        p (list): A list of points defining the edge.
    """
    def __init__(self, v0, v1, p):
        self.v = (v0, v1)
        self.p = p
        
    
class linesegment:
    """
    A helper class representing a line segment in 3D space.

    Attributes:
        p (tuple): Two points (p0, p1) defining the line segment.
    """
    def __init__(self, p0, p1):
        self.p = (p0, p1)

    def pointcloud(self, spacing):
        """
        Generate a point-cloud, sampling the line segment at the specified spacing.

        Args:
            spacing (float): Distance between sampled points.

        Returns:
            list: A list of points sampling the line segment.
        """
        v = self.p[1] - self.p[0]
        l = np.linalg.norm(v)
        n = int(np.ceil(l / spacing))
        if n == 0:
            return [self.p[0], self.p[1]]
        
        pc = [self.p[0] + v * (i/n) for i in range(n+1)]
        return pc


class NWT:
    """
    A class representing a 3D network.

    Attributes:
        header (str): Header of the NWT file.
        desc (str): Description of the network.
        v (list): List of vertices in the network.
        e (list): List of edges in the network.
    """
    def __init__(self, filename):
        """
        Args:
            filename (str): Path to the input file (NWT or OBJ format).

        """
        _, fext = os.path.splitext(filename)                            #get the file extension so that we know the file type
        if fext == ".nwt":                                              #if the file extension is NWT
            self.load_nwt(filename)                                     #load a NWT file
        elif fext == ".obj":                                            #if the file extension is OBJ
            self.load_obj(filename)                                     #load an OBJ file
        else:                                                           #otherwise raise an exception
            raise ValueError("Unsupported file type. Expected '.nwt' or '.obj' file.")
    
    def load_nwt(self, filename):
        """
        Load a network from an NWT file.

        Args:
            filename (str): Path to the NWT file.

        """
        fid = open(filename, "rb")                                      #open a binary file for reading
        self.header = fid.read(14).decode("utf-8")                      #load the header
        self.desc = fid.read(58).decode("utf-8")                        #load the description
        nv = struct.unpack("I", fid.read(4))[0]                         #load the number of vertices and edges
        ne = struct.unpack("I", fid.read(4))[0]

        self.v = []                                                     #create an empty list to store the vertices        
        for _ in range(nv):                                             #iterate across all vertices
            p = np.fromfile(fid, np.float32, 3)                         #read the vertex position
            E = np.fromfile(fid, np.uint32, 2)                          #read the number of edges
            e_out = np.fromfile(fid, np.uint32, E[0])                   #read the indices of the outgoing edges
            e_in = np.fromfile(fid, np.uint32, E[1])                    #read the indices of the incoming edges
            self.v.append(vertex(p[0], p[1], p[2], e_out, e_in))        #create a vertex
            
        self.e = []                                                     #create an empty array to store the edges        
        for _ in range(ne):                                             #iterate over all edges            
            v = np.fromfile(fid, np.uint32, 2)                          #load the vertex indices that this edge connects            
            npts = struct.unpack("I", fid.read(4))[0]                   #read the number of points defining this edge            
            pv = np.fromfile(fid, np.float32, 4*npts)                   #read the array of points            
            p = [(pv[i],pv[i+1]) for i in range(0,npts,2)]              #conver the point values to an array of 4-element tuples            
            self.e.append(edge(v[0], v[1], p))                          #create and append the edge to the edge list
    
    def load_obj(self, filename):
        """
        Load a network from an OBJ file.

        Args:
            filename (str): Path to the OBJ file.
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

        self.e = []                                                     #create an empty list storing the NWT vertex IDs for each edge (inlet and outlet)
        for li, line in enumerate(lines):                               #for each line
            v0 = obj2nwt[line[0]]                                       #get the NWT index for the starting and ending points (vertices)
            v1 = obj2nwt[line[-1]]
            v_out[v0].append(li)                                        #add the line index to a list of inlet edges
            v_in[v1].append(li)                                         #add the line index to a list of outlet edges
            p = [np.array(vertices[pi - 1]) for pi in line[1:-1]]       #add the coordinates of the point that is not an end point in the NWT graph
            self.e.append(edge(v0, v1, p))                              #create an edge, specifying the inlet and outlet vertices and all defining points

        #for each vertex in the set, create a NWT vertex containing all of the necessary edge information
        self.v = []                                                     #create an empty list to store the vertices
        for si in vertex_set:                                           #for each OBJ vertex in the vertex set
            vi = obj2nwt[si]                                            #calculate the corresponding NWT index
            self.v.append(vertex(vertices[si-1][0], vertices[si-1][1], vertices[si-1][2], v_out[vi], v_in[vi]))    #create a vertex object, consisting of a position and attached edges


    def linesegments(self):
        """
        Generate line segments for all edges in the network.

        Returns:
            list: A list of line segments.
        """
        segments = []                                                   #create an empty list of line segments
        #print('num edges:', len(self.e))
        for e in self.e:                                                #for each edge in the graph
            p0 = self.v[e.v[0]].p                                       #load the first point (from the starting vertex)
            for p in e.p:                                               #for each point in the edge
                p1 = np.array([p[0], p[1], p[2]])                       #get the second point for the line segment
                segments.append(linesegment(p0, p1))                    #append the line segment to the list of line segments
                p0 = p1                                                 #update the start point for the next segment to the end point of this one
            p1 = self.v[e.v[1]].p                                       #load the last point (from the ending vertex)
            segments.append(linesegment(p0, p1))                        #append the last line segment for this edge to the list
        return segments

 
    def pointcloud(self, spacing):
        """
        Generate a point cloud sampling the network at the specified spacing.

        Args:
            spacing (float): Distance between sampled points.

        Returns:
            list: A list of points sampling the network.
        """
        segments  = self.linesegments()
        pc = []
        for i, l in enumerate(segments ):
            if i%10000 == 0:
                log.debug(f"Processed {i // 10000} segments.")
            pc.extend(l.pointcloud(spacing))
        return pc
    
    def save_obj(self, output_name):
        """
        Save the network as an OBJ file.

        Args:
            output_name (str): Path to the output OBJ file.
        """
        with open(output_name, 'w') as output:
            output.write('# vertices\n')
            for vert in self.v:             #vertex variable
                x, y, z = vert.p
                output.write(f'v {x} {y} {z}\n')
            output.write('# edges\n')
            for edg in self.e:
                v1, v2 = edg.v
                output.write(f'l {v1+1} {v2+1}\n')



def adjust_brightness_contrast(image, clip_hist_percent=1):
    """
    Adjust the brightness and contrast of an image automatically.

    Args:
        image (np.ndarray): Input image.
        clip_hist_percent (float): Percentage of histogram to clip for contrast adjustment.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The adjusted image.
            - float: Alpha (contrast adjustment factor).
            - float: Beta (brightness adjustment factor).
    """
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