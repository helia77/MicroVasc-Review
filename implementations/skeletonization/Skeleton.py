import os
import sys
import time
import skfmm            # pip install scikit-fmm
import subprocess
import numpy as np
import logging as log
import manage_data as md
import skimage.morphology as mph
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Skeleton:
    """
    A class for skeletonizing 3D binarized volumes using various methods.
    
    Attributes:
        None
    """
    
    def __init__(self):
        log.info("Skeleton object created.")
        
    def lee(self, volume):
        """
        Compute the skeleton of the input binarized volume via Lee's method (thinning).
        
        Args:
            volume (ndarray): The input binarized volume to be skeletonized. (0: background, 1: object)
        Returns:
            ndarray: A binary numpy array containin the 1-D skeleton
        References:
        .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
           via 3-D medial surface/axis thinning algorithms.
           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.
            
        For full description, refer to https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/_skeletonize.py
        """        
        # alternative function : mph.skeletonize(sample_gr, method='lee')
        log.info("Applying Lee's thinning method.")
        return mph.skeletonize_3d(volume)
    
    
    def palagyi(self, volume):
        """ 
        Compute the skeleton of the input binarized volume via Palagyi's method (thinning).
        -> Based on the code of ClearMap2 repository (@author: ChristophKirst)

        Args:
            volume (ndarray): The input binarized volume to be skeletonized. (0: background, 1: object)
        Requires: 
            "PK12.npy" file, contatinig the 14-templates, located in skeletonization folder
        Returns:
            ndarray: A binary numpy array containin the 1-D skeleton

        References:
        .. [Palagy1999] Palagyi & Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm,
            Graphical Models and Image Processing 61, 199-221 (1999).

        """
        log.info("Applying Palagyi's thinning method.")
        # extract indices vessel points in the input
        points_arg = np.argwhere(volume)
        binary = np.copy(volume)
        deletable = np.load('PK12.npy')
        
        # create an array with base 2 numbers on the cube for convolution and LUT matching, default center is 0
        cube = np.zeros((3,3,3), dtype=int);
        k = 0;
        for z in range(3):
          for y in range(3):
            for x in range(3):
              if x == 1 and y ==1 and z == 1:
                cube[x,y,z] = 0;
              else:
                cube[x, y, z] = 2**k;
                k+=1;
        
        rotations = self._rotations12(cube)
        
        # the 6-Neighborhood excluding center
        n6 = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                    [[0,1,0],[1,0,1],[0,1,0]], 
                    [[0,0,0],[0,1,0],[0,0,0]]]);
        
                
        # Border point definition: a black point is a border point if its N_6 neighborhood has at least one white point
        while True:
            # Find which black points are border points 
            # the index of the border varibale is the same as black points in point_arg
            border = self._convolution_3d(binary, n6(), points_arg) < 6
            border_points = points_arg[border]
            border_ids = np.nonzero(border)[0]
            keep = np.ones(len(border), dtype=bool)
            
            iterat = 0
            for i in range(12):         # 12 sub-iterations
                removables = self._convolution_3d(binary, rotations[i], border_points)
                rem_borders = deletable[removables]
                rem_points = border_points[rem_borders]
                
                binary[rem_points[:, 0], rem_points[:, 1], rem_points[:, 2]] = 0
                keep[border_ids[rem_borders]] = False
                
                iterat += len(rem_points)
            
            # update foreground
            points_arg = points_arg[keep]
            if iterat == 0:
                break
        
        log.info("Palagyi's skeletonization completed.")
        return binary
    
    def _convolution_3d(input_data, kernel, points):
        """
        Perform 3D convolution at specific points.

        Args:
            input_data (ndarray): Input 3D volume.
            kernel (ndarray): Convolution kernel.
            points (ndarray): Points to perform convolution on.

        Returns:
            ndarray: Convolution results at the specified points.
        """
        num_points = points.shape[0]
        output = np.zeros(num_points, dtype=kernel.dtype)
        
        dk, dj, di = input_data.shape
        
        for n in range(num_points):
            z, y, x = points[n]
            
            for k in range(3):
                zk = z + k - 1
                if zk < dk and zk >= 0:
                    for j in range(3):
                        yj = y + j - 1
                        if yj < dj and yj >= 0:
                            for i in range(3):
                                xi = x + i - 1
                                if xi < di and xi >= 0:        
                                    output[n] += input_data[zk, yj, xi] * kernel[k, j, i]
        return output
    
    # rotate a cube around an axis in 90 degrees steps
    def _rotate(self, input_data, axis=2, steps = 0):
        """
        Rotate a cube around an axis in 90-degree steps.

        Args:
            input_data (ndarray): Input 3D cube.
            axis (int): Axis of rotation (0: x, 1: y, 2: z).
            steps (int): Number of 90-degree steps to rotate.

        Returns:
            ndarray: Rotated cube.
        """
        cube = input_data.copy();  
        steps = steps % 4;
        if steps == 0:
            return cube;
        
        if axis == 0:
            if steps == 1:
                return cube[:, ::-1, :].swapaxes(1, 2)
            elif steps == 2:    # rotate 180 degrees around x
                return cube[:, ::-1, ::-1]
            elif steps == 3:    # rotate 270 degrees around x
                return cube.swapaxes(1, 2)[:, ::-1, :]
            
        elif axis == 1:
            if steps == 1:
                return cube[:, :, ::-1].swapaxes(2, 0)
            elif steps == 2:    # rotate 180 degrees around x
                return cube[::-1, :, ::-1]
            elif steps == 3:    # rotate 270 degrees around x
                return cube.swapaxes(2, 0)[:, :, ::-1]
            
        elif axis == 2:         # z axis rotation
            if steps == 1:
                return cube[::-1, :, :].swapaxes(0, 1)
            elif steps == 2:    # rotate 180 degrees around z
                return cube[::-1, ::-1, :]
            elif steps == 3:    # rotate 270 degrees around z
                return cube.swapaxes(0, 1)[::-1, :, :]

    # generate rotations in 12 diagonal directions
    def _rotations12(self, cube):
        """
        Generate rotations in 12 diagonal directions.

        Args:
            cube (ndarray): Input 3D cube.

        Returns:
            list: List of rotated cubes.
        """
        rotUS = cube.copy();
        rotUW = self.rotate(cube, axis = 2, steps = 1);  
        rotUN = self.rotate(cube, axis = 2, steps = 2); 
        rotUE = self.rotate(cube, axis = 2, steps = 3);  
    
        rotDS = self.rotate(cube,  axis = 1, steps = 2);
        rotDW = self.rotate(rotDS, axis = 2, steps = 1); 
        rotDN = self.rotate(rotDS, axis = 2, steps = 2); 
        rotDE = self.rotate(rotDS, axis = 2, steps = 3);
    
        rotSW = self.rotate(cube, axis = 1, steps = 1);   
        rotSE = self.rotate(cube, axis = 1, steps = 3); 
    
        rotNW = self.rotate(rotUN, axis = 1, steps = 1);
        rotNE = self.rotate(rotUN, axis = 1, steps = 3);
        
        return [rotUS, rotNE, rotDW,  rotSE, rotUW, rotDN,  rotSW, rotUN, rotDE,  rotNW, rotUE, rotDS];
  
    def kerautret(self, exe_path, input_name, output_name, dilateDist=2.0, deltaG=3.0, radius=10.0, threshold=0.5):
        """
        Compute the skeleton of the input binarized volume via Kerautret's method (Gradient-Based).
        The exe file can be obtained by compiling the main Kerautret's project.
        -> Based on the code of https://github.com/kerautret/CDCVAM/tree/master (@author: kerautret)
        
        Args:
            exe_path (str):         Path to the execution file ("CenterLineGeodesicGraph" function)
            input_name (str):       Path to the input volume in OFF format
            output_name (str):      Name of the output file
            dilateDist (float):     Dilate distance of the confidence voxels. Defaults to 2.0
            deltaG (float):         The parameter to consider interval of distances. Defaults to 3.0
            radius (float):         The radius used to compute the accumulation. Defaults to 10.0
            threshold (float):      The threshold in the confidence estimation. Defaults to 0.5

        Returns:
            None: Saves the skeleton as an OBJ file
            
        References:
        .. [Kerautret2016] B. Kerautret, A. Krähenbühl, I. Debled-Rennesson and J. -O. Lachaud, 
        "Centerline detection on partial mesh scans by confidence vote in accumulation map," 2016, 
        pp. 1376-1381, doi: 10.1109/ICPR.2016.7899829.
        """
        
        log.info("Applying Kerautret's skeletonization method.")
        # Remove the previous files if exist
        if os.path.isfile(output_name+'Vertex.sdp'):
            os.remove(output_name+'Vertex.sdp')
            os.remove(output_name+'Edges.sdp')
        command = [
            exe_path,
            '-i', input_name,
            "-o", output_name,
            "--dilateDist", str(dilateDist),
            "-g", str(deltaG),
            "-R", str(radius),
            "-t", str(threshold)
        ]
        # Execute the command
        subprocess.run(command, check=True)                         # saves three files: Vertex, Edges, and Radius in .SDP format
        
        # Convert to .OBJ and save
        md.sdp2obj(output_name)
        log.info("Kerautret's skeletonization completed.")
        
    def kline(self, vol, startID, **kwargs):
        """
        Compute the skeleton of the input binarized volume via Kline's method (Gradient-Based).
        -> Based on the code of https://github.com/TLKline/poreture (@author: TLKline)
        
        Args:
            vol (ndarray): The input binarized volume to be skeletonized. (0: background, 1: object)
            startID (list): Index of root, as [x,y,z] location
        
            Optional Args (kwargs):
                dist_map_weight (int): Weight for distance map. Defaults to 6.
                min_branch_to_root (int): Minimum branch length to root. Defaults to 10.

        Returns:
            tuple: A binary numpy array containin a 1-D skeleton of the input volume
        Reference:
            kline_vessel - [Kline et al. ABME 2010]
            kline_pore - [Kline et al. J Porous Mat 2011]
            
        """
        
        log.info("Applying Kline's skeletonization method.")
        
        # Make sure object is equal to 1, without specifying dtype, would be logical
        B2 = np.array(vol.copy() > 0, dtype='int8')

        # Set defaults
        dmw = kwargs.get('dist_map_weight', 6)
        mbtr = kwargs.get('min_branch_to_root', 10)

        # Find 3D coordinates of volume
        nz_coords = np.nonzero(vol)
        x3, y3, z3 = [nz_coords[i].tolist() for i in range(3)]

        # Limit volume size
        B2 = B2[np.min(x3):np.max(x3) + 1, np.min(y3):np.max(y3) + 1, np.min(z3):np.max(z3) + 1]

        # Setup starting index list and correct for change in volume size
        sx = startID[0] - np.min(x3) 
        sy = startID[1] - np.min(y3) 
        sz = startID[2] - np.min(z3) 

        
        # Perform first fast march to determine endpoints
        phi = B2.copy()
        phi[sx,sy,sz] = -1
        constant_speed = np.ones_like(phi)
        mask = B2 < 1
        phi = np.ma.MaskedArray(phi, mask)
        binary_travel_time = skfmm.travel_time(phi, constant_speed)

        # Fill in masked values and set to zero
        binary_travel_time = binary_travel_time.filled()
        binary_travel_time[binary_travel_time == 1.e20] = 0
  
        # Normalize and apply cluster graph weighting
        # Find endpoints
        hold_binary_travel_time = binary_travel_time.copy()
        endx, endy, endz = self._detect_local_maxima(hold_binary_travel_time)

        # Perform second FMM, to create field for gradient descent
        dMap = morphology.distance_transform_edt(constant_speed)
        weighted_speed = dMap ** dmw
        weighted_travel_time = skfmm.travel_time(phi, weighted_speed)
        weighted_travel_time = weighted_travel_time.filled()

        # Order endpoints by distance from start
        Euc = [np.sqrt((endx[i] - sx)**2 + (endy[i] - sy)**2 + (endz[i] - sz)**2) for i in range(len(endx))]
        order_indici = np.argsort(Euc) # returns indices to sort elements
        Euc = np.sort(Euc)

        X, Y, Z = [], [], []
        # Check whether endpoint is sufficiently far from root voxel (min_branch_to_root)
        for i in range(len(order_indici)):
            if Euc[i] > mbtr:
                X.append(endx[order_indici[i]])
                Y.append(endy[order_indici[i]])
                Z.append(endz[order_indici[i]])

     
        # Implement march back method to build centerline (enlarge volume)
        # The approach proceeds by updating skeleton as equal to 2
        # When branch is finished, the skeleton is solidified and set to 1
        skel = np.zeros((B2.shape[0] + 2, B2.shape[1] + 2, B2.shape[2] + 2), dtype='uint8')
        D = skel.copy() + 1.e20
        D[1:B2.shape[0] + 1, 1:B2.shape[1] + 1, 1:B2.shape[2] + 1] = weighted_travel_time
        skel[sx+1, sy+1, sz+1] = 1 # initialize root

        # Begin extracting skeleton
        for ijk in range(len(X)):

            # Initialize endpoints and correct for larger volume   
            i, j, k = X[ijk] + 1, Y[ijk] + 1, Z[ijk] + 1
            
            # Check whether endpoint in neighborhood of skeleton (whisker)
            if np.all(skel[i-1:i+2, j-1:j+2, k-1:k+2]) != 1: 
                if D[i,j,k] != 1.e20:
                    done_loop = 0               
                    skel[skel > 0] = 1                
                    
                    # Check whether branch is now connected to rest of tree (stopping criteria)
                    while ((i!= sx+1) or (j != sy+1) or (k != sz+1)) and done_loop != 1:
                        skel[i,j,k] = 2               
                        d_neighborhood = D[i-1:i+2, j-1:j+2, k-1:k+2]                    
                        
                        if np.all(skel[i-1:i+2, j-1:j+2, k-1:k+2]) != 1:        
                            currentMin = 1.e21
                            # Find min in neighborhood
                            for ni in range(3):
                                for nj in range(3):
                                    for nk in range(3):
                                        if (d_neighborhood[ni,nj,nk] < currentMin) and (skel[i+ni-1,j+nj-1,k+nk-1] != 2):
                                            ii, jj, kk = ni, nj, nk
                                            currentMin = d_neighborhood[ni, nj, nk]
                            # Update                         
                            i, j, k = i + ii - 1, j + jj - 1, k + kk - 1

                            if D[i,j,k] == 1.e20:
                                done_loop = 1
                                skel[skel == 2] = 0 # remove branch, not marching back to root (local min in weighted_travel_time)
                        else:
                            done_loop = 1
                

        # shift skel and start points back to correspond with original volume
        centerline_extracted = skel[1:B2.shape[0]+1, 1:B2.shape[1]+1, 1:B2.shape[2]+1]
        final_centerline = np.zeros(vol.shape, dtype='uint8')
        final_centerline[np.min(x3):np.max(x3)+1, np.min(y3):np.max(y3)+1, np.min(z3):np.max(z3)+1] = centerline_extracted

        return final_centerline

    def _detect_local_maxima(self, vol):
        """
        Detects the peaks using the local maximum filter.
        
        Args:
            vol (ndarray): Input volume.
        Returns:
            A boolean mask of the peaks (i.e. 1 when the pixel's value is the neighborhood maximum, 0 otherwise)
        """
        # define a 26-connected neighborhood
        neighborhood = morphology.generate_binary_structure(3,3) # first is dimension, next is relative connectivity

        # apply the local maximum filter; all locations of maximum value 
        # in their neighborhood are set to 1
        local_max = (filters.maximum_filter(vol, footprint=neighborhood)==vol)

        # Remove background
        local_max[vol==0] = 0

        # Find endpoint indici
        [xOrig,yOrig,zOrig] = np.shape(vol)
        x = []
        y = []
        z = []
        for i in range(0,xOrig):
            for j in range(0,yOrig):
                for k in range(0,zOrig):
                    if local_max[i,j,k] > 0:
                        x.append(i)
                        y.append(j)
                        z.append(k)

        return x, y, z