import os
import sys
import time
import skfmm
import logging
import argparse
import subprocess
import numpy as np
import manage_data as md
import skimage.morphology as mph
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from typing import Path, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class Skeletonize:
    """
    Wraps multiple 3D skeletonization algorithms.
    """
    
        
    def lee(self, volume: np.ndarray) -> np.ndarray:
        """
        3D thinning via Lee's algorithm.
        
        References:
        .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
           via 3-D medial surface/axis thinning algorithms.
           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.
            
        For full description, refer to https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/_skeletonize.py
        """        
        # alternative function : mph.skeletonize(sample_gr, method='lee')
        logger.info("Running Lee's thinning.")
        return mph.skeletonize_3d(volume)
    
    
    def palagyi(self, volume: np.ndarray, templates_path: Path) -> np.ndarray:
        """ 
        3D thinning via Palágyi's method (12-subiteration).
        -> Based on ClearMap2 (@author: ChristophKirst)

        Requires: 
            "PK12.npy" file, contatinig the 14-templates

        References:
        .. [Palagy1999] Palagyi & Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm,
            Graphical Models and Image Processing 61, 199-221 (1999).
        """
        logger.info("Running Palágyi's thinning.")
        # extract indices vessel points in the input
        points_arg = np.argwhere(volume)
        binary = np.copy(volume)
        deletable = np.load(templates_path)
        
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
            border = self._convolution3d(binary, n6(), points_arg) < 6
            border_points = points_arg[border]
            border_ids = np.nonzero(border)[0]
            keep = np.ones(len(border), dtype=bool)
            
            iterat = 0
            for i in range(12):         # 12 sub-iterations
                removables = self._convolution3d(binary, rotations[i], border_points)
                rem_borders = deletable[removables]
                rem_points = border_points[rem_borders]
                
                binary[rem_points[:, 0], rem_points[:, 1], rem_points[:, 2]] = 0
                keep[border_ids[rem_borders]] = False
                
                iterat += len(rem_points)
            
            # update foreground
            points_arg = points_arg[keep]
            if iterat == 0:
                break
        
        return binary
    
    def _convolution3d(volume: np.ndarray, kernel: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Perform 3D convolution at specific points.

        Args:
            input_data: 3D volume.
            kernel: Convolution kernel.
            point: Points to perform convolution on.
        """
        num_points = points.shape[0]
        output = np.zeros(num_points, dtype=kernel.dtype)
        
        dk, dj, di = volume.shape
        
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
                                    output[n] += volume[zk, yj, xi] * kernel[k, j, i]
        return output
    
    def _rotate(self, volume: np.ndarray, axis: int = 2, steps: int = 0) -> np.ndarray:
        """
        Rotate a cube around an axis in 90-degree steps.

        Args:
            input_data: 3D cube.
            axis: Axis of rotation (0: x, 1: y, 2: z).
            steps: Number of 90-degree steps to rotate.
        """
        cube = volume.copy();  
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

    def _rotations12(self, cube: np.ndarray) -> List[np.ndarray]:
        """
        Generate rotations in 12 diagonal directions.

        Args:
            cube: 3D cube.
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
        
        return [rotUS, rotNE, rotDW,  rotSE, rotUW, rotDN,  rotSW, rotUN, rotDE,  rotNW, rotUE, rotDS]
  
    def kerautret(self, exe_path: Path, off_input: Path, output_path: Path, 
                  dilateDist: Optional(float)=2.0, deltaG: Optional(float)=3.0, radius: Optional(float)=10.0, threshold: Optional(float)=0.5):
        """
        Run external Kerautret binary on OFF mesh.
        The exe file can be obtained by compiling the main Kerautret's project.
        Returns path to generated OBJ.
        -> Based on https://github.com/kerautret/CDCVAM (@author: kerautret)
        
        Args:
            exe_path:           Path to the execution file ("CenterLineGeodesicGraph" function)
            off_input:          Path to the input volume( OFF format)
            output_path:        Output file name
            dilateDist:         Dilate distance of the confidence voxels. 2.0 (default)
            deltaG:             The parameter to consider interval of distances. 3.0  (default)
            radius:             The radius used to compute the accumulation. 10.0  (default)
            threshold:          The threshold in the confidence estimation. 0.5  (default)
            
        References:
        .. [Kerautret2016] B. Kerautret, A. Krähenbühl, I. Debled-Rennesson and J. -O. Lachaud, 
        "Centerline detection on partial mesh scans by confidence vote in accumulation map," 2016, 
        pp. 1376-1381, doi: 10.1109/ICPR.2016.7899829.
        """
        
        logger.info("Running Kerautret's skeletonization.")
        # Remove the previous files if exist
        if os.path.isfile(output_path+'Vertex.sdp'):
            os.remove(output_path+'Vertex.sdp')
            os.remove(output_path+'Edges.sdp')
        command = [
            exe_path,
            '-i', off_input,
            "-o", output_path,
            "--dilateDist", str(dilateDist),
            "-g", str(deltaG),
            "-R", str(radius),
            "-t", str(threshold)
        ]
        # Execute the command
        logger.info("Executing: %s", ' '.join(command))
        subprocess.run(command, check=True)                         # saves three files: Vertex, Edges, and Radius in .SDP format
        
        # Convert to .OBJ and save
        md.sdp2obj(output_path)
        logger.info("Kerautret output at %s", output_path)
            
    def kline(self, volume: np.ndarray, startID: Tuple[int, int, int], dmw: int=6, mbtr: int=10):
        """
        Kline's method.
        -> Based on https://github.com/TLKline/poreture (@author: TLKline)
        
        Args:
            volume: Binarized volume
            startID: Index of root, as (x,y,z) location
        
            dist_map_weight: Weight for distance map. Defaults to 6.
            min_branch_to_root: Minimum branch length to root. Defaults to 10.

        Returns:
            tuple: A binary numpy array containin a 1-D skeleton of the input volume
        Reference:
            kline_vessel - [Kline et al. ABME 2010]
            kline_pore - [Kline et al. J Porous Mat 2011]
            
        """
        
        logger.info("Running Kline's method.")
        
        # Make sure object is equal to 1, without specifying dtype, would be logical
        B2 = np.array(volume.copy() > 0, dtype=np.uint8)

        # Find 3D coordinates of volume
        nz_coords = np.nonzero(volume)
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
        hold_binary_travel_time = binary_travel_time.copy()
        # Find endpoints
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
        final_centerline = np.zeros(volume.shape, dtype=np.uint8)
        final_centerline[np.min(x3):np.max(x3)+1, np.min(y3):np.max(y3)+1, np.min(z3):np.max(z3)+1] = centerline_extracted

        return final_centerline

    def _detect_local_maxima(self, vol: np.ndarray):
        """
        Detects the peaks using the local maximum filter.
        
        Args:
            vol: Input volume.
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
    
    
def main():
    parser = argparse.ArgumentParser("3D Skeleton CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # lee subcommand
    p_lee = sub.add_parser("lee", help="Lee thinning")
    p_lee.add_argument("--input", type=Path, required=True)
    p_lee.add_argument("--output", type=Path, required=True)

    # palagyi subcommand
    p_pal = sub.add_parser("palagyi", help="Palagyi thinning")
    p_pal.add_argument("--input", type=Path, required=True)
    p_pal.add_argument("--templates", type=Path, required=True)
    p_pal.add_argument("--output", type=Path, required=True)

    # kerautret subcommand
    p_ker = sub.add_parser("kerautret", help="External Kerautret method")
    p_ker.add_argument("--input", type=Path, required=True)
    p_ker.add_argument("--output", type=Path, required=True)
    p_ker.add_argument("--exe-path", type=Path, required=True)
    p_ker.add_argument("--params", type=List, default=[2.0, 3.0, 10.0, 0.5])

    # kline subcommand
    p_kl = sub.add_parser("kline", help="Kline centerline extraction method")
    p_kl.add_argument("--input", type=Path, required=True)
    p_kl.add_argument(
        "--root", type=int, nargs=3, required=True,
        help="X Y Z index of starting point"
    )
    p_kl.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    ske = Skeletonize()

    if args.cmd == "lee":
        vol = np.load(args.input)
        out = ske.lee(vol)
        np.save(args.output, out)

    elif args.cmd == "palagyi":
        vol = np.load(args.input)
        out = ske.palagyi(vol, args.templates)
        np.save(args.output, out)

    elif args.cmd == "kerautret":
        ske.kerautret(
            args.input,
            args.output,
            args.exe_path,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3]
        )

    elif args.cmd == "kline":
        vol = np.load(args.input)
        _ = ske.kline(vol, tuple(args.root))

    else:
        parser.error(f"Unknown command {args.cmd}")

    logger.info("Done (%s)", args.cmd)


if __name__ == "__main__":
    main()