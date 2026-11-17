import os
import itertools
import logging
import argparse
import subprocess
import numpy as np
import scipy.ndimage as ndi
from microvasc import data as md
from pathlib import Path
import skimage.morphology as mph
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class Skeletonizer:
    """
    Skeletonize binarized 3D volumes with Lee, Palagyi, Kline or Kerautret.

    Attributes:
        DEFAULT_TEMPLATES: Path of the Palagyi lookup table installed with the package.

    Examples:
        >>> import numpy as np
        >>> from microvasc.skeleton import Skeletonizer
        >>> volume = np.zeros((7, 7, 15), dtype=np.uint8)
        >>> volume[2:5, 2:5, 2:13] = 1
        >>> skeleton = Skeletonizer().lee(volume)
        >>> skeleton.dtype, int(skeleton.sum())
        (dtype('uint8'), 9)
    """
    # Palagyi lookup table shipped with the package
    DEFAULT_TEMPLATES = Path(__file__).with_name("PK12.npz")
    # Palagyi: the 12 deletion directions in the paper's order, and the six major directions
    # (axis 0 east, axis 1 north, axis 2 up)
    _DIRECTIONS = ("US", "NE", "WD", "ES", "UW", "ND", "SW", "UN", "ED", "NW", "UE", "SD")
    _UNIT = {"E": (1, 0, 0), "W": (-1, 0, 0), "N": (0, 1, 0), "S": (0, -1, 0), "U": (0, 0, 1), "D": (0, 0, -1)}
    
        
    def lee(self, volume: np.ndarray) -> np.ndarray:
        """
        Lee thinning, with skimage.morphology.skeletonize(volume, method='lee').

        Args:
            volume: Binarized 3D volume. Any nonzero voxel is foreground.

        Returns:
            uint8 array of 0 and 1, same shape as the input.

        References:
            T.-C. Lee, R. L. Kashyap and C.-N. Chu, Building skeleton models via 3-D medial
            surface/axis thinning algorithms. Computer Vision, Graphics, and Image Processing
            56(6), 462-478, 1994.
        """
        logger.info("Running Lee's thinning.")
        return mph.skeletonize(volume, method='lee').astype(np.uint8)
    
    
    def palagyi(self, volume: np.ndarray, templates_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Palagyi and Kuba 12-subiteration curve thinning.

        Axis 0 points east, axis 1 north and axis 2 up. Each iteration runs 12 subiterations
        in the order US, NE, WD, ES, UW, ND, SW, UN, ED, NW, UE, SD and deletes all matching
        points of a subiteration together. Voxels outside the volume count as white.
        Border points are found once per iteration and reused in all 12 subiterations. The
        paper checks every point again in each subiteration.

        Args:
            volume: Binarized 3D volume. Any nonzero voxel is foreground.
            templates_path: Lookup table of the deletion templates T1 to T14 (direction US),
                as a .npz file with packed bits under "pk12" or a bool .npy file with 2^26
                entries. None (default) uses DEFAULT_TEMPLATES. tests/build_pk12_table.py
                rebuilds the table from the templates of the paper.

        Returns:
            uint8 array of 0 and 1, same shape as the input.

        Raises:
            ValueError: If the lookup table does not have 2^26 entries.

        References:
            K. Palagyi and A. Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm.
            Graphical Models and Image Processing 61, 199-221, 1999.
        """
        table = self._load_table(self.DEFAULT_TEMPLATES if templates_path is None else templates_path)
        logger.info("Running Palagyi's thinning.")

        padded = np.pad(volume > 0, 1).astype(np.uint8)
        x = padded.reshape(-1)                                      # flat view of padded
        strides = np.array(padded.strides) // padded.itemsize
        six = np.concatenate([strides, -strides])
        offsets = [self._direction_offsets(d, strides) for d in self._DIRECTIONS]
        weights = 1 << np.arange(26, dtype=np.int64)

        while True:
            points = np.flatnonzero(x)
            border = points[(x[points[:, None] + six] == 0).any(axis=1)]
            deleted = 0
            for flat in offsets:
                border = border[x[border] == 1]                     # skip points deleted earlier
                index = x[border[:, None] + flat] @ weights
                remove = border[table[index]]
                x[remove] = 0
                deleted += remove.size
            if deleted == 0:
                break

        return padded[1:-1, 1:-1, 1:-1].copy()

    def _load_table(self, path: Path) -> np.ndarray:
        """
        Load the Palagyi lookup table as a bool array of 2^26 entries.

        Args:
            path: .npz file with the table as packed bits under "pk12" (np.packbits order),
                or .npy file with the bool table.
        """
        if Path(path).suffix == ".npz":
            with np.load(path) as data:
                table = np.unpackbits(data["pk12"], count=2 ** 26).astype(bool)
        else:
            table = np.load(path).astype(bool)
        if table.shape != (2 ** 26,):
            raise ValueError(f"The lookup table must have 2^26 entries, got shape {table.shape}")
        return table

    def _direction_offsets(self, direction: str, strides: np.ndarray) -> np.ndarray:
        """
        Flat offsets of the 26 neighbors, in lookup table bit order, for one deletion direction.

        The table holds the templates of direction US. Bit k is the k-th neighbor (u, n, e) in
        C order over {-1, 0, 1}^3, center skipped. For direction XY, the rotation that sends U
        to X and S to Y gives each (u, n, e) its offset in the volume.

        Args:
            direction: Two letters from U, D, N, S, E, W, e.g. "NE".
            strides: Strides of the padded volume, in elements.
        """
        up = np.array(self._UNIT[direction[0]])
        north = -np.array(self._UNIT[direction[1]])
        east = np.cross(north, up)
        neighbors = [u * up + n * north + e * east
                     for u, n, e in itertools.product((-1, 0, 1), repeat=3) if (u, n, e) != (0, 0, 0)]
        return np.array(neighbors) @ strides
  
    def kerautret(self, exe_path: Union[str, Path], off_path: Union[str, Path], output_prefix: Union[str, Path],
                  dilate_dist: float=2.0, delta_g: float=3.0, radius: float=10.0, threshold: float=0.5) -> Path:
        """
        Run the external CDCVAM program centerLineGeodesicGraph on a surface mesh.

        CDCVAM (https://github.com/kerautret/CDCVAM) must be built separately. Old .sdp files
        of the same output prefix are removed before the run.

        Args:
            exe_path: Path to the centerLineGeodesicGraph executable.
            off_path: Path to the input surface mesh in OFF format.
            output_prefix: Output prefix. Writes <prefix>Vertex.sdp, <prefix>Edges.sdp,
                <prefix>Radius.sdp and <prefix>.obj.
            dilate_dist: Dilate distance of the confidence voxels.
            delta_g: Interval of distances considered.
            radius: Radius used to compute the accumulation.
            threshold: Threshold in the confidence estimation.

        Returns:
            Path of the OBJ file, <prefix>.obj.

        Raises:
            FileNotFoundError: If the executable is missing or not executable, if the mesh is
                missing, or if the program did not write <prefix>Vertex.sdp and <prefix>Edges.sdp.
            subprocess.CalledProcessError: If the program exits with an error.

        References:
            B. Kerautret, A. Krähenbühl, I. Debled-Rennesson and J.-O. Lachaud, Centerline
            detection on partial mesh scans by confidence vote in accumulation map. ICPR 2016,
            1376-1381, doi:10.1109/ICPR.2016.7899829.
        """
        exe_path, off_path, output_prefix = Path(exe_path), Path(off_path), Path(output_prefix)
        if not exe_path.is_file() or not os.access(exe_path, os.X_OK):
            raise FileNotFoundError(
                f"Kerautret executable not found or not executable: {exe_path}. "
                "Get CDCVAM from https://github.com/kerautret/CDCVAM and build it with CMake "
                "(needs DGtal): mkdir build && cd build && cmake .. && make. "
                "The executable is build/bin/centerLineGeodesicGraph.")
        if not off_path.is_file():
            raise FileNotFoundError(f"Input mesh not found: {off_path}")

        logger.info("Running Kerautret's skeletonization.")
        # Remove previous results for this output prefix only
        for suffix in ("Vertex.sdp", "Edges.sdp", "Radius.sdp"):
            old = output_prefix.with_name(output_prefix.name + suffix)
            if old.is_file():
                old.unlink()
        command = [
            str(exe_path),
            "-i", str(off_path),
            "-o", str(output_prefix),
            "--dilateDist", str(dilate_dist),
            "-g", str(delta_g),
            "-R", str(radius),
            "-t", str(threshold)
        ]
        # Execute the command
        logger.info("Executing: %s", ' '.join(command))
        subprocess.run(command, check=True)                         # saves three files: Vertex, Edges, and Radius in .SDP format

        # Convert to .OBJ and save
        obj_path = Path(md.sdp2obj(output_prefix))
        logger.info("Kerautret output at %s", obj_path)
        return obj_path
            
    def kline(self, volume: np.ndarray, root: Tuple[int, int, int], dist_map_weight: float=6, min_branch_to_root: float=10) -> np.ndarray:
        """
        Kline's fast marching centerline extraction from a single root. Needs scikit-fmm.

        This follows the code used for the paper, which is based on poreture
        (https://github.com/TLKline/poreture, BSD 3-clause license, see
        THIRD_PARTY_NOTICES.md). It differs from poreture's kline_vessel in two ways. Both
        fast marches use an all-ones speed array over the bounding box (np.ones_like(phi)),
        and the distance map is computed from it; poreture uses the vessel mask
        (constant_speed = B2.copy()). poreture reads cluster_graph_weight and
        min_branch_length but does not use them, so they are not parameters here.

        With a large dist_map_weight, such as 19.25, the result depends on floating point
        rounding and can differ between machines.

        Args:
            volume: Binarized 3D volume. Any nonzero voxel is foreground.
            root: Index of the root voxel along axes 0, 1 and 2. It must lie on a vessel voxel.
            dist_map_weight: Exponent of the distance map, used as the speed of the second
                fast march.
            min_branch_to_root: Minimum Euclidean distance, in voxels, from an endpoint to the
                root. Endpoints at this distance or closer are not traced.

        Returns:
            uint8 array of 0 and 1, same shape as the input, with the centerline.

        Raises:
            ValueError: If the volume is not 3D, root does not have 3 indices, or root is
                outside the volume or not on a vessel voxel.
            ImportError: If scikit-fmm is not installed.

        References:
            Kline et al., Annals of Biomedical Engineering, 2010 (kline_vessel).
            Kline et al., Journal of Porous Materials, 2011 (kline_pore).
        """
        root = tuple(int(c) for c in root)
        if volume.ndim != 3 or len(root) != 3:
            raise ValueError(f"kline needs a 3D volume and a root with 3 indices, got shape {volume.shape} and root {root}")
        if any(c < 0 or c >= n for c, n in zip(root, volume.shape)):
            raise ValueError(f"Root {root} is outside the volume of shape {volume.shape}. "
                             "Give array indices along axes 0, 1 and 2.")
        if not volume[root] > 0:
            raise ValueError(f"Root {root} is not on a vessel voxel (value {volume[root]}).")

        try:
            import skfmm
        except ImportError as err:
            raise ImportError("kline requires scikit-fmm. Install it with: pip install scikit-fmm") from err

        logger.info("Running Kline's method.")

        # Make sure object is equal to 1 (signed, so the root can be marked with -1)
        B2 = np.array(volume.copy() > 0, dtype=np.int8)

        # Find 3D coordinates of volume
        nz_coords = np.nonzero(B2)
        x3, y3, z3 = [nz_coords[i].tolist() for i in range(3)]

        # Limit volume size
        B2 = B2[np.min(x3):np.max(x3) + 1, np.min(y3):np.max(y3) + 1, np.min(z3):np.max(z3) + 1]

        # Setup starting index list and correct for change in volume size
        sx = root[0] - np.min(x3)
        sy = root[1] - np.min(y3)
        sz = root[2] - np.min(z3)

        # Perform first fast march to determine endpoints
        # works on binary speed function
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
        dMap = ndi.distance_transform_edt(constant_speed)
        weighted_speed = dMap ** dist_map_weight
        weighted_travel_time = skfmm.travel_time(phi, weighted_speed)
        weighted_travel_time = weighted_travel_time.filled()

        # Order endpoints by distance from start
        Euc = [np.sqrt((endx[i] - sx)**2 + (endy[i] - sy)**2 + (endz[i] - sz)**2) for i in range(len(endx))]
        order_indici = np.argsort(Euc) # returns indices to sort elements
        Euc = np.sort(Euc)

        X, Y, Z = [], [], []
        # Check whether endpoint is sufficiently far from root voxel (min_branch_to_root)
        for i in range(len(order_indici)):
            if Euc[i] > min_branch_to_root:
                X.append(endx[order_indici[i]])
                Y.append(endy[order_indici[i]])
                Z.append(endz[order_indici[i]])

        # Implement march back method to build centerline (enlarge volume)
        # The approach proceeds by updating skeleton as equal to 2
        # When branch is finished, the skeleton is solidified and set to 1
        skel = np.zeros((B2.shape[0] + 2, B2.shape[1] + 2, B2.shape[2] + 2), dtype='uint8')
        D = skel.copy() + 1.e20
        D[1:B2.shape[0] + 1, 1:B2.shape[1] + 1, 1:B2.shape[2] + 1] = weighted_travel_time
        D[sx+1, sy+1, sz+1] = 0
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
                                        if (d_neighborhood[ni,nj,nk] < currentMin) and (np.all([ni,nj,nk]) != 1) and (skel[i+ni-1,j+nj-1,k+nk-1] != 2):
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
        skel[skel == 2] = 1
        centerline_extracted = skel[1:B2.shape[0]+1, 1:B2.shape[1]+1, 1:B2.shape[2]+1]
        final_centerline = np.zeros(volume.shape, dtype=np.uint8)
        final_centerline[np.min(x3):np.max(x3)+1, np.min(y3):np.max(y3)+1, np.min(z3):np.max(z3)+1] = centerline_extracted

        return final_centerline

    def _detect_local_maxima(self, vol: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        """
        Find the nonzero voxels that are the maximum of their 26-neighborhood.

        Args:
            vol: Input volume.

        Returns:
            The indices of the peaks along axes 0, 1 and 2, as three lists in C order.
        """
        # define a 26-connected neighborhood
        neighborhood = ndi.generate_binary_structure(3,3) # first is dimension, next is relative connectivity

        # apply the local maximum filter; all locations of maximum value
        # in their neighborhood are set to 1
        local_max = (ndi.maximum_filter(vol, footprint=neighborhood)==vol)

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
    
    
def main() -> None:
    """
    Run the skeleton command: load the input, apply the method, save the result.

    Subcommands are lee, palagyi, kline and kerautret. For example:

        skeleton lee --input binary.npy --output skeleton.npy
        skeleton kline --input binary.npy --output centerline.npy --root 95 76 35
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Skeletonization method to apply.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # lee subcommand
    p_lee = sub.add_parser("lee", help="Lee thinning")
    p_lee.add_argument("--input", type=Path, required=True)
    p_lee.add_argument("--output", type=Path, required=True)

    # palagyi subcommand
    p_pal = sub.add_parser("palagyi", help="Palagyi thinning")
    p_pal.add_argument("--input", type=Path, required=True)
    p_pal.add_argument("--templates", type=Path, default=None,
                       help="PK12.npz lookup table (default: the copy bundled with the package)")
    p_pal.add_argument("--output", type=Path, required=True)

    # kerautret subcommand
    p_ker = sub.add_parser("kerautret", help="External Kerautret method")
    p_ker.add_argument("--input", type=Path, required=True)
    p_ker.add_argument("--output", type=Path, required=True)
    p_ker.add_argument("--exe-path", type=Path, required=True)
    p_ker.add_argument("--dilate-dist", type=float, default=2.0,
                       help="Dilate distance of the confidence voxels (default: 2.0)")
    p_ker.add_argument("--delta-g", type=float, default=3.0,
                       help="Interval of distances considered (default: 3.0)")
    p_ker.add_argument("--radius", type=float, default=10.0,
                       help="Radius used to compute the accumulation (default: 10.0)")
    p_ker.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold in the confidence estimation (default: 0.5)")

    # kline subcommand
    p_kl = sub.add_parser("kline", help="Kline centerline extraction method")
    p_kl.add_argument("--input", type=Path, required=True)
    p_kl.add_argument(
        "--root", type=int, nargs=3, required=True, metavar=("I0", "I1", "I2"),
        help="Index of the root voxel along array axes 0, 1 and 2 (must be a vessel voxel)"
    )
    p_kl.add_argument("--output", type=Path, required=True)
    p_kl.add_argument("--dist-map-weight", type=float, default=6,
                      help="Distance map weight (default: 6)")
    p_kl.add_argument("--min-branch-to-root", type=float, default=10,
                      help="Minimum distance from an endpoint to the root, in voxels (default: 10)")

    args = parser.parse_args()
    ske = Skeletonizer()

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
            exe_path=args.exe_path,
            off_path=args.input,
            output_prefix=args.output,
            dilate_dist=args.dilate_dist,
            delta_g=args.delta_g,
            radius=args.radius,
            threshold=args.threshold
        )

    elif args.cmd == "kline":
        vol = np.load(args.input)
        out = ske.kline(vol, tuple(args.root), dist_map_weight=args.dist_map_weight,
                        min_branch_to_root=args.min_branch_to_root)
        np.save(args.output, out)

    else:
        parser.error(f"Unknown command {args.cmd}")

    logger.info("Done (%s)", args.cmd)


if __name__ == "__main__":
    main()