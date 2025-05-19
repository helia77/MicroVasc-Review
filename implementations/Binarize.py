import gc
import math
import logging
import argparse
import numpy as np
from pathlib import Path
import scipy.ndimage.filters as filters
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class Binarizer:
    """
    A class to binarize 3D volumes using Otsu's thresholding, Frangi and Beyond Frangi filter.
    
    Args:
        volume: 3D grayscale volume.
        background: 'black' (default) or 'white'.
    """
    
    def __init__(self, volume: np.ndarray, background: str = 'black'):
        if background not in ("black", "white"):
            raise ValueError("background must be 'black' or 'white'")
        self.volume = volume
        self.background = background
        logger.info("Binarizer initialized (background=%s)", background)
        
        
    def _compute_threshold(self, data: Optional[np.ndarray] = None) -> float:
        """
        Compute Otsu's threshold over the 3D histogram.
        """
        arr = data if data is not None else self.volume
        flat = arr.ravel()
        if flat.size == 0:
            raise ValueError("Empty volume")
        hist, edges = np.histogram(flat, bins=256, range=(flat.min(), flat.max()))
        prob = hist.astype(float) / hist.sum()
        centers = (edges[:-1] + edges[1:]) / 2.0

        w0 = np.cumsum(prob)
        w1 = np.cumsum(prob[::-1])[::-1]
        m0 = np.cumsum(prob * centers) / np.maximum(w0, 1e-8)
        m1 = (np.cumsum((prob * centers)[::-1]) / np.maximum(w1[::-1], 1e-8))[::-1]

        sigma_b = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]) ** 2
        thr = centers[:-1][np.argmax(sigma_b)]
        return thr

    def otsu3d(self) -> np.ndarray:
        """
        3D Otsu's thresholding.
        """   
        thr = self._compute_threshold()
        logger.info("Otsu threshold: %.3f", thr)

        if self.background == 'black':
            mask = (self.volume >= thr)
        elif self.background == 'white':
            mask = (self.volume <= thr)
            
        logger.info("Completed 3D Otsu binarization.")
        return mask.astype(np.uint8)
    
    def otsu2d(self) -> np.ndarray:
        """
        Slice-by-slice Otsu thresholding.
        """
        out = []
        vol = self.volume
        for i in range(vol.shape[0]):
            slice_ = vol[i]
            t = self._compute_threshold(data=slice_)
            if self.background == "black":
                mask = slice_ >= t
            else:
                mask = slice_ <= t
            out.append(mask.astype(np.uint8))
        result = np.stack(out, axis=0)
        logger.info("Completed 2D Otsu binarization.")
        return result.astype(np.uint8)
    
    def _hessian(self, scale: float):
        """
        Compute the Hessian mask at a given scale.

        Args:
            scale: Sigma for Gaussian derivatives.
        """
        X, Y, Z = self.volume.shape
        Hxx = np.zeros((X, Y, Z))
        Hyy = np.zeros((X, Y, Z))
        Hzz = np.zeros((X, Y, Z))
        Hxy = np.zeros((X, Y, Z))
        Hxz = np.zeros((X, Y, Z))
        Hzy = np.zeros((X, Y, Z))
        
        # convolving image with Gaussian derivatives - including Hxx, Hxy, Hyy
        filters.gaussian_filter(self.volume, (scale, scale, scale), (0, 0, 2), Hxx)
        filters.gaussian_filter(self.volume, (scale, scale, scale), (0, 1, 1), Hxy)
        filters.gaussian_filter(self.volume, (scale, scale, scale), (0, 2, 0), Hyy)
        filters.gaussian_filter(self.volume, (scale, scale, scale), (2, 0, 0), Hzz)
        filters.gaussian_filter(self.volume, (scale, scale, scale), (1, 0, 1), Hxz)
        filters.gaussian_filter(self.volume, (scale, scale, scale), (1, 1, 0), Hzy)
        
        # scale normalization (gamma factor = 2)
        s2 = scale * scale
        Hxx *= s2; Hyy *= s2; Hzz *= s2
        Hxy *= s2; Hxz *= s2; Hzy *= s2
    
        # reduce computation by computing vesselness only where is needed
        # based on the paper of S.-F. Yang and C.-H. Cheng, “Fast computation of Hessian-based enhancement filters for medical images”
        B1 = -(Hxx + Hyy + Hzz)
        B2 = (Hxx * Hyy) + (Hxx * Hzz) + (Hyy * Hzz) - (Hxy * Hxy) - (Hxz * Hxz) - (Hzy * Hzy)
        B3 = (Hxx * Hzy * Hzy) + (Hxy * Hxy * Hzz) + (Hxz * Hyy * Hxz) - (Hxx * Hyy * Hzz) - (Hxy * Hzy * Hxz) - (Hxz * Hxy * Hzy)
        
        T = np.ones_like(B1, dtype=np.uint8)   
        if self.background == 'black':
            T[B1 <= 0] = 0
            T[(B2 <= 0) & (B3 == 0)] = 0
            T[(B1 > 0) & (B3 > 0) & (B1*B2 < B3)] = 0
        else:
            T[B1 >= 0] = 0
            T[(B2 >= 0) & (B3 == 0)] = 0
            T[(B1 < 0) & (B2 < 0) & ((-B1)*(-B2) < (-B3))] = 0
        
        # free memory
        del B1, B2, B3
        gc.collect()
        
        Hxx *= T; Hyy *= T; Hzz *= T
        Hxy *= T; Hxz *= T; Hzy *= T
        
        H = np.zeros((X, Y, Z, 3, 3))
        H[:, :, :, 2, 2] = Hxx;     H[:, :, :, 1, 1] = Hyy;     H[:, :, :, 0, 0] = Hzz;
        H[:, :, :, 1, 2] = Hxy;     H[:, :, :, 0, 2] = Hxz;     H[:, :, :, 0, 1] = Hzy;
        H[:, :, :, 2, 1] = Hxy;     H[:, :, :, 2, 0] = Hxz;     H[:, :, :, 1, 0] = Hzy;
        
        # free memory
        del Hxx, Hyy, Hzz, Hxy, Hxz, Hzy
        gc.collect()
        
        return H, T
    
    def frangi(self, alpha: float, beta: float, c: float, scale_range: List[float], threshold: Optional[str] = None) -> np.ndarray:
        """
        Frangi (aka. vesselness) filter for 3D tubular enhancement.

        Args:
            alpha, beta, c: Frangi parameters.
            scale_range: List of scales for scale space.
            threshold: None to filter, 'otsu2d' or 'otsu3d' to binarize the volume.
        """         
        logger.info("Running Frangi filter on %d scales.", len(scale_range))
        all_filters = []
        alpha2 = 2 * (alpha**2) if alpha != 0 else math.nextafter(0, 1)
        beta2  = 2 * (beta**2) if beta != 0 else math.nextafter(0, 1)
        c2     = 2 * (c**2) if c != 0 else math.nextafter(0, 1)
        X, Y, Z = self.volume.shape

        for scale in scale_range:
            H, T = self._hessian(scale)
            # eigendecomposition
            lambdas = np.linalg.eigvalsh(H)
            idx = np.argwhere(T == 1)
            V0 = np.zeros_like(self.volume, dtype=np.float64)
                
            for arg in idx:
                i, j, k = arg
                l1, l2, l3 = sorted(lambdas[i, j, k], key=abs)
                if self.background == 'white' and (l2 < 0 or l3 < 0):
                    continue
                elif self.background == 'black' and (l2 > 0 or l3 > 0):
                    continue
                
                if (l3 == 0):
                    l3 = math.nextafter(0,1)
                if (l2 == 0):
                    l2 = math.nextafter(0,1)
                
                Rb2 = (l1**2)/(l2 * l3)
                Ra2 = (l2**2) / (l3**2)
                S2 = (l1**2) + (l2**2) + (l3**2)
                    
                term1 = math.exp(-Ra2 / alpha2)
                term2 = math.exp(-Rb2 / beta2)
                term3 = math.exp(-S2 / c2)
                V0[i, j, k] = (1.0 - term1) * (term2) * (1.0 - term3)
                
            all_filters.append(V0)
        
        # stack filters and compute maximum vesselness
        stacked_filters = np.stack(all_filters, axis=-1)
        vesselness = np.max(stacked_filters, axis=-1)
        self.background = 'black'

        if threshold == 'otsu2d':
            self.volume = vesselness
            return self.otsu2d()
        if threshold == 'otsu3d':
            self.volume = vesselness
            return self.otsu3d()
        
        logger.info("Frangi filter completed.")
        return vesselness

    def bfrangi(self, tau: float, scale_range: List[float], threshold: Optional[str] = None) -> np.ndarray:
        """
        Beyond Frangi filter for 3D tubular enhancement.

        Args:
            tau: Beyond Frangi parameter (regularization).
            scale_range: List of scales for scale space.
            threshold: None to filter, 'otsu2d' or 'otsu3d' to binarize the volume.
        """
        logger.info("Running Beyond Frangi filter on %d scales.", len(scale_range))
        all_filters = []
        X, Y, Z = self.volume.shape
        
        for scale in scale_range:
            H, T = self._hessian(scale)
            
            # eigendecomposition
            lambdas = np.linalg.eigvalsh(H)
            idx = np.argwhere(T == 1)
            V0 = np.zeros_like(self.volume, dtype=np.float64)
            for arg in idx:
                # sort the eigenvalues
                i, j, k = arg
                lambdas[i, j, k] = sorted(lambdas[i, j, k], key=abs)
            
            # find the maximum lambda3 across the volume with scale s
            max_l3 = np.max(abs(lambdas[:, :, :, 2]))
            for arg in idx:
                i, j, k = arg
                _, l2, l3 = lambdas[i, j, k]        # no need for lambda1
                
                if self.background == 'black':
                    l2 = -l2
                    l3 = -l3
                
                # calculate lambda rho
                reg_term = tau * max_l3             # regularized term
                l_rho = l3
                if l3 > 0 and l3 < reg_term:
                    l_rho = reg_term
                elif l3 <= 0:
                    l_rho = 0
                    
                # modified vesselness (Beyond Frangi) function
                V0[i, j, k] = (l2**2) * (l_rho - l2) * 27 / ((l2 + l_rho) ** 3)
                if l2 >= (l_rho/2) and l_rho > 0:
                    V0[i, j, k] = 1
                elif l2 <= 0 or l_rho <= 0:
                    V0[i, j, k] = 0
                
            all_filters.append(V0)
        
        # stack filters and compute maximum vesselness
        stacked_filters = np.stack(all_filters, axis=-1)
        output = np.max(stacked_filters, axis=-1)
        self.background = 'black'

        if threshold == 'otsu2d':
            self.volume = output
            return self.otsu2d()
        if threshold == 'otsu3d':
            self.volume = output
            return self.otsu3d()
        
        logger.info("Beyond Frangi filter completed.")
        return output

def parse_args() -> argparse.Namespace:
    parent = argparse.ArgumentParser(add_help=False, description='3D Volume Filtering and Binarization')
    parent.add_argument("--input", required=True, type=Path, help="Path to input .npy volume")
    parent.add_argument("--output", required=True, type=Path, help="Path for output .npy mask or binarization.")
    parent.add_argument("--background", choices=["black", "white"], default="black")
    
    parser = argparse.ArgumentParser(description="Binarization or filter method to apply.")
    sub = parser.add_subparsers(dest='method', required=True)

    # thresholding methods
    sub.add_parser('otsu2d', parents=[parent], help="2D Otsu")
    sub.add_parser('otsu3d', parents=[parent], help="3D Otsu")

    # vesselness filters
    vessel = argparse.ArgumentParser(add_help=False)
    vessel.add_argument('--scale', nargs='+', type=float, required=True)
    vessel.add_argument('--th', choices=["otus2d", "otsu3d"], type=str, default=None, help="Optional thresholding technique.")

    fr = sub.add_parser('frangi', parents=[parent, vessel], help="Frangi Filter")
    fr.add_argument('--params', nargs=3, type=float, metavar=('alpha', 'beta', 'c'), required=True)
    
    bf = sub.add_parser('bfrangi', parents=[parent, vessel], help="Beyond Frangi Filter")
    bf.add_argument('--params', nargs=1, type=float, metavar='tau', required=True)

    return parser.parse_args()

def main():
    """
    Binarize.py
    
    A command-line tool and Python module for binarizing 3D volumes using:
      - Otsu’s thresholding (2D slices or full 3D)
      - Frangi vesselness filter
      - Beyond-Frangi filter
    
    Usage (CLI):
        python binarize.py frangi --input in.npy --output out.npy --background white \
        --params 1 0.5 3 --scale 1 2 3 4 5
        or
        python binarize.py otsu3d --input in.npy --output out.npy --background white

    """
    args = parse_args()
    vol = np.load(args.input)
    binzr = Binarizer(vol, background=args.background)

    if args.method in ('otsu2d', 'otsu3d'):
        out = getattr(binzr, args.method)()
    elif args.method == 'frangi':
        a,b,c = args.params
        scale_range = args.scale
        out = binzr.frangi(alpha=a,  beta=b, c=c, scale_range=scale_range, threshold=args.th)
    else: # bfrangi
        scale_range = args.scale
        out = binzr.bfrangi(tau=args.params[0], scale_range=scale_range, threshold=args.th)
    
    np.save(args.output, out)
    logger.info("Saved output to %s", args.output)


if __name__ == "__main__":
    main()