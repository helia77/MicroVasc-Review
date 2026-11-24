import gc
import math
import numbers
import logging
import argparse
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
from typing import List, Optional, Sequence, Tuple, Union

from microvasc._oof import oof_response

logger = logging.getLogger(__name__)

class Binarizer:
    """
    Binarize a 3D volume with Otsu thresholding, Frangi, Beyond Frangi or optimally oriented
    flux (OOF).

    Args:
        volume: 3D grayscale volume.
        background: 'black' (default) for bright vessels on a dark background, 'white' for
            dark vessels on a bright background.

    Raises:
        ValueError: If background is not 'black' or 'white'.

    Examples:
        >>> import numpy as np
        >>> from microvasc.binarize import Binarizer
        >>> volume = np.zeros((5, 5, 5))
        >>> volume[1:4, 1:4, 1:4] = 100
        >>> mask = Binarizer(volume).otsu3d()
        >>> mask.dtype, int(mask.sum())
        (dtype('uint8'), 27)
    """

    def __init__(self, volume: np.ndarray, background: str = 'black') -> None:
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

    def _otsu3d(self, vol: np.ndarray, background: str) -> np.ndarray:
        thr = self._compute_threshold(data=vol)
        logger.info("Otsu threshold: %.3f", thr)

        if background == 'black':
            mask = (vol >= thr)
        elif background == 'white':
            mask = (vol <= thr)

        logger.info("Completed 3D Otsu binarization.")
        return mask.astype(np.uint8)

    def _otsu2d(self, vol: np.ndarray, background: str) -> np.ndarray:
        out = []
        for i in range(vol.shape[0]):
            slice_ = vol[i]
            t = self._compute_threshold(data=slice_)
            if background == "black":
                mask = slice_ >= t
            else:
                mask = slice_ <= t
            out.append(mask.astype(np.uint8))
        result = np.stack(out, axis=0)
        logger.info("Completed 2D Otsu binarization.")
        return result.astype(np.uint8)

    def otsu3d(self) -> np.ndarray:
        """
        Threshold the whole volume with Otsu's method.

        The threshold comes from a 256-bin histogram of all voxels. Voxels at or above it are
        foreground for a black background, at or below it for a white background.

        Returns:
            uint8 array of 0 and 1, same shape as the volume.

        Raises:
            ValueError: If the volume is empty.
        """
        return self._otsu3d(self.volume, self.background)

    def otsu2d(self) -> np.ndarray:
        """
        Threshold each slice along axis 0 with its own Otsu threshold.

        Each threshold comes from a 256-bin histogram of that slice. Foreground is chosen as
        in `otsu3d`.

        Returns:
            uint8 array of 0 and 1, same shape as the volume.

        Raises:
            ValueError: If the volume is empty.
        """
        return self._otsu2d(self.volume, self.background)

    def _check_threshold(self, threshold: Optional[str]) -> None:
        if threshold not in (None, 'otsu2d', 'otsu3d'):
            raise ValueError(f"threshold must be None, 'otsu2d' or 'otsu3d', got {threshold!r}")

    def _threshold_vesselness(self, vesselness: np.ndarray, threshold: Optional[str]) -> Optional[np.ndarray]:
        """
        Binarize a vesselness map with Otsu. Vesselness is bright on black, whatever the input background.
        Returns None when no threshold is requested.
        """
        if threshold == 'otsu2d':
            return self._otsu2d(vesselness, 'black')
        if threshold == 'otsu3d':
            return self._otsu3d(vesselness, 'black')
        return None

    def _hessian(self, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the scale-normalized Hessian at a given scale.

        Args:
            scale: Sigma for Gaussian derivatives.

        Returns:
            The Hessian, shape volume.shape + (3, 3), and the uint8 mask of the voxels where
            vesselness is computed (Yang and Cheng). The Hessian is zero outside the mask.
        """
        X, Y, Z = self.volume.shape
        Hxx = np.zeros((X, Y, Z))
        Hyy = np.zeros((X, Y, Z))
        Hzz = np.zeros((X, Y, Z))
        Hxy = np.zeros((X, Y, Z))
        Hxz = np.zeros((X, Y, Z))
        Hzy = np.zeros((X, Y, Z))
        
        # convolving image with Gaussian derivatives - including Hxx, Hxy, Hyy
        ndi.gaussian_filter(self.volume, (scale, scale, scale), (0, 0, 2), Hxx)
        ndi.gaussian_filter(self.volume, (scale, scale, scale), (0, 1, 1), Hxy)
        ndi.gaussian_filter(self.volume, (scale, scale, scale), (0, 2, 0), Hyy)
        ndi.gaussian_filter(self.volume, (scale, scale, scale), (2, 0, 0), Hzz)
        ndi.gaussian_filter(self.volume, (scale, scale, scale), (1, 0, 1), Hxz)
        ndi.gaussian_filter(self.volume, (scale, scale, scale), (1, 1, 0), Hzy)
        
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
    
    def _sort_by_magnitude(self, lambdas: np.ndarray) -> np.ndarray:
        """
        Sort eigenvalues (n, 3) by |lambda| along the last axis. The sort is stable,
        like sorted(key=abs), so equal magnitudes keep the order of np.linalg.eigvalsh.
        """
        order = np.argsort(np.abs(lambdas), axis=-1, kind="stable")
        return np.take_along_axis(lambdas, order, axis=-1)

    def frangi(self, alpha: float, beta: float, c: float, scales: List[float], threshold: Optional[str] = None) -> np.ndarray:
        """
        Frangi vesselness filter.

        Eigenvalues are sorted so that |lambda1| <= |lambda2| <= |lambda3|. The Hessian is
        scale-normalized with gamma = 2. Voxels outside the Yang and Cheng mask get 0.

        Args:
            alpha: Weight of the ratio R_A = |lambda2| / |lambda3|.
            beta: Weight of the ratio R_B = |lambda1| / sqrt(|lambda2 * lambda3|).
            c: Weight of the structure term S = sqrt(lambda1^2 + lambda2^2 + lambda3^2).
                For alpha, beta and c, 0 is replaced by the smallest positive float.
            scales: Gaussian sigmas in voxels. The response is the maximum over the scales.
            threshold: None (default) returns the vesselness. 'otsu2d' or 'otsu3d' thresholds
                it with Otsu, treating the vesselness as bright on black for either background.

        Returns:
            float64 vesselness in [0, 1], or a uint8 array of 0 and 1 when threshold is set.
                Same shape as the volume.

        Raises:
            ValueError: If threshold is not None, 'otsu2d' or 'otsu3d'.
        """
        self._check_threshold(threshold)
        logger.info("Running Frangi filter on %d scales.", len(scales))
        all_filters = []
        alpha2 = 2 * (alpha**2) if alpha != 0 else math.nextafter(0, 1)
        beta2  = 2 * (beta**2) if beta != 0 else math.nextafter(0, 1)
        c2     = 2 * (c**2) if c != 0 else math.nextafter(0, 1)
        X, Y, Z = self.volume.shape

        for scale in scales:
            H, T = self._hessian(scale)
            # eigendecomposition
            lambdas = np.linalg.eigvalsh(H)
            V0 = np.zeros_like(self.volume, dtype=np.float64)

            # eigenvalues of the voxels with T == 1, sorted by |lambda|
            l1, l2, l3 = self._sort_by_magnitude(lambdas[T == 1]).T
            # bright vessels need l2, l3 <= 0 (black background), dark vessels l2, l3 >= 0
            if self.background == 'white':
                keep = ~((l2 < 0) | (l3 < 0))
            else:
                keep = ~((l2 > 0) | (l3 > 0))
            l1, l2, l3 = l1[keep], l2[keep], l3[keep]

            l3 = np.where(l3 == 0, math.nextafter(0, 1), l3)
            l2 = np.where(l2 == 0, math.nextafter(0, 1), l2)

            # float_power calls pow like the per-voxel code did (array ** 2 uses x * x, which can differ in the last bit)
            sq1, sq2, sq3 = np.float_power(l1, 2), np.float_power(l2, 2), np.float_power(l3, 2)
            Rb2 = sq1 / (l2 * l3)
            Ra2 = sq2 / sq3
            S2 = sq1 + sq2 + sq3

            term1 = np.exp(-Ra2 / alpha2)
            term2 = np.exp(-Rb2 / beta2)
            term3 = np.exp(-S2 / c2)
            values = np.zeros(keep.size)
            values[keep] = (1.0 - term1) * (term2) * (1.0 - term3)
            V0[T == 1] = values

            all_filters.append(V0)
        
        # stack filters and compute maximum vesselness
        stacked_filters = np.stack(all_filters, axis=-1)
        vesselness = np.max(stacked_filters, axis=-1)

        mask = self._threshold_vesselness(vesselness, threshold)
        if mask is not None:
            return mask

        logger.info("Frangi filter completed.")
        return vesselness

    def bfrangi(self, tau: float, scales: List[float], threshold: Optional[str] = None) -> np.ndarray:
        """
        Beyond Frangi vesselness filter (Jerman et al.).

        The regularization uses tau * max(|lambda3|) at each scale, taken before the sign is
        flipped for the background. Jerman's vesselness3D.m flips the sign first and uses the
        signed maximum, max(lambda3), so results can differ between the two. Voxels outside
        the Yang and Cheng mask get 0.

        Args:
            tau: Regularization, as a fraction of max(|lambda3|) at each scale.
            scales: Gaussian sigmas in voxels. The response is the maximum over the scales.
            threshold: None (default) returns the vesselness. 'otsu2d' or 'otsu3d' thresholds
                it with Otsu, treating the vesselness as bright on black for either background.

        Returns:
            float64 vesselness in [0, 1], or a uint8 array of 0 and 1 when threshold is set.
                Same shape as the volume.

        Raises:
            ValueError: If threshold is not None, 'otsu2d' or 'otsu3d'.
        """
        self._check_threshold(threshold)
        logger.info("Running Beyond Frangi filter on %d scales.", len(scales))
        all_filters = []
        X, Y, Z = self.volume.shape
        
        for scale in scales:
            H, T = self._hessian(scale)
            
            # eigendecomposition
            lambdas = np.linalg.eigvalsh(H)
            V0 = np.zeros_like(self.volume, dtype=np.float64)
            # sort the eigenvalues of the voxels with T == 1 by |lambda|
            lambdas[T == 1] = self._sort_by_magnitude(lambdas[T == 1])

            # find the maximum lambda3 across the volume with scale s
            max_l3 = np.max(abs(lambdas[:, :, :, 2]))
            _, l2, l3 = lambdas[T == 1].T           # no need for lambda1

            if self.background == 'black':
                l2 = -l2
                l3 = -l3

            # calculate lambda rho
            reg_term = tau * max_l3                 # regularized term
            l_rho = np.where((l3 > 0) & (l3 < reg_term), reg_term, np.where(l3 <= 0, 0.0, l3))

            # modified vesselness (Beyond Frangi) function
            # float_power calls pow like the per-voxel code did (array ** 2 uses x * x, which can differ in the last bit)
            values = np.float_power(l2, 2) * (l_rho - l2) * 27 / np.float_power(l2 + l_rho, 3)
            values = np.where((l2 >= (l_rho/2)) & (l_rho > 0), 1.0,
                              np.where((l2 <= 0) | (l_rho <= 0), 0.0, values))
            V0[T == 1] = values

            all_filters.append(V0)
        
        # stack filters and compute maximum vesselness
        stacked_filters = np.stack(all_filters, axis=-1)
        output = np.max(stacked_filters, axis=-1)

        mask = self._threshold_vesselness(output, threshold)
        if mask is not None:
            return mask

        logger.info("Beyond Frangi filter completed.")
        return output

    def oof(self, radii: Sequence[float], sigma: float = 1.0, spacing: Union[float, Sequence[float]] = 1.0,
            threshold: Union[None, str, float] = None, pad: bool = True) -> np.ndarray:
        """
        Optimally oriented flux (OOF), a port of Law's MATLAB oof3response.

        At each radius, the response is the eigenvalue of the oriented flux matrix with the
        largest magnitude (Law and Chung, ECCV 2008). Radii are compared with the curvilinear
        normalization of Law et al. (ECCV 2012), and the output keeps, per voxel, the signed
        response with the largest magnitude over the radii. For a white background the volume
        is negated first, so vessels are positive for either background.

        The normalization needs min(radii) >= sigma. Otherwise the surface area normalization
        of the 2008 paper is used, with a warning, as in Law's code.

        M.W.K. Law and A.C.S. Chung, "Three Dimensional Curvilinear Structure Detection using
        Optimally Oriented Flux", ECCV 2008, pp. 368-382.
        M.W.K. Law et al., "Dilated Divergence based Scale-Space Representation for Curve
        Analysis", ECCV 2012, pp. 557-571.

        Args:
            radii: Radii in the units of `spacing`, for example in voxels with the default.
            sigma: Gaussian smoothing, in the units of `spacing`.
            spacing: Voxel size along array axes 0, 1 and 2, as one number or 3 values.
            threshold: None (default) returns the signed response. 'otsu2d' or 'otsu3d'
                thresholds it with Otsu, treating it as bright on black for either background.
                A number t returns response > t.
            pad: Pad each side by ceil((max(radii) + 3 sigma) / spacing) voxels, mirrored,
                before the FFT, and crop after. Without it the FFT wraps around at the border.

        Returns:
            float64 response, or a uint8 array of 0 and 1 when threshold is set. Same shape as
                the volume.

        Raises:
            ValueError: If threshold is not None, 'otsu2d', 'otsu3d' or a finite number, if the
                volume is not 3D, or if radii, sigma or spacing are not positive and finite.

        Examples:
            >>> import numpy as np
            >>> from microvasc.binarize import Binarizer
            >>> z, y, x = np.mgrid[:16, :16, :16]
            >>> tube = 100.0 * ((y - 8)**2 + (x - 8)**2 <= 4)   # bright tube along axis 0
            >>> response = Binarizer(tube).oof(radii=[1, 2, 3])
            >>> bool(response[8, 8, 8] > 0), bool(response[8, 8, 8] > response[8, 1, 1])
            (True, True)
        """
        number = isinstance(threshold, numbers.Real) and not isinstance(threshold, bool)
        named = threshold is None or (isinstance(threshold, str) and threshold in ('otsu2d', 'otsu3d'))
        if not (named or (number and math.isfinite(threshold))):
            raise ValueError(f"threshold must be None, 'otsu2d', 'otsu3d' or a finite number, got {threshold!r}")
        logger.info("Running OOF on %d radii.", len(np.atleast_1d(radii)))

        volume = np.asarray(self.volume, dtype=np.float64)
        if self.background == 'white':
            volume = -volume
        response = oof_response(volume, radii, sigma=sigma, spacing=spacing, pad=pad)

        if number:
            return (response > threshold).astype(np.uint8)
        mask = self._threshold_vesselness(response, threshold)
        if mask is not None:
            return mask
        logger.info("OOF completed.")
        return response

def parse_args() -> argparse.Namespace:
    """
    Parse the arguments of the binarize command from sys.argv.

    Returns:
        The parsed arguments. method holds the subcommand name.
    """
    parent =argparse.ArgumentParser(add_help=False, description='3D Volume Filtering and Binarization')
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
    vessel.add_argument('--threshold', choices=["otsu2d", "otsu3d"], type=str, default=None, help="Optional thresholding technique.")

    fr = sub.add_parser('frangi', parents=[parent, vessel], help="Frangi Filter")
    fr.add_argument('--alpha', type=float, required=True)
    fr.add_argument('--beta', type=float, required=True)
    fr.add_argument('--c', type=float, required=True)

    bf = sub.add_parser('bfrangi', parents=[parent, vessel], help="Beyond Frangi Filter")
    bf.add_argument('--tau', type=float, required=True)

    return parser.parse_args()

def main() -> None:
    """
    Run the binarize command: load the input .npy volume, apply the method, save the result.

    Subcommands are otsu2d, otsu3d, frangi and bfrangi. For example:

        binarize otsu3d --input in.npy --output out.npy --background white
        binarize frangi --input in.npy --output out.npy --alpha 1 --beta 0.5 --c 3 --scale 1 2 3
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args()
    vol = np.load(args.input)
    binzr = Binarizer(vol, background=args.background)

    if args.method in ('otsu2d', 'otsu3d'):
        out = getattr(binzr, args.method)()
    elif args.method == 'frangi':
        out = binzr.frangi(alpha=args.alpha, beta=args.beta, c=args.c, scales=args.scale, threshold=args.threshold)
    else: # bfrangi
        out = binzr.bfrangi(tau=args.tau, scales=args.scale, threshold=args.threshold)
    
    np.save(args.output, out)
    logger.info("Saved output to %s", args.output)


if __name__ == "__main__":
    main()