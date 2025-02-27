import gc
import math
import torch
import numpy as np
import logging as log
import scipy.linalg as lin
import scipy.ndimage.filters as filters

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Binarize:
    """
    A class to binarize 3D volumes using Otsu's thresholding, Frangi filter, Beyond Frangi filter, and U-Net.
    
    Attributes:
        volume (np.ndarray): The grayscale input 3D volume to be processed.
        background (str): Background type, either 'black' or 'white'. Defaults to 'black'.
        model (torch.nn.Module): Optional U-Net model for segmentation. Defaults to None
    """
    
    def __init__(self, volume, background='black', unet_model=None):
        self.volume = volume
        self.background = background
        self.model = unet_model
        log.info("Binarize object created.")
        
        
    def _compute_threshold(self):
        """
        Compute the Otsu threshold for the input volume.

        Returns:
            float: The Otsu threshold value.
        """
        if np.all(self.volume == 0):
            log.warning("Input volume is all zeros.")
            return 0
        # compute historgram and probabilities
        hist, bin_edges = np.histogram(self.volume, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # compute cumilative sums and means
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
        
        # compute inter-class variance
        inter_class_variance = weight1[:-1] * weight2[1:] * ((mean1[:-1] - mean2[1:]) ** 2)
        threshold = bin_centers[:-1][np.argmax(inter_class_variance)]
        
        log.info(f"Computed Otsu threshold: {threshold}")
        return threshold
    
    
    def otsu_2d(self):
        """
        Apply Otsu's thresholding on each 2D slice of the volume.

        Returns:
            np.ndarray: Thresholded 3D volume.
        """
        if self.volume.size == 0:
            log.error("Input volume is empty.")
            raise ValueError("Input volume is empty.")
        elif np.all(self.volume == 0):
            log.warning("Input volume is all zeros.")
            return np.uint8(self.volume)
        
        log.info("Applying Otsu's thresholding on 2D slices.")
        # apply otsu on each image of the volume
        thresh_imgs = []
        for j in range(self.volume.shape[0]):
            image = self.volume[j]
            if image.dtype == np.float64:
                image = np.uint8(image*255)
            
            threshold = self._compute_threshold()
            
            # apply threshold
            if(self.background == 'black'):
                thresh_img = (image >= threshold).astype(np.uint8)
            elif(self.background == 'white'):
                thresh_img = (image <= threshold).astype(np.uint8)
            else:
                log.error("Invalid background option. Choose 'black' or 'white'.")
                raise ValueError("Invalid background option. Choose 'black' or 'white'.")
            thresh_imgs.append(thresh_img)
            
        log.info("Otsu's thresholding completed.")
        return np.stack(thresh_imgs, axis=0)


    def otsu_3d(self):
        """
        Apply Otsu's thresholding on the entire 3D volume.

        Returns:
            np.ndarray: Thresholded 3D volume.
        """
        if self.volume.size == 0:
            log.error("Input volume is empty.")
            raise ValueError("Input volume is empty.")
        elif np.all(self.volume == 0):
            log.warning("Input volume is all zeros.")
            return np.uint8(self.volume)
        
        log.info("Applying Otsu's thresholding on 3D volume.")
        threshold = self._compute_threshold()
        
        if self.background == 'black':
            threshed_otsu3d = (self.volume >= threshold).astype(np.uint8)
        elif self.background == 'white':
            threshed_otsu3d = (self.volume < threshold).astype(np.uint8)
        else:
            log.error("Invalid background option. Choose 'black' or 'white'.")
            raise ValueError("Invalid background option. Choose 'black' or 'white'.")
        
        log.info("Otsu's thresholding completed.")
        return threshed_otsu3d


    def _compute_hessian(self, scale):
        """
        Compute the Hessian matrix for a given scale.

        Args:
            scale (float): Scale (sigma) for Gaussian derivatives.

        Returns:
            tuple: Hessian matrix (H) and mask (T).
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
    
    
    def frangi_filter(self, alpha, beta, c, scale_range, threshold=None):
        """
        Apply the Frangi (aka. vesselness) filter for tubular enhancement in the volume.

        Args:
            alpha (float): parameter for line-like structures.
            beta (float): parameter for blob-like structures.
            c (float): parameter for background noise suppression.
            scale_range (list): List of scales to apply the filter.
            threshold (str): Thresholding method, either 'otsu2d', 'otsu3d', or None.

        Returns:
            np.ndarray: Filtered (or binarized if threshold applied) 3D volume.
        """
        if not scale_range:
            log.error("Scale range cannot be empty.")
            raise ValueError("Scale range cannot be empty.")
            
        log.info("Applying Frangi filter.")
        all_filters = []
        alpha2 = 2 * (alpha**2) if alpha != 0 else math.nextafter(0, 1)
        beta2  = 2 * (beta**2) if beta != 0 else math.nextafter(0, 1)
        c2     = 2 * (c**2) if c != 0 else math.nextafter(0, 1)
        X, Y, Z = self.volume.shape

        for scale in scale_range:
            H, T = self._compute_hessian(scale)
            
            # eigendecomposition
            lambdas = lin.eigvalsh(H)
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
        output = np.max(stacked_filters, axis=-1)

        if threshold is None:
            log.info("Frangi filter completed.")
            return output
        elif threshold == 'otsu2d':
            log.info("Frangi filter completed (Binarized using Otsu's on 2D slices.)")
            self.volume = output
            return self.otsu_2d()
        elif threshold == 'otsu3d':
            log.info("Frangi filter completed (Binarized using Otsu's on 3D volume.)")
            self.volume = output
            return self.otsu_3d()
        else:
            log.error("Invalid thresholding method. Options: None, 'otsu2d', 'otsu3d'.")
            raise ValueError("Invalid thresholding method. Options: None, 'otsu2d', 'otsu3d'.")



    def bfrangi_filter(self, tau, scale_range, threshold=False):
        """
        Apply the Beyond Frangi filter for tubular enhancement in the volume.

        Args:
            tau (float): Regularization parameter for Beyond Frangi filter.
            scale_range (list): List of scales to apply the filter.
            threshold (str): Thresholding method, either 'otsu2d', 'otsu3d', or None.

        Returns:
            np.ndarray: Filtered 3D volume.
        """
        if not scale_range:
            log.error("Scale range cannot be empty.")
            raise ValueError("Scale range cannot be empty.")
        
        log.info("Applying Beyond Frangi filter.")
        all_filters = []
        X, Y, Z = self.volume.shape()
        
        for scale in scale_range:
            H, T = self._compute_hessian(scale)
            
            # eigendecomposition
            lambdas = lin.eigvalsh(H)
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

        if threshold == None:
            log.info("Beyond Frangi filter completed.")
            return output
        elif threshold == 'otsu2d':
            log.info("Beyond Frangi filter completed (Binarized using Otsu's on 2D slices.)")
            self.volume = output
            return self.otsu_2d()
        elif threshold == 'otsu3d':
            log.info("Beyond Frangi filter completed (Binarized using Otsu's on 3D volume.)")
            self.volume = output
            return self.otsu_3d()
        else:
            log.error("Invalid thresholding method. Options: None, 'otsu2d', 'otsu3d'.")
            raise ValueError("Invalid thresholding method. Options: None, 'otsu2d', 'otsu3d'.")

    def unet(self, threshold=False):
        """
        Apply a U-Net model for segmentation or binarization.
        
        Args:
            threshold (bool): Returns binarized volume if True, 
        Returns:
            np.ndarray: Segmented 3D volume.
        """
        
        if self.model is None:
            log.error("No U-Net model provided.")
            raise ValueError("No U-Net model provided.")
        
        # TODO: write what you ran through terminal for the model prediction and binarization