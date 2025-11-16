"""
============================================
📌 GLRLM (Gray Level Run Length Matrix) Computation Class (Python)
============================================

Description:
------------
This program implements a class `getGrayRumatrix` to compute the 
Gray Level Run Length Matrix (GLRLM) and its statistical texture features. 
GLRLM is a widely used method in texture analysis, medical imaging, 
and computer vision tasks, providing insight into the spatial 
distribution of gray levels in an image.

The program allows users to:
- Load and convert images to grayscale
- Compute GLRLM for four directions (0°, 45°, 90°, 135°)
- Extract 11 standard GLRLM-based statistical features

Features:
---------
✅ Load images and convert them into grayscale format  
✅ Compute GLRLM in 4 directional angles: 0°, 45°, 90°, 135°  
✅ Implements 11 GLRLM statistical features:
   - Short Run Emphasis (SRE)  
   - Long Run Emphasis (LRE)  
   - Gray Level Non-Uniformity (GLN)  
   - Run Length Non-Uniformity (RLN)  
   - Run Percentage (RP)  
   - Low Gray Level Run Emphasis (LGLRE)  
   - High Gray Level Run Emphasis (HGLRE)  
   - Short Run Low Gray Level Emphasis (SRLGLE)  
   - Short Run High Gray Level Emphasis (SRHGLE)  
   - Long Run Low Gray Level Emphasis (LRLGLE)  
   - Long Run High Gray Level Emphasis (LRHGLE)  
✅ Handles numerical stability (avoids NaN/Inf values)  
✅ Modular design for easy integration with ML/DL pipelines  

Usage:
------
1. Initialize the class:
       test = getGrayRumatrix()

2. Load an image:
       img_data = test.read_img("sample_image.jpg")

3. Compute GLRLM:
       rlmatrix = test.getGrayLevelRumatrix(img_data, ['deg0'])

4. Extract features:
       sre = test.getShortRunEmphasis(rlmatrix)

Example:
--------
from GrayRumatrix import getGrayRumatrix

test = getGrayRumatrix()
img_data = test.read_img("sample_image.jpg")
rlmatrix = test.getGrayLevelRumatrix(img_data, ['deg0'])
sre = test.getShortRunEmphasis(rlmatrix)
print("Short Run Emphasis:", sre)

Author: Fillipus Aditya Nugroho
============================================
"""

import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np
from itertools import groupby

class getGrayRumatrix:
    def __init__(self):
        """
        Constructor for the `getGrayRumatrix` class.
        Initializes the object with a `data` attribute to store the grayscale image.
        """
        self.data = None
    
    def read_img(self, path=" ", lbp="off"):
        """
        Loads an image from a specified path and converts it to grayscale.

        Parameters
        ----------
        path : str
            Path to the image file.
        lbp : str
            Option to apply Local Binary Pattern (LBP) preprocessing 
            (currently not implemented). Default is 'off'.

        Returns
        -------
        np.ndarray
            Grayscale image data as a numpy array.
        """
        try:
            img = Image.open(path)              # Open image using PIL
            img = img.convert('L')              # Convert image to grayscale
            self.data = np.array(img)           # Store as numpy array
            return self.data
        except Exception as e:
            print(f"Error reading image: {e}")  # Handle errors gracefully
            self.data = None
            return None

    def getGrayLevelRumatrix(self, array, theta):
        """
        Computes the Gray-Level Run Length Matrix (GLRLM) for the image at given angles.

        Parameters
        ----------
        array : np.ndarray
            Grayscale image as a numpy array.
        theta : list of str
            List of angles. Allowed: ['deg0', 'deg45', 'deg90', 'deg135'].

        Returns
        -------
        np.ndarray
            The computed GLRLM with dimensions (gray_levels, run_lengths, angles).
        """
        P = array
        x, y = P.shape
        min_pixels = np.min(P).astype(np.int32)     # Minimum pixel intensity
        max_pixels = np.max(P).astype(np.int32)     # Maximum pixel intensity
        run_length = max(x, y)                      # Maximum possible run length
        num_level = max_pixels - min_pixels + 1     # Number of gray levels

        # Extract pixel sequences for each direction
        deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]
        deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]

        # Diagonal extraction for 45 degrees
        diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]
        deg45 = [n.tolist() for n in diags]

        # Diagonal extraction for 135 degrees
        Pt = np.rot90(P, 3)
        diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
        deg135 = [n.tolist() for n in diags]

        def length(l):
            """Helper function to compute the length of a run."""
            if hasattr(l, '__len__'):
                return np.size(l)
            else:
                return sum(1 for _ in l)

        # Initialize the GLRLM matrix: (gray_levels x run_lengths x angles)
        glrlm = np.zeros((num_level, run_length, len(theta)))
        
        for angle in theta:
            # Iterate over each pixel sequence for the angle
            for splitvec in range(0, len(eval(angle))):
                flattened = eval(angle)[splitvec]
                answer = []
                for key, iter in groupby(flattened):
                    answer.append((key, length(iter)))
                for ansIndex in range(0, len(answer)):
                    glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1
        
        return glrlm

    def apply_over_degree(self, function, x1, x2):
        """
        Applies an element-wise operation over each directional GLRLM matrix.

        Parameters
        ----------
        function : callable
            A function to apply (e.g., np.multiply, np.divide).
        x1 : np.ndarray
            First input array.
        x2 : np.ndarray
            Second input for the function.

        Returns
        -------
        np.ndarray
            The resulting array after applying the function for each angle.
        """
        rows, cols, nums = x1.shape
        result = np.ndarray((rows, cols, nums))
        for i in range(nums):
            result[:, :, i] = function(x1[:, :, i], x2)
        # Replace any inf or nan with zero for numerical stability
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
        return result 

    def calcuteIJ(self, rlmatrix):
        """
        Calculates index matrices for gray levels (I) and run lengths (J).

        Parameters
        ----------
        rlmatrix : np.ndarray
            Input GLRLM.

        Returns
        -------
        tuple
            Tuple (I, J+1) for use in feature calculations.
        """
        gray_level, run_length, _ = rlmatrix.shape
        I, J = np.ogrid[0:gray_level, 0:run_length]
        return I, J+1

    def calcuteS(self, rlmatrix):
        """
        Calculates the total sum of the GLRLM.

        Parameters
        ----------
        rlmatrix : np.ndarray
            Input GLRLM.

        Returns
        -------
        float
            Sum of all elements.
        """
        return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

    # Below are the standard GLRLM statistical feature

    # 1. Short Run Emphasis (SRE)
    def getShortRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.divide, rlmatrix, (J*J)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 2. Long Run Emphasis (LRE)
    def getLongRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.multiply, rlmatrix, (J*J)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 3. Gray Level Non-Uniformity (GLN)
    def getGrayLevelNonUniformity(self, rlmatrix):
        G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
        numerator = np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 4. Run Length Non-Uniformity (RLN)
    def getRunLengthNonUniformity(self, rlmatrix):
        R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
        numerator = np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 5. Run Percentage (RP)
    def getRunPercentage(self, rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        num_voxels = gray_level * run_length
        return self.calcuteS(rlmatrix) / num_voxels

    # 6. Low Gray Level Run Emphasis (LGLRE)
    def getLowGrayLevelRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.divide, rlmatrix, (I*I)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 7. High Gray Level Run Emphasis (HGLRE)
    def getHighGrayLevelRunEmphais(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.multiply, rlmatrix, (I*I)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 8. Short Run Low Gray Level Emphasis (SRLGLE)
    def getShortRunLowGrayLevelEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.divide, rlmatrix, (I*I*J*J)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 9. Short Run High Gray Level Emphasis (SRHGLE)
    def getShortRunHighGrayLevelEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (I*I))
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.divide, temp, (J*J)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 10. Long Run Low Gray Level Emphasis (LRLGLE)
    def getLongRunLow(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (J*J))
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.divide, temp, (J*J)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 11. Long Run High Gray Level Emphasis (LRHGLE)
    def getLongRunHighGrayLevelEmphais(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(
            np.sum,
            self.apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)),
            axes=(0, 1)
        )[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
