"""
============================================
📌 Local Binary Pattern (LBP) Feature Extraction (Python)
============================================

Description:
------------
This program extracts texture features from images using the 
Local Binary Pattern (LBP) method. LBP is a powerful 
texture descriptor that encodes local patterns of pixel 
intensity variations, widely used in image analysis, 
face recognition, and pattern classification.

Features:
---------
✅ Computes LBP code for each pixel based on its 8 neighbors
✅ Generates an LBP-transformed grayscale image
✅ Extracts basic statistical descriptors from the LBP image:
   - Mean
   - Median
   - Standard Deviation
   - Kurtosis
   - Skewness
✅ Returns both feature values and feature names for integration 
   with machine learning pipelines

Usage:
------
1. Provide an image path to `get_lbp_features(path)` to extract 
   statistical features.
2. Use `lbp_implementation(path)` to obtain the LBP-transformed image.
3. Call `get_lbp_feature_names()` to retrieve feature names.

Example:
--------
features = get_lbp_features("sample_image.jpg")
feature_names = get_lbp_feature_names()

print(dict(zip(feature_names, features)))

Author: Fillipus Aditya Nugroho
============================================
"""

import numpy as np
import cv2


def get_pixel(img, center, x, y):
    """
    Compare the pixel at position (x, y) to the center pixel.
    If the neighbor pixel is greater or equal to the center, returns 1, else 0.
    Handles out-of-bounds by returning 0.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image.
    center : int
        Intensity value of the center pixel.
    x : int
        X-coordinate of the neighbor pixel.
    y : int
        Y-coordinate of the neighbor pixel.

    Returns
    -------
    int
        1 if neighbor >= center, else 0.
    """
    try:
        return 1 if img[x][y] >= center else 0
    except IndexError:
        return 0


def lbp_calculated_pixel(img, x, y):
    """
    Compute the LBP code for a single pixel.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image.
    x : int
        X-coordinate of the center pixel.
    y : int
        Y-coordinate of the center pixel.

    Returns
    -------
    int
        The LBP binary code converted to decimal.
    """
    center = img[x][y]

    # Get binary pattern by comparing 8 neighbors
    val_ar = [
        get_pixel(img, center, x - 1, y - 1),  # top-left
        get_pixel(img, center, x - 1, y),      # top
        get_pixel(img, center, x - 1, y + 1),  # top-right
        get_pixel(img, center, x, y + 1),      # right
        get_pixel(img, center, x + 1, y + 1),  # bottom-right
        get_pixel(img, center, x + 1, y),      # bottom
        get_pixel(img, center, x + 1, y - 1),  # bottom-left
        get_pixel(img, center, x, y - 1)       # left
    ]

    # Each bit has a corresponding power of two
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    # Convert binary pattern to decimal value
    return sum(val_ar[i] * power_val[i] for i in range(8))


def lbp_implementation(path):
    """
    Apply Local Binary Pattern (LBP) transformation on an image.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        2D LBP image as a grayscale numpy array.
    """
    img_bgr = cv2.imread(path, 1)                         # Load image in BGR
    height, width, _ = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Initialize output LBP image
    img_lbp = np.zeros((height, width), np.uint8)

    # Loop through each pixel and compute its LBP code
    for i in range(height):
        for j in range(width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    return img_lbp


def get_lbp_features(path):
    """
    Extract basic statistical features from the LBP image.

    Parameters
    ----------
    path : str
        Path to the input image.

    Returns
    -------
    list
        List containing the following LBP statistics:
        - Mean
        - Median
        - Standard Deviation
        - Kurtosis
        - Skewness
    """
    lbp_image = lbp_implementation(path).flatten()

    mean = np.mean(lbp_image)
    median = np.median(lbp_image)
    std = np.std(lbp_image)
    n = len(lbp_image)

    # Calculate kurtosis (using custom formula)
    squared_differences = (lbp_image - mean) ** 4
    sum_of_squared_differences = np.sum(squared_differences)
    kurtosis = (4 * sum_of_squared_differences) / (n * std ** 4) - 3

    # Calculate skewness (Pearson's second skewness coefficient)
    skewness = (3 * (mean - median)) / std

    return [mean, median, std, kurtosis, skewness]


def get_lbp_feature_names():
    """
    Get the names of the LBP features extracted.

    Returns
    -------
    list
        Ordered list of feature names:
        - 'mean_lbp'
        - 'median_lbp'
        - 'std_lbp'
        - 'kurtosis_lbp'
        - 'skewness_lbp'
    """
    return ['mean_lbp', 'median_lbp', 'std_lbp', 'kurtosis_lbp', 'skewness_lbp']
