"""
============================================
📌 Tamura Texture Feature Extraction (Python)
============================================

Description:
------------
This program extracts Tamura texture features from a given image, which are 
widely used in image processing and computer vision for texture analysis. 
Tamura’s features are designed to capture human visual perception of texture 
based on psychological studies.

The extracted features include:
- Coarseness: Measures the granularity of texture patterns.
- Contrast: Represents the dynamic range of gray levels.
- Directionality: Indicates the degree of orientation of image structures.
- Roughness: Combined feature from coarseness and contrast.

Features:
---------
✅ Computes coarseness, contrast, directionality, and roughness
✅ Implements Tamura’s original texture feature formulas
✅ Works with grayscale images (converts automatically from RGB if needed)
✅ Provides helper function for feature names

Usage:
------
1. Provide the path to an image file.
2. Call `get_tamura_features(image_path)` to extract the features.
3. Use `get_tamura_feature_names()` to get the ordered feature names.

Example:
--------
from tamura_features import get_tamura_features, get_tamura_feature_names

features = get_tamura_features("example.jpg")
feature_names = get_tamura_feature_names()

print(dict(zip(feature_names, features)))

Author: Fillipus Aditya Nugroho
============================================
"""

import cv2
import numpy as np


def coarseness(image, kmax):
    """
    Calculate the coarseness feature of an image based on Tamura's texture features.

    Coarseness measures the granularity or scale of the texture patterns.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    kmax : int
        Maximum window size exponent for local averaging.

    Returns
    -------
    float
        Coarseness value.
    """
    image = np.array(image)
    w, h = image.shape

    # Ensure kmax does not exceed image dimensions
    kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))

    average_gray = np.zeros([kmax, w, h])  # Local mean gray level
    horizon = np.zeros([kmax, w, h])       # Horizontal differences
    vertical = np.zeros([kmax, w, h])      # Vertical differences
    Sbest = np.zeros([w, h])               # Best window size for each pixel

    for k in range(kmax):
        window = np.power(2, k)
        # Calculate local averages
        for wi in range(window, w - window):
            for hi in range(window, h - window):
                average_gray[k][wi][hi] = np.sum(image[wi - window:wi + window, hi - window:hi + window])
        # Calculate horizontal and vertical differences
        for wi in range(window, w - window - 1):
            for hi in range(window, h - window - 1):
                horizon[k][wi][hi] = average_gray[k][wi + window][hi] - average_gray[k][wi - window][hi]
                vertical[k][wi][hi] = average_gray[k][wi][hi + window] - average_gray[k][wi][hi - window]
        # Normalize differences
        horizon[k] *= (1.0 / np.power(2, 2 * (k + 1)))
        vertical[k] *= (1.0 / np.power(2, 2 * (k + 1)))

    # Select best window size per pixel based on maximum difference
    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:, wi, hi])
            h_max_index = np.argmax(horizon[:, wi, hi])
            v_max = np.max(vertical[:, wi, hi])
            v_max_index = np.argmax(vertical[:, wi, hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2, index)

    fcrs = np.mean(Sbest)
    return fcrs


def contrast(image):
    """
    Calculate the contrast feature of an image based on Tamura's texture features.

    Contrast measures the dynamic range of gray levels in the image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    float
        Contrast value.
    """
    image = np.array(image)
    image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    m4 = np.mean(np.power(image - np.mean(image), 4))  # Fourth moment
    v = np.var(image)                                  # Variance
    std = np.sqrt(v)                                   # Standard deviation
    alfa4 = m4 / np.power(v, 2)                        # Normalized fourth moment
    fcon = std / np.power(alfa4, 0.25)                 # Tamura contrast
    return fcon


def directionality(image):
    """
    Calculate the directionality feature of an image based on Tamura's texture features.

    Directionality measures the degree of orientation and alignment of patterns.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    float
        Directionality value.
    """
    image = np.array(image, dtype='int64')
    h, w = image.shape

    # Prewitt-like kernels for horizontal and vertical gradients
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    deltaH = np.zeros([h, w])  # Horizontal gradient
    deltaV = np.zeros([h, w])  # Vertical gradient
    theta = np.zeros([h, w])   # Orientation angles

    # Compute horizontal gradient
    for hi in range(1, h - 1):
        for wi in range(1, w - 1):
            deltaH[hi][wi] = np.sum(np.multiply(image[hi - 1:hi + 2, wi - 1:wi + 2], convH))
    # Handle borders for deltaH
    for wi in range(1, w - 1):
        deltaH[0][wi] = image[0][wi + 1] - image[0][wi]
        deltaH[h - 1][wi] = image[h - 1][wi + 1] - image[h - 1][wi]
    for hi in range(h):
        deltaH[hi][0] = image[hi][1] - image[hi][0]
        deltaH[hi][w - 1] = image[hi][w - 1] - image[hi][w - 2]

    # Compute vertical gradient
    for hi in range(1, h - 1):
        for wi in range(1, w - 1):
            deltaV[hi][wi] = np.sum(np.multiply(image[hi - 1:hi + 2, wi - 1:wi + 2], convV))
    # Handle borders for deltaV
    for wi in range(w):
        deltaV[0][wi] = image[1][wi] - image[0][wi]
        deltaV[h - 1][wi] = image[h - 1][wi] - image[h - 2][wi]
    for hi in range(1, h - 1):
        deltaV[hi][0] = image[hi + 1][0] - image[hi][0]
        deltaV[hi][w - 1] = image[hi + 1][w - 1] - image[hi][w - 1]

    # Compute gradient magnitude and orientation
    deltaG = (np.abs(deltaH) + np.abs(deltaV)) / 2.0
    deltaG_vec = deltaG.flatten()

    for hi in range(h):
        for wi in range(w):
            if deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0:
                theta[hi][wi] = 0
            elif deltaH[hi][wi] == 0:
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
    theta_vec = theta.flatten()

    # Histogram of directions
    n = 16  # Number of bins
    t = 12  # Threshold for gradient magnitude
    hd = np.zeros(n)
    dlen = deltaG_vec.shape[0]

    for ni in range(n):
        for k in range(dlen):
            if ((deltaG_vec[k] >= t) and
                (theta_vec[k] >= (2 * ni - 1) * np.pi / (2 * n)) and
                (theta_vec[k] < (2 * ni + 1) * np.pi / (2 * n))):
                hd[ni] += 1

    hd /= np.mean(hd)  # Normalize histogram
    hd_max_index = np.argmax(hd)

    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]

    return fdir


def roughness(fcrs, fcon):
    """
    Calculate the roughness feature by combining coarseness and contrast.

    Roughness = Coarseness + Contrast

    Parameters
    ----------
    fcrs : float
        Coarseness value.
    fcon : float
        Contrast value.

    Returns
    -------
    float
        Roughness value.
    """
    return fcrs + fcon


def get_tamura_features(image_path):
    """
    Extract Tamura texture features (coarseness, contrast, directionality, roughness)
    from a given image.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    list
        List of feature values in order.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fcrs = coarseness(img, 5)
    fcon = contrast(img)
    fdir = directionality(img)
    frgh = roughness(fcrs, fcon)

    return [fcrs, fcon, fdir, frgh]


def get_tamura_feature_names():
    """
    Get the names of Tamura texture features.

    Returns
    -------
    list
        Ordered list of Tamura feature names:
        - coarseness_tamura
        - contrast_tamura
        - directionality_tamura
        - roughness_tamura
    """
    return ['coarseness_tamura', 'contrast_tamura', 'directionality_tamura', 'roughness_tamura']
