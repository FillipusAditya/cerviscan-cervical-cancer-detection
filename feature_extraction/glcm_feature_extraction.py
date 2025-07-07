import cv2 as cv
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy

import os
from tqdm import tqdm
import pandas as pd
import glob

def get_glcm_features(image_path):
    """
    Extract texture features from a grayscale co-occurrence matrix (GLCM) for an image.

    Parameters:
        image_path (str): Path to the image file to be analyzed.

    Returns:
        list: List of extracted feature values, including:
              - contrast1 (float): Contrast value from GLCM.
              - correlation1 (float): Correlation value from GLCM.
              - energy1 (float): Energy value from GLCM.
              - homogeneity1 (float): Homogeneity value from GLCM.
              - res_entropy (float): Entropy of the original image.
    """

    # Read image from file
    image = cv.imread(image_path)

    # Convert image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Define GLCM parameters
    distances = [1, 2]  # Pixel pair distance offsets
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles in radians
    levels = 256  # Number of gray levels (intensity levels)

    # Compute the GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(
        gray_image.astype(int),
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    # Extract texture features from GLCM
    contrast = graycoprops(glcm, prop='contrast')
    contrast1 = round(contrast.flatten()[0], 3)  # Use only the first value

    correlation = graycoprops(glcm, prop='correlation')
    correlation1 = round(correlation.flatten()[0], 3)

    energy = graycoprops(glcm, prop='energy')
    energy1 = round(energy.flatten()[0], 3)

    homogeneity = graycoprops(glcm, prop='homogeneity')
    homogeneity1 = round(homogeneity.flatten()[0], 3)
    
    # Calculate entropy of the original image
    res_entropy = round(entropy(image), 3)
    
    # Return the extracted feature values as a list
    return [contrast1, correlation1, energy1, homogeneity1, res_entropy]

def get_glcm_feature_names():
    """
    Get the names of the texture features extracted from the GLCM.

    Returns:
        list: List of feature names:
              - 'contrast': Contrast feature.
              - 'correlation': Correlation feature.
              - 'energy': Energy feature.
              - 'homogeneity': Homogeneity feature.
              - 'entropy': Image entropy.
    """
    return ['contrast', 'correlation', 'energy', 'homogeneity', 'entropy']
