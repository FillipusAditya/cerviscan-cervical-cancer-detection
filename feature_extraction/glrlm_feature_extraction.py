"""
============================================
📌 GLRLM (Gray Level Run Length Matrix) Feature Extraction (Python)
============================================

Description:
------------
This program extracts GLRLM (Gray Level Run Length Matrix) features from a given image. 
GLRLM is a texture analysis method widely used in image processing and pattern 
recognition tasks, particularly in medical imaging and computer vision. 

It supports computing features in multiple directions (0°, 45°, 90°, 135°) and provides 
commonly used statistical measures such as Short Run Emphasis, Long Run Emphasis, 
Gray Level Non-Uniformity, Run Percentage, and more.

Features:
---------
✅ Extracts 11 GLRLM texture features per direction
✅ Supports 4 directional angles: 0°, 45°, 90°, 135°
✅ Optional Local Binary Pattern (LBP) preprocessing
✅ Returns both feature values and feature names with direction labels
✅ Easily integrable with machine learning pipelines

Usage:
------
1. Provide the path to the input image.
2. Optionally enable Local Binary Pattern (LBP) preprocessing by setting `lbp='on'`.
3. Call `get_glrlm_features(path)` to extract GLRLM features.
4. Use `get_glrlm_feature_names()` to get the corresponding feature names.

Example:
--------
from glrlm_feature_extraction import get_glrlm_features, get_glrlm_feature_names

features = get_glrlm_features("sample_image.jpg", lbp='off')
feature_names = get_glrlm_feature_names()

print(dict(zip(feature_names, features)))

Author: Fillipus Aditya Nugroho
============================================
"""

import numpy as np
import warnings
from GrayRumatrix import getGrayRumatrix

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

def get_glrlm_features(path, lbp='off'):
    """
    Calculate GLRLM (Gray Level Run Length Matrix) features for a given image.

    Parameters
    ----------
    path : str
        Path to the input image.
    lbp : str, optional
        If 'on', apply Local Binary Pattern (LBP) transformation before computing GLRLM.
        Defaults to 'off'.

    Returns
    -------
    list
        List of extracted GLRLM feature values for each specified direction.
    """

    # Initialize GLRLM processing object
    test = getGrayRumatrix()

    # Read and preprocess image (optionally apply LBP)
    test.read_img(path)

    # Define directions for which GLRLM will be computed
    DEG = [['deg0'], ['deg45'], ['deg90'], ['deg135']]

    glrlm_features_value = []

    # Loop through each direction to compute features
    for deg in DEG:
        # Compute GLRLM matrix for the current direction
        test_data = test.getGrayLevelRumatrix(test.data, deg)

        # Calculate various statistical measures from GLRLM
        SRE = test.getShortRunEmphasis(test_data)  # Short Run Emphasis
        SRE = float(np.squeeze(SRE))
        
        LRE = test.getLongRunEmphasis(test_data)   # Long Run Emphasis
        LRE = float(np.squeeze(LRE))
        
        GLN = test.getGrayLevelNonUniformity(test_data)  # Gray Level Non-Uniformity
        GLN = float(np.squeeze(GLN))
        
        RLN = test.getRunLengthNonUniformity(test_data)  # Run Length Non-Uniformity
        RLN = float(np.squeeze(RLN))

        RP = test.getRunPercentage(test_data)  # Run Percentage
        RP = float(np.squeeze(RP))
        
        LGLRE = test.getLowGrayLevelRunEmphasis(test_data)  # Low Gray Level Run Emphasis
        LGLRE = float(np.squeeze(LGLRE))
        
        HGL = test.getHighGrayLevelRunEmphais(test_data)  # High Gray Level Run Emphasis
        HGL = float(np.squeeze(HGL))
        
        SRLGLE = test.getShortRunLowGrayLevelEmphasis(test_data)  # Short Run Low Gray Level Emphasis
        SRLGLE = float(np.squeeze(SRLGLE))
        
        SRHGLE = test.getShortRunHighGrayLevelEmphasis(test_data)  # Short Run High Gray Level Emphasis
        SRHGLE = float(np.squeeze(SRHGLE))
        
        LRLGLE = test.getLongRunLow(test_data)  # Long Run Low Gray Level Emphasis
        LRLGLE = float(np.squeeze(LRLGLE))
        
        LRHGLE = test.getLongRunHighGrayLevelEmphais(test_data)  # Long Run High Gray Level Emphasis
        LRHGLE = float(np.squeeze(LRHGLE))

        # Combine all features for the current direction
        glrlm_features_value_per_deg = [
            SRE, LRE, GLN, RLN, RP,
            LGLRE, HGL, SRLGLE, SRHGLE, LRLGLE, LRHGLE
        ]
        
        # Append features to the overall result list
        for value in glrlm_features_value_per_deg:
            glrlm_features_value.append(value)

    # Return final list of GLRLM feature values for all directions
    return glrlm_features_value

def get_glrlm_names(features, degs):
    """
    Generate feature names for GLRLM, combining each feature name with directional labels.

    Parameters
    ----------
    features : list
        List of base feature names (e.g., SRE, LRE, etc.).
    degs : list of lists
        List of directional angles to append to feature names
        (e.g., ['deg0'], ['deg45'], etc.).

    Returns
    -------
    list
        List of concatenated feature names with directional suffixes.
    """
    glrlm_features_name = []

    # For each direction and feature, create a combined name
    for deg in degs:
        for feature in features:
            glrlm_features_name.append(f"{feature}_{deg[0]}")
    return glrlm_features_name

def get_glrlm_feature_names():
    """
    Get the complete list of all possible GLRLM feature names across all directions.

    Returns
    -------
    list
        Full list of GLRLM feature names with directional labels.
    """
    glrlm_features = [
        'SRE', 'LRE', 'GLN', 'RLN', 'RP',
        'LGLRE', 'HGL', 'SRLGLE', 'SRHGLE', 'LRLGLE', 'LRHGLE'
    ]
    glrlm_degs = [['deg0'], ['deg45'], ['deg90'], ['deg135']]
    return get_glrlm_names(glrlm_features, glrlm_degs)
