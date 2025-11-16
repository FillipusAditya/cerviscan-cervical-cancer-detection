"""
============================================
📌 YUV Color Moment Feature Extraction (Python)
============================================

Description:
------------
This program extracts statistical color moment features (mean, standard deviation, 
and skewness) from images in the YUV color space. The YUV representation separates 
luminance (Y) from chrominance (U and V), which can improve performance in tasks 
where brightness and color variations need to be analyzed independently.

Features:
---------
✅ Converts RGB images into YUV color space using a standard transformation matrix  
✅ Extracts first three color moments (mean, standard deviation, skewness)  
✅ Works on all images that can be opened using Pillow (e.g., .png, .jpg, .jpeg)  
✅ Provides both feature values and corresponding feature names  

Usage:
------
1. Provide the path to the input image file.
2. Call the `get_yuv_color_moment_features(image_path)` function to extract features.
3. Use `get_yuv_color_moment_feature_names()` to retrieve the ordered feature labels.
4. Features can be used for tasks such as image classification, retrieval, or clustering.

Example:
--------
features = get_yuv_color_moment_features("sample.jpg")  
feature_names = get_yuv_color_moment_feature_names()  

print(dict(zip(feature_names, features)))

Author: Fillipus Aditya Nugroho
============================================
"""

import numpy as np
from scipy.stats import skew
from PIL import Image

def get_yuv_color_moment_features(image_path):
    """
    Extract color moment features from an image in the YUV color space.

    This function converts an RGB image to the YUV color space using a standard 
    conversion matrix. It then computes the first three color moments: mean, 
    standard deviation, and skewness for each of the Y, U, and V channels.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    list
        A list containing mean, standard deviation, and skewness for Y, U, and V.
        Order: [mean_y, mean_u, mean_v, std_y, std_u, std_v, skew_y, skew_u, skew_v]
    """
    # Read the input image using Pillow
    image = Image.open(image_path)
    
    # Convert the image to a numpy array (RGB)
    image_array = np.array(image)
    
    # RGB to YUV standard conversion matrix
    yuv_matrix = np.array([
        [0.299,  0.587,  0.114],   # Y channel
        [-0.147, -0.289, 0.436],   # U channel
        [0.615, -0.515, 0.100]     # V channel
    ])

    # Get image dimensions
    image_shape = image_array.shape

    # Initialize output array to store YUV values
    yuv_image = np.zeros(image_shape, dtype=np.float64)

    # Perform RGB to YUV conversion pixel by pixel
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            # Multiply RGB vector by YUV matrix
            yuv_image[i, j] = np.dot(yuv_matrix, image_array[i, j])

    # Compute mean of each channel
    mean_y = np.mean(yuv_image[:, :, 0])
    mean_u = np.mean(yuv_image[:, :, 1])
    mean_v = np.mean(yuv_image[:, :, 2])

    # Compute standard deviation of each channel
    std_y = np.std(yuv_image[:, :, 0])
    std_u = np.std(yuv_image[:, :, 1])
    std_v = np.std(yuv_image[:, :, 2])

    # Compute skewness of each channel
    skew_y = skew(yuv_image[:, :, 0].flatten())
    skew_u = skew(yuv_image[:, :, 1].flatten())
    skew_v = skew(yuv_image[:, :, 2].flatten())

    # Return all color moment features as a list
    return [mean_y, mean_u, mean_v, std_y, std_u, std_v, skew_y, skew_u, skew_v]

def get_yuv_color_moment_feature_names():
    """
    Get the list of feature names for the YUV color moment extraction.

    Returns
    -------
    list
        Names for mean, standard deviation, and skewness for each YUV channel.
    """
    return [
        'mean_y', 'mean_u', 'mean_v',
        'std_y', 'std_u', 'std_v',
        'skew_y', 'skew_u', 'skew_v'
    ]
