import numpy as np
import skimage
import cv2

from scipy.stats import skew

def get_lab_color_moment_features(image_path):
    """
    Extract color moment features from an image in the LAB color space.

    Parameters:
        image_path (str): Path to the image file to be analyzed.

    Returns:
        list: A list of extracted statistical features:
              - Mean (L, A, B channels)
              - Standard deviation (L, A, B channels)
              - Skewness (L, A, B channels)
    """

    # Read the image from file (BGR format by default in OpenCV)
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB for correct color conversion
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the RGB image to a numpy array
    image_array = np.array(rgb_image)
    
    # Normalize RGB values to [0, 1] for skimage compatibility
    rgb_img_normalized = [
        [[element / 255 for element in sublist] for sublist in inner_list]
        for inner_list in image_array
    ]
    
    # Convert normalized RGB image to LAB color space using skimage
    lab_image = skimage.color.rgb2lab(rgb_img_normalized)

    # Calculate mean for each LAB channel
    mean_l = np.mean(lab_image[:, :, 0])
    mean_a = np.mean(lab_image[:, :, 1])
    mean_b = np.mean(lab_image[:, :, 2])

    # Calculate standard deviation for each LAB channel
    std_l = np.std(lab_image[:, :, 0])
    std_a = np.std(lab_image[:, :, 1])
    std_b = np.std(lab_image[:, :, 2])

    # Calculate skewness for each LAB channel
    skew_l = skew(lab_image[:, :, 0].flatten())
    skew_a = skew(lab_image[:, :, 1].flatten())
    skew_b = skew(lab_image[:, :, 2].flatten())

    # Return list of all extracted color moment features
    return [mean_l, mean_a, mean_b, std_l, std_a, std_b, skew_l, skew_a, skew_b]

def get_lab_color_moment_feature_names():
    """
    Get the feature names for the LAB color moment extraction.

    Returns:
        list: Ordered list of feature names corresponding to the extracted values.
    """
    return [
        'mean_l', 'mean_a', 'mean_b',   # Means for L, A, B channels
        'std_l', 'std_a', 'std_b',      # Standard deviations for L, A, B channels
        'skew_l', 'skew_a', 'skew_b'    # Skewness for L, A, B channels
    ]
