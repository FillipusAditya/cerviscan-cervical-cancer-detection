import numpy as np
from scipy.stats import skew
from PIL import Image

def get_rgb_color_moment_features(image_path):
    """
    Extract first three color moments (mean, standard deviation, skewness)
    for each channel in the RGB color space.

    Parameters:
        image_path (str): Path to the input image file.

    Returns:
        list: A list of statistical features in order:
              - Mean for R, G, B channels
              - Standard deviation for R, G, B channels
              - Skewness for R, G, B channels
    """
    # Open image using PIL
    image = Image.open(image_path)

    # Convert image to numpy array
    image_array = np.array(image)

    # Ensure the image has 3 channels (RGB)
    if len(image_array.shape) < 3 or image_array.shape[2] != 3:
        raise ValueError(f"Image at {image_path} is not in RGB format.")

    # Calculate mean for each channel
    mean_r = np.mean(image_array[:, :, 0])
    mean_g = np.mean(image_array[:, :, 1])
    mean_b = np.mean(image_array[:, :, 2])

    # Calculate standard deviation for each channel
    std_r = np.std(image_array[:, :, 0])
    std_g = np.std(image_array[:, :, 1])
    std_b = np.std(image_array[:, :, 2])

    # Calculate skewness for each channel
    skew_r = skew(image_array[:, :, 0].flatten())
    skew_g = skew(image_array[:, :, 1].flatten())
    skew_b = skew(image_array[:, :, 2].flatten())

    # Return list of features in order
    return [mean_r, mean_g, mean_b, std_r, std_g, std_b, skew_r, skew_g, skew_b]

def get_rgb_color_moment_feature_names():
    """
    Get the names of the RGB color moment features extracted.

    Returns:
        list: Ordered list of feature names.
    """
    return [
        'mean_r', 'mean_g', 'mean_b',   # Means for R, G, B channels
        'std_r', 'std_g', 'std_b',      # Standard deviations for R, G, B channels
        'skew_r', 'skew_g', 'skew_b'    # Skewness for R, G, B channels
    ]
