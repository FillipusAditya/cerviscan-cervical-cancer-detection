import numpy as np
import cv2

def get_pixel(img, center, x, y):
    """
    Compares the pixel at position (x, y) to the center pixel.
    If the neighbor pixel is greater or equal to the center, returns 1, else 0.

    Handles out-of-bounds by returning 0.

    Parameters:
        img (np.ndarray): Grayscale image.
        center (int): Intensity value of the center pixel.
        x (int): X-coordinate of the neighbor pixel.
        y (int): Y-coordinate of the neighbor pixel.

    Returns:
        int: 1 if neighbor >= center, else 0.
    """
    try:
        return 1 if img[x][y] >= center else 0
    except IndexError:
        return 0

def lbp_calculated_pixel(img, x, y):
    """
    Computes the LBP code for a single pixel.

    Parameters:
        img (np.ndarray): Grayscale image.
        x (int): X-coordinate of the center pixel.
        y (int): Y-coordinate of the center pixel.

    Returns:
        int: The LBP binary code converted to decimal.
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

    Parameters:
        path (str): Path to the image file.

    Returns:
        np.ndarray: 2D LBP image as a grayscale numpy array.
    """
    img_bgr = cv2.imread(path, 1)                   # Load image in BGR
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

    Parameters:
        path (str): Path to the input image.

    Returns:
        list: List containing mean, median, standard deviation, kurtosis, and skewness of LBP values.
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

    Returns:
        list: Ordered list of feature names.
    """
    return ['mean_lbp', 'median_lbp', 'std_lbp', 'kurtosis_lbp', 'skewness_lbp']
