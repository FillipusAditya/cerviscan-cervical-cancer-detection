"""
Multi-Otsu Segmentation with Gamma Correction
---------------------------------------------

This script performs image segmentation on a set of RGB images by combining
Gamma Correction and Multi-Otsu Thresholding.

Workflow:
    1. Input images are read from a flat directory (no subfolders).
    2. Each image is processed with multiple gamma correction values.
    3. For each gamma value:
        - Convert BGR → RGB.
        - Convert RGB → Grayscale.
        - Apply Gamma Correction.
        - Apply Multi-Otsu Thresholding (classes=5).
        - Create a binary mask of the highest intensity region.
        - Multiply the mask with the original RGB image to obtain segmentation.
        - Save results in: {output_root}/segmented_gamma_correction_{gamma_value}
"""

import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from skimage.filters import threshold_multiotsu

# ---------- Utility Functions ----------
def gamma_correction(image, gamma):
    """
    Apply gamma correction to a grayscale image.

    Parameters:
        image (np.ndarray): Grayscale image (2D array).
        gamma (float): Gamma value to apply.

    Returns:
        np.ndarray: Gamma-corrected grayscale image.
    """
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    corrected_img = np.uint8(corrected * 255)
    return corrected_img


def normalize_gamma_name(gamma):
    """
    Normalize gamma value for folder naming.

    Example:
        0.5 -> "0_5"
        1.0 -> "1_0"

    Parameters:
        gamma (float): Gamma value.

    Returns:
        str: Normalized string for folder naming.
    """
    return str(gamma).replace(".", "_")


# ---------- Main Pipeline ----------
def process_images(input_dir, output_root, gamma_values):
    """
    Process all images in a directory with multiple gamma values
    using Multi-Otsu segmentation.

    Parameters:
        input_dir (str): Path to input images directory (flat structure).
        output_root (str): Path to save output segmented images.
        gamma_values (list of float): Gamma correction values to apply.
    """
    os.makedirs(output_root, exist_ok=True)

    # Collect image files (only .png, .jpg, .jpeg)
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for gamma in gamma_values:
        # Create output folder for this gamma
        gamma_str = normalize_gamma_name(gamma)
        output_dir = os.path.join(output_root, f"segmented_gamma_correction_{gamma_str}")
        os.makedirs(output_dir, exist_ok=True)

        # Print info
        print(f"\nℹ️ Processing Gamma={gamma}")
        print(f"ℹ️ Output directory: {output_dir}")

        # Iterate over images with progress bar
        for file in tqdm(image_files, desc=f"Gamma {gamma}", unit="image"):
            img_path = os.path.join(input_dir, file)

            # Step 3: Read image and convert BGR → RGB
            img_bgr = cv.imread(img_path)
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

            # Step 4: Convert to grayscale
            gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

            # Step 5: Apply Gamma correction
            gray_corrected = gamma_correction(gray, gamma)

            # Step 6: Multi-Otsu thresholding (classes=5)
            thresholds = threshold_multiotsu(gray_corrected, classes=5)

            # Step 7: Assign pixels into regions
            regions = np.digitize(gray_corrected, bins=thresholds)

            # Step 8: Select highest intensity region (binary mask)
            mask = (regions == regions.max()).astype(np.uint8) * 255

            # Step 9: Expand mask to 3 channels and segment image
            mask_rgb = cv.merge([mask, mask, mask])
            segmented = cv.bitwise_and(img_rgb, mask_rgb)

            # Step 10: Save segmented image in output directory
            save_path = os.path.join(output_dir, file)
            cv.imwrite(save_path, cv.cvtColor(segmented, cv.COLOR_RGB2BGR))


# ---------- Run ----------
if __name__ == "__main__":
    input_directory = "../datasets/base_dataset/og"
    output_directory = "../datasets/filtered_dataset/multiotsu_gamma"
    gamma_values = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]  # Multiple gamma values

    process_images(input_directory, output_directory, gamma_values)
