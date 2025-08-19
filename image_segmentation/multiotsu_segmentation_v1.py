import cv2 as cv
import numpy as np
import os
import glob
from tqdm import tqdm
from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

def multi_otsu_segmentation(input_root: str, output_root: str, classes: int = 5) -> None:
    """
    Apply Multi-Otsu thresholding segmentation on grayscale images in class subdirectories.

    Parameters:
        input_root (str): Root directory containing 'normal' and 'abnormal' folders with images.
        output_root (str): Directory where binary masks will be saved.
        classes (int): Number of classes for thresholding (default: 5).
    """
    # Get all image paths in subdirectories (normal, abnormal)
    image_paths = glob.glob(os.path.join(input_root, "*", "*"))

    print(f"Found {len(image_paths)} images for segmentation.")

    for img_path in tqdm(image_paths, desc="Multi-Otsu Segmentation", ncols=100):
        # Read image in grayscale
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue

        # Apply Multi-Otsu thresholding
        thresholds = threshold_multiotsu(img, classes=classes)
        regions = np.digitize(img, bins=thresholds)

        # Create binary mask of highest intensity region only
        output = (regions * (255 // (regions.max() + 1))).astype(np.uint8)
        output[output < np.unique(output)[-1]] = 0
        output[output >= np.unique(output)[-1]] = 1

        # Prepare save path
        rel_path = os.path.relpath(img_path, input_root)  # e.g., normal/image1.jpg
        save_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save binary mask image
        plt.imsave(save_path, output, cmap='gray')

    print(f"Segmentation complete. Masks saved to: {output_root}")

if __name__ == "__main__":
    # Define your input and output directories here
    input_dir = "../datasets/dataset_g/cropped_image_after_iva_reference_4/equalized"
    output_dir = "../datasets/dataset_g/cropped_image_after_iva_reference_4/equalized_mask"

    # Run segmentation
    multi_otsu_segmentation(input_dir, output_dir, classes=5)
