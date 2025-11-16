"""
============================================================
📌 Multi-Otsu Thresholding Segmentation (Python)
============================================================

Description:
------------
This program performs image segmentation using the Multi-Otsu 
thresholding method on grayscale images organized into class 
subdirectories (e.g., 'normal' and 'abnormal').

Multi-Otsu segmentation divides an image into multiple intensity 
regions based on optimal threshold values and is widely used in 
medical image preprocessing, lesion detection, and feature extraction.

This implementation extracts only the highest-intensity region to 
generate a clean binary mask suitable for downstream analysis.

Features:
---------
✅ Supports automatic scanning of all images inside class folders  
✅ Applies Multi-Otsu thresholding with configurable number of classes  
✅ Generates binary masks (0 and 1) of the brightest/intense region  
✅ Preserves subdirectory structure when saving outputs  
✅ Useful for preprocessing medical images for classification or segmentation models  

Usage:
------
1. Place image folders under the input root directory 
   (e.g., input_root/normal/, input_root/abnormal/).
2. Set the desired number of thresholding classes (default = 5).
3. Run the segmentation function:
       multi_otsu_segmentation(input_root, output_root, classes=5)
4. Binary mask images will be saved to the output directory.

Example:
--------
from segmentation_multiotsu import multi_otsu_segmentation

input_dir = "path/to/grayscale_images"
output_dir = "path/to/save_masks"

multi_otsu_segmentation(input_dir, output_dir, classes=5)

Output Structure:
-----------------
input_root/
 ├── normal/
 │    ├── img1.jpg
 │    ├── img2.jpg
 └── abnormal/
      ├── img3.jpg
      ├── img4.jpg

output_root/
 ├── normal/
 │    ├── img1.jpg   (binary mask)
 │    ├── img2.jpg   (binary mask)
 └── abnormal/
      ├── img3.jpg   (binary mask)
      ├── img4.jpg   (binary mask)

Author:
-------
Fillipus Aditya Nugroho
============================================================
"""

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
    input_dir = ""
    output_dir = ""

    # Run segmentation
    multi_otsu_segmentation(input_dir, output_dir, classes=5)
