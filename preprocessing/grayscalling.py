"""
============================================================
📌 Convert Dataset Images to Grayscale (Python)
============================================================

Description:
------------
This program converts all images inside a dataset folder into grayscale
format and saves the results into a separate output directory.

It is useful for preprocessing image datasets in tasks such as
classification, segmentation, anomaly detection, or any workflow where
color information is not required. Converting images to grayscale can
reduce noise, minimize storage size, and simplify feature extraction 
for traditional or deep learning pipelines.

Features:
---------
✅ Automatically scans and loads all images from the specified input folder
✅ Supports multiple image extensions (JPG, PNG, JPEG)
✅ Converts each image to grayscale using OpenCV's color conversion
✅ Maintains original filenames in the output folder
✅ Skips unreadable or corrupted images automatically
✅ Ensures the output directory is created before saving results

Usage:
------
1. Place all images inside the input dataset folder.
2. Set the path of the input and output directories.
3. Run the function:
       convert_dataset_to_grayscale(input_folder, output_folder)
4. All grayscale images will be saved in the output directory.

Example:
--------
from grayscale_converter import convert_dataset_to_grayscale

input_dir = "../datasets/base_dataset/001_cropped/abnormal"
output_dir = "../datasets/base_dataset/003_grayed/abnormal"

convert_dataset_to_grayscale(input_dir, output_dir)

Input Folder Structure:
-----------------------
input_root/
 ├── img001.jpg
 ├── img002.png
 ├── img003.jpeg
 └── ...

Output Folder Structure:
------------------------
output_root/
 ├── img001.jpg      (grayscale)
 ├── img002.png      (grayscale)
 ├── img003.jpeg     (grayscale)
 └── ...

Notes:
------
- The script uses OpenCV (cv2) to perform grayscale conversion.
- If the script encounters an unreadable image, it will skip the file
  automatically and continue processing.
- Output images retain the same filenames as the original ones.

Author:
-------
Fillipus Aditya Nugroho
============================================================
"""

import cv2
import os
import glob

def convert_dataset_to_grayscale(input_folder, output_folder, extensions=("*.jpg", "*.png", "*.jpeg")):
    """
    Convert all images in the dataset to grayscale and save them to the output folder.
    
    Args:
        input_folder (str): Path to the input dataset folder.
        output_folder (str): Path to the output folder for saving grayscale images.
        extensions (tuple): Image extensions to process.
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Collect all image files
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    print(f"Total images found: {len(image_paths)}")

    for idx, img_path in enumerate(image_paths, start=1):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Unable to read: {img_path}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create output filename
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)

        # Save grayscale image
        cv2.imwrite(save_path, gray)

        print(f"[{idx}/{len(image_paths)}] Saved: {save_path}")

    print("Processing completed")

# Example usage
input_dataset = ""
output_dataset = ""
convert_dataset_to_grayscale(input_dataset, output_dataset)
