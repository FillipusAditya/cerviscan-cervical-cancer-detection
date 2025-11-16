"""
============================================================
📌 Bitwise Segmentation Using Pre-Generated Masks (Python)
============================================================

Description:
------------
This program performs image segmentation using bitwise AND 
between original images and their corresponding binary masks.
It is designed for datasets with a directory structure organized 
by class labels (e.g., 'normal', 'abnormal', or other categories).

The segmentation process works by applying a mask image onto the 
original cropped image to isolate the region of interest (ROI). 
If the mask and original image have different sizes, the mask will 
automatically be resized to match the original image.

Bitwise segmentation is widely used in medical imaging, lesion 
extraction, preprocessing for feature extraction, and cleaning 
background noise before feeding images into ML/DL models.

Features:
---------
✅ Automatically scans all class subdirectories  
✅ Ensures mask and image are matched using identical filenames  
✅ Automatically resizes masks when dimension mismatch occurs  
✅ Applies bitwise AND to isolate the masked region  
✅ Preserves directory structure when saving segmented outputs  
✅ Works for any image dataset with mask pairs  

Usage:
------
1. Place your cropped input images inside the root image directory
   (e.g., root_dir/normal/, root_dir/abnormal/).

2. Place mask images with the **same filenames** in the mask directory:
       mask_dir/normal/, mask_dir/abnormal/, etc.

3. Call the segmentation function:
       bitwiseSegmentation(root_dir, mask_dir, output_dir)

4. Segmented images will be saved under the output directory, 
   maintaining the same class/subfolder structure.

Example:
--------
from segmentation_bitwise import bitwiseSegmentation

root_dir   = "../datasets/cropped_images"
mask_dir   = "../datasets/masks"
output_dir = "../datasets/segmented_output"

bitwiseSegmentation(root_dir, mask_dir, output_dir)

Directory Structure:
--------------------
root_dir/
 ├── classA/
 │    ├── img1.jpg
 │    ├── img2.jpg
 └── classB/
      ├── img3.jpg
      ├── img4.jpg

mask_dir/
 ├── classA/
 │    ├── img1.jpg
 │    ├── img2.jpg
 └── classB/
      ├── img3.jpg
      ├── img4.jpg

output_dir/
 ├── classA/
 │    ├── img1.jpg   (segmented)
 │    ├── img2.jpg   (segmented)
 └── classB/
      ├── img3.jpg   (segmented)
      ├── img4.jpg   (segmented)

Notes:
------
- Mask images should be binary (0/255) for best results.
- Nonexistent masks are skipped automatically.
- Image and mask filenames must match.

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

def bitwiseSegmentation(root_dir, mask_dir, output_dir):
    """
    Perform bitwise segmentation by applying pre-generated mask to original images.

    Parameters:
        root_dir (str): Path to directory containing cropped input images organized by class.
        mask_dir (str): Path to directory containing mask images with identical filenames.
        output_dir (str): Path to directory where segmented images will be saved.

    Description:
        - Iterates through class folders inside root_dir.
        - For each image, finds the corresponding mask with the same filename.
        - Resizes mask if its dimensions do not match the original image.
        - Converts the mask into 3 channels and applies bitwise AND.
        - Saves the segmented image in output_dir/<class_name>/.

    Notes:
        - Mask filenames must match the input image filenames.
        - Mask will be resized to match image dimensions if necessary.
    """

    # Get list of class folders under root_dir
    sub_img_dirs = glob.glob(os.path.join(root_dir, '*'))

    for img_dir in tqdm(sub_img_dirs, desc="Bitwise Segmentation", ncols=100):
        label = os.path.basename(img_dir)

        # Prepare save directory for segmented images
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        # Corresponding mask folder for this class
        mask_label_dir = os.path.join(mask_dir, label)

        # Process all images inside this class folder
        for img_path in glob.glob(os.path.join(img_dir, '*')):
            img_id = os.path.basename(img_path)

            # Read original image
            original_image = cv.imread(img_path)
            if original_image is None:
                print(f"Unable to read image: {img_path}")
                continue

            # Expected mask path
            mask_path = os.path.join(mask_label_dir, img_id)

            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask not found for {img_id}")
                continue

            # Read mask in grayscale
            mask_image = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if mask_image is None:
                print(f"Unable to read mask: {mask_path}")
                continue

            # Resize if mismatch
            if mask_image.shape[:2] != original_image.shape[:2]:
                mask_image = cv.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

            # Convert to 3-channel mask
            mask_image_3ch = cv.cvtColor(mask_image, cv.COLOR_GRAY2BGR)

            # Perform segmentation
            segmented_image = cv.bitwise_and(original_image, mask_image_3ch)

            # Save output
            save_path = os.path.join(save_dir, img_id)
            cv.imwrite(save_path, segmented_image)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    bitwiseSegmentation(
        root_dir="",
        mask_dir="",
        output_dir=""
    )
