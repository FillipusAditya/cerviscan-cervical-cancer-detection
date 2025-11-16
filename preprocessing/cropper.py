"""
============================================================
📌 Crop Images Using Manual ROI Selection (Python)
============================================================

Description:
------------
This program allows users to crop multiple images manually by selecting
a Region of Interest (ROI) on a resized preview of each image. The ROI 
selected on the resized image is automatically mapped back to the original 
resolution to ensure accurate cropping.

This implementation is useful for preprocessing image datasets where 
manual supervision is required, such as medical imaging, object 
localization, dataset cleaning, or preparing training data for 
classification/segmentation models.

Features:
---------
✅ Automatically scans and loads all image files from a specified folder
✅ Supports multiple image file extensions (JPG, PNG, JPEG)
✅ Resizes images proportionally to fit within a specified percentage of 
  the screen size for easier ROI selection
✅ Allows manual ROI selection using OpenCV's interactive interface
✅ Maps coordinates back to the original image resolution for precise cropping
✅ Saves all cropped outputs to a separate output directory while retaining filenames

Usage:
------
1. Place all images you want to crop inside the input folder path.
2. Set the desired maximum display ratio (default: 0.9 for 90% of screen size).
3. Run the function:
       crop_images(input_folder, output_folder, max_display_ratio=0.8)
4. A window will appear for each image, allowing you to manually select an ROI.

Example:
--------
from crop_tool import crop_images

input_dir = "path/to/raw_images"
output_dir = "path/to/cropped_images"

crop_images(input_dir, output_dir, max_display_ratio=0.8)

Input Folder Structure:
-----------------------
input_root/
 ├── img1.jpg
 ├── img2.png
 ├── img3.jpeg
 └── ...

Output Folder Structure:
------------------------
output_root/
 ├── img1.jpg    (cropped)
 ├── img2.png    (cropped)
 ├── img3.jpeg   (cropped)
 └── ...

Notes:
------
- If the ROI selection is cancelled (width or height = 0), the image 
  will be skipped automatically.
- Screen resolution is detected automatically using tkinter for 
  cross-platform support.
- ROI selection uses OpenCV’s interactive tool; press ENTER to confirm 
  or C to cancel.

Author:
-------
Fillipus Aditya Nugroho
============================================================
"""

import cv2
import os
import glob

def crop_images(folder_path, output_folder_path, max_display_ratio=0.9, extensions=("*.jpg", "*.png", "*.jpeg")):
    """
    Crop multiple images from a folder using manual ROI selection on resized previews.

    Args:
        folder_path (str): Path to the input images.
        output_folder_path (str): Path to save cropped images.
        max_display_ratio (float): Maximum display ratio relative to the screen size (default: 0.9 -> 90%).
        extensions (tuple): Tuple of allowed file extensions (default: jpg, png, jpeg).
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Collect image files with the specified extensions
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    # If no images found
    if not image_files:
        print("No image files found in the input folder.")
        return

    # Get screen resolution (using tkinter for cross-platform support)
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        screen_width, screen_height = 1280, 720  # fallback default resolution

    success_count = 0

    for idx, image_file in enumerate(image_files, 1):
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to read image: {image_file}")
            continue

        original_height, original_width = image.shape[:2]

        # Determine scale factor so the image fits within the screen
        scale_w = (screen_width * max_display_ratio) / original_width
        scale_h = (screen_height * max_display_ratio) / original_height
        scale = min(scale_w, scale_h)

        display_width = int(original_width * scale)
        display_height = int(original_height * scale)

        # Resize image for ROI selection
        resized_image = cv2.resize(image, (display_width, display_height))

        # User selects ROI
        print(f"[{idx}/{len(image_files)}] Select ROI for image: {os.path.basename(image_file)}")
        r = cv2.selectROI("Select ROI and press ENTER (or C to cancel)", resized_image, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        # If ROI is not selected
        if r[2] == 0 or r[3] == 0:
            print(f"ROI cancelled for image: {os.path.basename(image_file)}")
            continue

        # Convert ROI coordinates back to original scale
        x1 = int(r[0] / scale)
        y1 = int(r[1] / scale)
        x2 = int((r[0] + r[2]) / scale)
        y2 = int((r[1] + r[3]) / scale)

        # Crop original image
        cropped_image = image[y1:y2, x1:x2]

        # Save cropped image
        filename = os.path.basename(image_file)
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, cropped_image)

        print(f"Cropped image saved to: {output_path}")
        success_count += 1

    print(f"\nFinished. {success_count}/{len(image_files)} images successfully cropped.")

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    crop_images(input_folder, output_folder, max_display_ratio=0.8)
