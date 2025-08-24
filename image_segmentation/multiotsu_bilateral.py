"""
Multi-Otsu Segmentation with Bilateral Blur
-------------------------------------------

Workflow:
    1. Input directory: ../images (only images, no subfolders).
    2. Output directory: ../my_specified_folder/output_bilateral_d{diameter}_sc{sigma_color}_ss{sigma_space}.
    3. Reads image (cv.imread) → BGR → RGB.
    4. Convert to grayscale.
    5. Apply Bilateral Blur.
    6. Multi-Otsu thresholding (classes=5).
    7. Create binary mask (highest intensity region).
    8. Apply mask to RGB image.
    9. Save output with same filename into output directory.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.filters import threshold_multiotsu

# ---------- Utility Functions ----------
def bilateral_blur(img, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral blur to an image.
    """
    return cv2.bilateralFilter(img, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# ---------- Main Pipeline ----------
def process_images(input_dir, output_root, diameter=9, sigma_color=75, sigma_space=75):
    # Define output directory inside the specified root folder
    output_dir = os.path.join(
        output_root, f"output_bilateral_d{diameter}_sc{sigma_color}_ss{sigma_space}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Collect image files
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"\nℹ️ Processing {len(image_files)} images...")
    for file in tqdm(image_files, desc="Processing", unit="image"):
        img_path = os.path.join(input_dir, file)

        # Step 3: Read image → BGR → RGB
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Step 4: Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Step 5: Bilateral Blur
        gray_blur = bilateral_blur(gray, diameter, sigma_color, sigma_space)

        # Step 6: Multi-Otsu thresholding (classes=5)
        thresholds = threshold_multiotsu(gray_blur, classes=5)

        # Step 7: Assign pixels into regions
        regions = np.digitize(gray_blur, bins=thresholds)

        # Step 8: Select highest intensity region (binary mask)
        mask = (regions == regions.max()).astype(np.uint8) * 255

        # Step 9: Expand mask and segment image
        mask_rgb = cv2.merge([mask, mask, mask])
        segmented = cv2.bitwise_and(img_rgb, mask_rgb)

        # Step 10: Save segmented image
        save_path = os.path.join(output_dir, file)
        cv2.imwrite(save_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))


# ---------- Run ----------
if __name__ == "__main__":
    input_directory = "../datasets/base_dataset/og"
    output_directory = "../datasets/filtered_dataset/multiotsu_bilateral"
    process_images(input_directory, output_directory, diameter=15, sigma_color=75, sigma_space=75)
