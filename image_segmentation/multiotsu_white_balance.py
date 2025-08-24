"""
Multi-Otsu Segmentation with Gray World White Balance
-----------------------------------------------------

Workflow:
    1. Input directory: ../images (only images, no subfolders).
    2. Output directory: ../images/segmented.
    3. Reads image (cv.imread) → BGR → RGB.
    4. Apply Gray World White Balance.
    5. Convert to grayscale.
    6. Multi-Otsu thresholding (classes=5).
    7. Create binary mask (highest intensity region).
    8. Apply mask to RGB image.
    9. Save output with the same filename into output directory.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.filters import threshold_multiotsu

# ---------- Utility Functions ----------
def gray_world_white_balance(img):
    """
    Apply Gray World White Balance to an RGB image.
    """
    result = img.astype(np.float32)

    mean_r, mean_g, mean_b = np.mean(result[:, :, 0]), np.mean(result[:, :, 1]), np.mean(result[:, :, 2])
    mean_gray = (mean_r + mean_g + mean_b) / 3

    result[:, :, 0] = np.minimum(result[:, :, 0] * (mean_gray / mean_r), 255)
    result[:, :, 1] = np.minimum(result[:, :, 1] * (mean_gray / mean_g), 255)
    result[:, :, 2] = np.minimum(result[:, :, 2] * (mean_gray / mean_b), 255)

    return result.astype(np.uint8)

# ---------- Main Pipeline ----------
def process_images(input_dir, output_dir):
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

        # Step 4: Gray World White Balance
        img_balanced = gray_world_white_balance(img_rgb)

        # Step 5: Convert to grayscale
        gray = cv2.cvtColor(img_balanced, cv2.COLOR_RGB2GRAY)

        # Step 6: Multi-Otsu thresholding (classes=5)
        thresholds = threshold_multiotsu(gray, classes=5)

        # Step 7: Assign pixels into regions
        regions = np.digitize(gray, bins=thresholds)

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
    output_directory = "../datasets/filtered_dataset/multiotsu_white_balanced"
    
    process_images(input_directory, output_directory)