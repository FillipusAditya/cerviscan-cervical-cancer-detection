# -*- coding: utf-8 -*-
"""
Multi-Otsu Segmentation with Bilateral Blur + Gamma Correction
--------------------------------------------------------------

Workflow:
    1. Input images are read from a flat directory (no subfolders).
    2. Each image is processed with multiple gamma correction values.
    3. For each gamma value:
        - Convert BGR → RGB.
        - Convert RGB → Grayscale.
        - Apply Bilateral Blur.
        - Apply Gamma Correction.
        - Apply Multi-Otsu Thresholding (classes=5).
        - Create a binary mask of the highest intensity region.
        - Multiply the mask with the original RGB image to obtain segmentation.
        - Save results in: {output_root}/output_bilateral_d{diameter}_sc{sigma_color}_ss{sigma_space}_gamma_{gamma}
"""

import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from skimage.filters import threshold_multiotsu

# ---------- Utility Functions ----------
def gamma_correction(image, gamma):
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    return np.uint8(corrected * 255)


def bilateral_blur(img, diameter=9, sigma_color=75, sigma_space=75):
    return cv.bilateralFilter(img, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def normalize_gamma_name(gamma):
    return str(gamma).replace(".", "_")


# ---------- Main Pipeline ----------
def process_images(input_dir, output_root, gamma_values, diameter=9, sigma_color=75, sigma_space=75):
    os.makedirs(output_root, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for gamma in gamma_values:
        gamma_str = normalize_gamma_name(gamma)
        output_dir = os.path.join(
            output_root, f"output_bilateral_d{diameter}_sc{sigma_color}_ss{sigma_space}_gamma_{gamma_str}"
        )
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nℹ️ Processing Gamma={gamma}")
        print(f"ℹ️ Output directory: {output_dir}")

        for file in tqdm(image_files, desc=f"Gamma {gamma}", unit="image"):
            img_path = os.path.join(input_dir, file)

            # Step 3: Read image → BGR → RGB
            img_bgr = cv.imread(img_path)
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

            # Step 4: Convert to grayscale
            gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

            # Step 5: Bilateral Blur
            gray_blur = bilateral_blur(gray, diameter, sigma_color, sigma_space)

            # Step 6: Gamma Correction
            gray_corrected = gamma_correction(gray_blur, gamma)

            # Step 7: Multi-Otsu thresholding (classes=5)
            thresholds = threshold_multiotsu(gray_corrected, classes=5)

            # Step 8: Assign pixels into regions
            regions = np.digitize(gray_corrected, bins=thresholds)

            # Step 9: Select highest intensity region (binary mask)
            mask = (regions == regions.max()).astype(np.uint8) * 255

            # Step 10: Expand mask and segment image
            mask_rgb = cv.merge([mask, mask, mask])
            segmented = cv.bitwise_and(img_rgb, mask_rgb)

            # Step 11: Save segmented image
            save_path = os.path.join(output_dir, file)
            cv.imwrite(save_path, cv.cvtColor(segmented, cv.COLOR_RGB2BGR))


# ---------- Run ----------
if __name__ == "__main__":
    input_directory = "../datasets/base_dataset/og"
    output_root = "../datasets/filtered_dataset"
    gamma_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]

    process_images(
        input_directory,
        output_root,
        gamma_values,
        diameter=15,
        sigma_color=75,
        sigma_space=75
    )
