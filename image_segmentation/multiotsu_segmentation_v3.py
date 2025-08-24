"""
Multi-Otsu Image Segmentation Pipeline

This program performs image segmentation using the Multi-Otsu method.
The processing steps for each image are:

1. White Balance using Gray-World assumption
2. Convert RGB to LAB color space
3. Normalize LAB image
4. Extract the L channel
5. Apply CLAHE on the L channel
6. Perform Gamma Correction with multiple gamma values
7. Apply Multi-Otsu Segmentation
8. Save results in corresponding folders:
    - Normalization
    - Channel L
    - Gamma Correction (with subfolders per gamma value)
    - Masking (with subfolders per gamma value)
    - Final Segmentation (with subfolders per gamma value)
"""

import cv2 as cv
import numpy as np
import os
import glob
from tqdm import tqdm
from skimage.filters import threshold_multiotsu

# ---------- Utility Functions ----------
def gray_world_white_balance(img):
    """
    White balance using gray-world assumption.
    """
    result = img.copy().astype(np.float32)
    mean_b, mean_g, mean_r = cv.mean(result)[:3]
    mean_gray = (mean_b + mean_g + mean_r) / 3

    result[:, :, 0] = np.clip(result[:, :, 0] * (mean_gray / mean_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (mean_gray / mean_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (mean_gray / mean_r), 0, 255)

    return result.astype(np.uint8)

def normalize_image(img):
    """
    Normalize image to [0, 255].
    """
    norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    return norm.astype(np.uint8)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on grayscale image.
    """
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def gamma_correction(img, gamma):
    """
    Apply gamma correction.
    """
    img_float = img.astype(np.float32) / 255.0
    corrected = np.power(img_float, gamma)
    return np.uint8(corrected * 255)

# ---------- Pipeline ----------
def multi_otsu_pipeline():
    root_dir = '../datasets/base_dataset/og'      # Input directory
    output_dir = '../datasets/multiotsu_v3'    # Output root directory
    os.makedirs(output_dir, exist_ok=True)

    norm_dir = os.path.join(output_dir, "Normalization")
    chL_dir = os.path.join(output_dir, "Channel_L")
    gamma_dir = os.path.join(output_dir, "Gamma_Correction")
    mask_dir = os.path.join(output_dir, "Masking")
    seg_dir = os.path.join(output_dir, "Segmentation")

    for d in [norm_dir, chL_dir, gamma_dir, mask_dir, seg_dir]:
        os.makedirs(d, exist_ok=True)

    gammas = [0.5, 1.0, 2.0, 5.0, 8.0]

    # Collect all images
    all_imgs = glob.glob(os.path.join(root_dir, "*"))
    progress_bar = tqdm(all_imgs, desc="Multi-Otsu Pipeline", ncols=100)

    for img_path in all_imgs:
        img_id = os.path.basename(img_path)
        rgb_img = cv.imread(img_path)
        if rgb_img is None:
            print(f"[Warning] Failed to load {img_path}")
            continue

        # --- 1. White Balance ---
        wb_img = gray_world_white_balance(rgb_img)

        # --- 2. Convert RGB to LAB ---
        lab_img = cv.cvtColor(wb_img, cv.COLOR_BGR2LAB)

        # --- 3. Normalize LAB ---
        lab_norm = normalize_image(lab_img)
        cv.imwrite(os.path.join(norm_dir, img_id), lab_norm)

        # --- 4. Extract channel L ---
        L, A, B = cv.split(lab_norm)
        cv.imwrite(os.path.join(chL_dir, img_id), L)

        # --- 5. CLAHE on L channel ---
        L_clahe = apply_clahe(L)

        # --- 6. Gamma Correction for multiple gamma values ---
        for g in gammas:
            g_str = str(g).replace(".", "_")
            gamma_subdir = os.path.join(gamma_dir, f"gamma_{g_str}")
            mask_subdir = os.path.join(mask_dir, f"gamma_{g_str}")
            seg_subdir = os.path.join(seg_dir, f"gamma_{g_str}")
            os.makedirs(gamma_subdir, exist_ok=True)
            os.makedirs(mask_subdir, exist_ok=True)
            os.makedirs(seg_subdir, exist_ok=True)

            # Gamma correction
            L_gamma = gamma_correction(L_clahe, g)
            cv.imwrite(os.path.join(gamma_subdir, img_id), L_gamma)

            # --- 7. Multi-Otsu Segmentation ---
            thresholds = threshold_multiotsu(L_gamma, classes=3)
            regions = np.digitize(L_gamma, bins=thresholds)

            # Create mask: select region with highest intensity
            mask = np.zeros_like(regions, dtype=np.uint8)
            mask[regions == regions.max()] = 255
            cv.imwrite(os.path.join(mask_subdir, img_id), mask)

            # Apply mask to original RGB image
            mask_3ch = cv.merge([mask, mask, mask])
            segmented = cv.bitwise_and(rgb_img, mask_3ch)
            cv.imwrite(os.path.join(seg_subdir, img_id), segmented)

        progress_bar.update(1)

    progress_bar.close()

if __name__ == "__main__":
    multi_otsu_pipeline()
