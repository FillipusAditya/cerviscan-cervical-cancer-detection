import cv2 as cv
import numpy as np
import os
import glob
from tqdm import tqdm
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt

def multi_otsu_segmentation(input_root: str, output_root: str, classes: int = 5) -> None:
    """
    Apply Multi-Otsu thresholding segmentation on grayscale images in a folder.

    Parameters:
        input_root (str): Directory containing images.
        output_root (str): Directory where binary masks will be saved.
        classes (int): Number of classes for thresholding (default: 5).
    """
    # Get all image paths in the root folder (no subdirectories)
    image_paths = glob.glob(os.path.join(input_root, "*"))

    print(f"📂 Found {len(image_paths)} images for segmentation in '{input_root}'")

    for img_path in tqdm(image_paths, desc="Multi-Otsu Segmentation", ncols=100):
        # Read image in grayscale
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Warning: Could not read image: {img_path}")
            continue

        # Apply Multi-Otsu thresholding
        thresholds = threshold_multiotsu(img, classes=classes)
        regions = np.digitize(img, bins=thresholds)

        # Create binary mask of highest intensity region only
        output = (regions * (255 // (regions.max() + 1))).astype(np.uint8)
        output[output < np.unique(output)[-1]] = 0
        output[output >= np.unique(output)[-1]] = 1

        # Prepare save path (preserve file name)
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_root, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save binary mask image
        plt.imsave(save_path, output, cmap='gray')

    print(f"✅ Segmentation complete. Masks saved to: {output_root}")


if __name__ == "__main__":
    # You can manually list multiple folders here
    input_dirs = [
        "../datasets/filtered_dataset/gamma_correction/gamma_5_0",
        "../datasets/filtered_dataset/gamma_correction/gamma_6_0",
        "../datasets/filtered_dataset/gamma_correction/gamma_7_0",
        "../datasets/filtered_dataset/gamma_correction/gamma_8_0",
        "../datasets/filtered_dataset/gamma_correction_n_bilateral_d15_sc75_ss75/gamma_5_0",
        "../datasets/filtered_dataset/gamma_correction_n_bilateral_d15_sc75_ss75/gamma_6_0",
        "../datasets/filtered_dataset/gamma_correction_n_bilateral_d15_sc75_ss75/gamma_7_0",
        "../datasets/filtered_dataset/gamma_correction_n_bilateral_d15_sc75_ss75/gamma_8_0",
        "../datasets/filtered_dataset/gamma_correction_n_gray_world_white_balance/gamma_5_0",
        "../datasets/filtered_dataset/gamma_correction_n_gray_world_white_balance/gamma_6_0",
        "../datasets/filtered_dataset/gamma_correction_n_gray_world_white_balance/gamma_7_0",
        "../datasets/filtered_dataset/gamma_correction_n_gray_world_white_balance/gamma_8_0",
        "../datasets/filtered_dataset/gray_world_white_balance"
    ]

    # Run segmentation for each input folder
    for input_dir in input_dirs:
        # Derive output folder by appending "_mask"
        base_name = os.path.basename(input_dir.rstrip("/"))
        parent_dir = os.path.dirname(input_dir)
        output_dir = os.path.join(parent_dir, f"{base_name}_mask")

        multi_otsu_segmentation(input_dir, output_dir, classes=5)
