import cv2 as cv
import numpy as np
import os
import glob
from tqdm import tqdm
from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

def multiOtsuSegmentation():
    """
    Perform Multi-Otsu thresholding segmentation on a set of grayscale images.

    Steps:
        - Load grayscale images from a root directory organized by label/class.
        - Apply Multi-Otsu thresholding with a predefined number of classes.
        - Create a binary mask for the region with the highest intensity.
        - Save the generated binary mask for each image.

    Directories:
        - root_dir: Source images, grouped by label.
        - output_dir: Destination to save binary mask images, preserving label structure.
    """
    # Path configuration
    root_dir = '../datasets/dataset_j/001_eq_bilateral_blur_images'
    output_dir = '../datasets/dataset_j/002_mask_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories (labels)
    sub_img_dirs = glob.glob(os.path.join(root_dir, '*'))

    # Process each subdirectory
    for img_dir in tqdm(sub_img_dirs, desc="Multi-Otsu Segmentation", ncols=100):
        label = os.path.basename(img_dir)
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        # Process each image in the subdirectory
        for img_path in glob.glob(os.path.join(img_dir, '*')):
            img_id = os.path.basename(img_path)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            # Apply Multi-Otsu thresholding
            thresholds = threshold_multiotsu(img, classes=5)
            regions = np.digitize(img, bins=thresholds)

            # Convert regions to binary mask
            output = (regions * (255 // (regions.max() + 1))).astype(np.uint8)
            # Keep only the region with highest intensity, set others to 0
            output[output < np.unique(output)[-1]] = 0
            output[output >= np.unique(output)[-1]] = 1

            # Save binary mask
            save_path = os.path.join(save_dir, img_id)
            plt.imsave(save_path, output, cmap='gray')
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    multiOtsuSegmentation()
