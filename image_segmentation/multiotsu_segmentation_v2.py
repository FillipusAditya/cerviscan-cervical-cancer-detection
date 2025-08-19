import cv2 as cv
import numpy as np
import os
import glob
from tqdm import tqdm
from skimage.filters import threshold_multiotsu

def multiOtsuSegmentation():
    """
    Perform Multi-Otsu segmentation on RGB images:
        - Convert RGB to grayscale
        - Apply Multi-Otsu thresholding
        - Generate binary mask for the highest intensity region
        - Apply mask to original RGB image (bitwise segmented image)
        - Save segmented result into the output directory
    """
    # Path configuration
    root_dir = '../datasets/dataset_g/cropped_image_after_iva_reference_4/seperated'  # Input RGB images
    output_dir = '../datasets/dataset_g/cropped_image_after_iva_reference_4/segmented'  # Output segmented RGB images
    os.makedirs(output_dir, exist_ok=True)

    sub_img_dirs = glob.glob(os.path.join(root_dir, '*'))

    # Step 1: Collect all image paths to calculate total
    all_img_paths = []
    for img_dir in sub_img_dirs:
        all_img_paths.extend(glob.glob(os.path.join(img_dir, '*')))

    total_images = len(all_img_paths)
    progress_bar = tqdm(total=total_images, desc="Multi-Otsu RGB Segmentation", ncols=100)

    # Step 2: Iterate over all image paths
    for img_path in all_img_paths:
        label = os.path.basename(os.path.dirname(img_path))
        img_id = os.path.basename(img_path)
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        # Load RGB image
        rgb_img = cv.imread(img_path)
        if rgb_img is None:
            print(f"Warning: Failed to load image {img_path}")
            progress_bar.update(1)
            continue
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)

        # Convert to grayscale
        gray_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)

        # Apply Multi-Otsu thresholding
        thresholds = threshold_multiotsu(gray_img, classes=5)
        regions = np.digitize(gray_img, bins=thresholds)

        # Create binary mask for the highest intensity region
        mask = np.zeros_like(regions, dtype=np.uint8)
        mask[regions == regions.max()] = 1

        # Convert mask to 3 channels
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Generate segmented image using the mask
        segmented = rgb_img * mask_3ch

        # Save the segmented image (convert back to BGR for OpenCV)
        save_path = os.path.join(save_dir, img_id)
        segmented_bgr = cv.cvtColor(segmented, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, segmented_bgr)

        # Update progress bar with index information
        current = progress_bar.n + 1
        progress_bar.set_postfix_str(f"{current}/{total_images}")
        progress_bar.update(1)

    progress_bar.close()

if __name__ == "__main__":
    multiOtsuSegmentation()
