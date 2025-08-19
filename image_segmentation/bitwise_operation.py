import cv2 as cv
import numpy as np
import os
import glob
from tqdm import tqdm

def bitwiseSegmentation():
    """
    Perform bitwise segmentation by applying a pre-generated mask to original images.

    This function reads cropped images and their corresponding masks,
    applies a bitwise AND operation to segment the region of interest,
    and saves the segmented images to the output directory.

    Directory structure:
        - root_dir: Contains cropped input images organized by label/class.
        - mask_dir: Contains mask images matching the input images.
        - output_dir: Destination for saving the segmented output images.

    Note:
        - If the mask image dimensions do not match the original image, the mask is resized.
        - Each segmented image is saved with the same filename as the original.
    """
    # Path to the root directory containing cropped input images
    root_dir = '../datasets/dataset_g/cropped_image_after_iva_reference_4/seperated'
    
    # Path to the directory containing mask images
    mask_dir = '../datasets/dataset_g/cropped_image_after_iva_reference_4/equalized_mask'
    
    # Path to the output directory where segmented images will be saved
    output_dir = '../datasets/dataset_g/cropped_image_after_iva_reference_4/equalized_segmented'
    
    # Get a list of subdirectories (each representing a label/class)
    sub_img_dirs = glob.glob(os.path.join(root_dir, '*'))

    # Iterate through each label folder
    for img_dir in tqdm(sub_img_dirs, desc="Bitwise Segmentation", ncols=100):
        label = os.path.basename(img_dir)

        # Define the save directory for the segmented output of the current label
        save_dir = os.path.join(output_dir, label)

        # Define the corresponding mask folder for the current label
        mask_label_dir = os.path.join(mask_dir, label)
        
        # Create output directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        # Process each image in the current label folder
        for img_path in glob.glob(os.path.join(img_dir, '*')):
            img_id = os.path.basename(img_path)

            # Read the original input image
            original_image = cv.imread(img_path)

            # Build the expected path for the corresponding mask image
            mask_path = os.path.join(mask_label_dir, img_id)

            # Check if the mask exists
            if not os.path.exists(mask_path):
                print(f"Mask not found for: {img_path}")
                continue

            # Read the mask image as grayscale
            mask_image = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

            # If the dimensions don't match, resize the mask to match the original image
            if mask_image.shape[:2] != original_image.shape[:2]:
                mask_image = cv.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

            # Convert the single-channel mask to 3 channels to match the original image
            mask_image_3channel = cv.cvtColor(mask_image, cv.COLOR_GRAY2BGR)

            # Apply bitwise AND to segment the image
            segmented_image = cv.bitwise_and(original_image, mask_image_3channel)

            # Build the save path for the segmented output image
            save_path = os.path.join(save_dir, img_id)

            # Save the segmented image
            cv.imwrite(save_path, segmented_image)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Run the bitwise segmentation process when the script is executed
    bitwiseSegmentation()
