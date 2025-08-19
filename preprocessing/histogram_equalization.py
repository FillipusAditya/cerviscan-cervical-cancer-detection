import cv2
import os
from glob import glob
from tqdm import tqdm

def equalize_grayscale_images(input_root, output_root, image_ext="jpg", blur_kernel=(5, 5)):
    """
    Apply Gaussian blur followed by histogram equalization to all grayscale images
    in a dataset with class subdirectories.

    Parameters:
        input_root (str): Root directory containing subfolders (e.g., 'abnormal', 'normal') with images.
        output_root (str): Directory where processed images will be saved, preserving folder structure.
        image_ext (str): Extension of image files to process (default: 'jpg').
        blur_kernel (tuple): Kernel size for Gaussian blur (default: (5, 5)).

    Returns:
        None
    """
    # Get all image paths recursively inside subfolders
    pattern = os.path.join(input_root, "*", f"*.{image_ext}")
    image_paths = glob(pattern)
    total_images = len(image_paths)

    print(f"Processing {total_images} images with Gaussian blur and histogram equalization...")

    progress_bar = tqdm(image_paths, desc="Initializing", unit="image")
    for idx, path in enumerate(progress_bar, start=1):
        # Update dynamic description with image count
        progress_bar.set_description(f"Processing image {idx}/{total_images}")

        # Read the image in grayscale mode
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image at {path}. Skipping.")
            continue

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, blur_kernel, 0)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(blurred)

        # Generate relative path from input root to preserve structure
        rel_path = os.path.relpath(path, input_root)  # e.g., 'abnormal/img1.jpg'
        output_path = os.path.join(output_root, rel_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the processed image
        success = cv2.imwrite(output_path, equalized)
        if not success:
            print(f"Warning: Failed to save image at {output_path}.")

    print("Processing complete.")

if __name__ == "__main__":
    # Define paths
    input_dataset_dir = "../datasets/dataset_g/cropped_image_after_iva_reference_4/seperated"
    output_dataset_dir = "../datasets/dataset_g/cropped_image_after_iva_reference_4/equalized"

    # Run the processing function
    equalize_grayscale_images(input_dataset_dir, output_dataset_dir, blur_kernel = (7, 7))
