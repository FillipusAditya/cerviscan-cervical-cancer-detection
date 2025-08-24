import cv2 as cv
import numpy as np
import os
import glob


def process_image(img_path: str, save_path: str) -> None:
    """
    Reads an image, converts it to grayscale, applies histogram equalization,
    and bilateral filtering, then saves the processed image.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    save_path : str
        Path to save the processed image.
    """
    # Read image in BGR format
    img_bgr = cv.imread(img_path)

    if img_bgr is None:
        print(f"[WARNING] Failed to load image: {img_path}. Skipping...")
        return

    # Convert to grayscale
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Apply histogram equalization (contrast enhancement)
    equa_img = cv.equalizeHist(img_gray)

    # Apply bilateral filter (noise reduction while preserving edges)
    bilateral_img = cv.bilateralFilter(equa_img, d=9, sigmaColor=75, sigmaSpace=75)

    # Save processed image
    cv.imwrite(save_path, bilateral_img)
    print(f"[INFO] Saved: {save_path}")


def process_folder(input_dir: str, output_dir: str) -> None:
    """
    Processes all images in a folder and saves results to the output folder.

    Parameters
    ----------
    input_dir : str
        Path to the input folder containing images.
    output_dir : str
        Path to the output folder where processed images will be stored.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all images in the input directory
    image_paths = glob.glob(os.path.join(input_dir, "*"))

    if not image_paths:
        print(f"[ERROR] No images found in: {input_dir}")
        return

    # Process each image
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, img_name)
        process_image(img_path, save_path)


def main():
    # Define input and output directories
    input_dir = "../datasets/base_dataset_iarc_cropped/001_cropped"
    output_dir = "../datasets/dataset_j/001_eq_images"

    # Process the entire folder
    process_folder(input_dir, output_dir)


if __name__ == "__main__":
    main()
