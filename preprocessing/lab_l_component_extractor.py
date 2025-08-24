import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def extract_l_channel_from_lab(image_path: str, output_path: str) -> None:
    """
    Extracts the L (Lightness) component from the LAB color space of an image,
    normalizes it to the range [0, 255], and saves the result.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    output_path : str
        Path to save the processed L channel image.
    """
    # Read the input image
    img = cv2.imread(image_path)

    # Skip if the image cannot be loaded
    if img is None:
        print(f"[WARNING] Failed to load image: {image_path}. Skipping...")
        return

    # Convert the image from BGR (default in OpenCV) to LAB color space
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB into L, A, B channels and keep only the L channel
    l_channel, _, _ = cv2.split(image_lab)

    # Normalize the L channel to ensure intensity values range from 0 to 255
    l_channel_normalized = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)

    # Save the processed image
    cv2.imwrite(output_path, l_channel_normalized)


def process_images(input_folder: str, output_folder: str) -> None:
    """
    Processes all images in the input folder by extracting and saving
    the L component from LAB color space.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing input images.
    output_folder : str
        Path to the folder where processed images will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Collect all image file paths in the input folder
    image_files = glob.glob(os.path.join(input_folder, "*"))

    # Exit if no images are found
    if not image_files:
        print(f"[ERROR] No images found in: {input_folder}")
        return

    # Process each image with a progress bar
    for image_file in tqdm(image_files, desc="Extracting L Component", unit="file", ncols=100):
        filename = os.path.basename(image_file)
        output_path = os.path.join(output_folder, filename)
        extract_l_channel_from_lab(image_file, output_path)


if __name__ == "__main__":
    # Define input and output directories
    input_folder_path = "../image/cropped_image_after_iva"
    output_folder_path = "../image/L_Component_From_LAB"

    # Run the processing function
    process_images(input_folder_path, output_folder_path)
