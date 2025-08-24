# -*- coding: utf-8 -*-
import cv2
import os
from glob import glob
from tqdm import tqdm

def bilateral_blur(img, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral blur to an image.
    """
    result = cv2.bilateralFilter(img, d=diameter,
                                 sigmaColor=sigma_color,
                                 sigmaSpace=sigma_space)
    return result


def process_folder(input_folder, output_folder=None,
                   extensions=("*.jpg", "*.png", "*.jpeg"),
                   diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral blur to all images in a folder.
    """
    # Auto-generate folder name based on parameters
    param_folder = f"output_bilateral_d{diameter}_sc{sigma_color}_ss{sigma_space}"
    
    # If user provides base folder, combine it
    if output_folder is not None:
        output_folder = os.path.join(output_folder, param_folder)
    else:
        output_folder = param_folder
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect image paths
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    print(f"📂 Found {len(image_paths)} images in '{input_folder}'")
    print(f"📂 Results will be saved in '{output_folder}'")

    # Process each image with tqdm progress bar
    for img_path in tqdm(image_paths, desc="Applying Bilateral Blur"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        # Apply bilateral blur
        blurred_img = bilateral_blur(img, diameter, sigma_color, sigma_space)

        # Save to output folder with same filename
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, blurred_img)

    print(f"✅ Processing completed. Results saved in '{output_folder}'")


if __name__ == "__main__":
    input_folder = "../datasets/base_dataset/og"
    output_folder = "../datasets/filtered_dataset/bilateral_blur"
    
    # Bilateral filter parameters
    d = 15
    sc = 75
    ss = 75

    process_folder(input_folder, output_folder=output_folder,
                   diameter=d, sigma_color=sc, sigma_space=ss)
