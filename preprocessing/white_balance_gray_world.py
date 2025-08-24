# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm


def gray_world_white_balance(img):
    """
    Apply Gray World White Balance to an input image.
    """
    # Convert to float32 for precision
    result = img.astype(np.float32)
    
    # Compute mean for each channel
    mean_b, mean_g, mean_r = cv2.mean(result)[:3]
    mean_gray = (mean_b + mean_g + mean_r) / 3
    
    # Scale each channel
    result[:, :, 0] = np.minimum(result[:, :, 0] * (mean_gray / mean_b), 255)
    result[:, :, 1] = np.minimum(result[:, :, 1] * (mean_gray / mean_g), 255)
    result[:, :, 2] = np.minimum(result[:, :, 2] * (mean_gray / mean_r), 255)
    
    return result.astype(np.uint8)


def process_folder(input_folder, output_folder, extensions=("*.jpg", "*.png", "*.jpeg")):
    """
    Apply gray world white balance to all images in a folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect image paths
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))
    
    print(f"📂 Found {len(image_paths)} images in '{input_folder}'")

    # Process each image with tqdm progress bar
    for img_path in tqdm(image_paths, desc="Processing Images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue
        
        # Apply gray world white balance
        balanced_img = gray_world_white_balance(img)
        
        # Save to output folder with same filename
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, balanced_img)

    print(f"✅ Processing completed. Results saved in '{output_folder}'")


if __name__ == "__main__":
    # Example usage
    input_folder = "../datasets/base_dataset/og"   
    output_folder = "../datasets/filtered_dataset/gray_world_white_balance" 
    process_folder(input_folder, output_folder)