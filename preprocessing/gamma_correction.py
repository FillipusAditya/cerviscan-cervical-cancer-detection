# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from glob import glob
import pandas as pd
from tqdm import tqdm

def gamma_correction(image, gamma):
    """Apply gamma correction to a grayscale image."""
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    corrected_img = np.uint8(corrected * 255)
    return corrected_img

def normalize_gamma_name(gamma):
    """Normalize gamma value for folder naming (e.g., 0.5 -> 0_5)."""
    return str(gamma).replace(".", "_")

def process_images(input_folder, output_folder, gamma_values):
    """Apply gamma correction and save results in gamma_{value} folders."""
    image_files = glob(os.path.join(input_folder, "*.*"))
    
    if not image_files:
        print("⚠️ No images found in the folder:", input_folder)
        return

    print("📂 Found {} images in '{}'".format(len(image_files), input_folder))

    for gamma in gamma_values:
        gamma_name = normalize_gamma_name(gamma)
        gamma_folder = os.path.join(output_folder, "gamma_{}".format(gamma_name))
        os.makedirs(gamma_folder, exist_ok=True)

        print("\n🔹 Processing gamma = {} → Saving to '{}'".format(gamma, gamma_folder))

        for img_path in tqdm(image_files, desc="Gamma {}".format(gamma), unit="img"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("   ⚠️ Skipping file (not an image): {}".format(img_path))
                continue

            filename = os.path.basename(img_path)

            # Apply gamma correction
            corrected_img = gamma_correction(img, gamma)

            # Save image in gamma folder with original name
            output_path = os.path.join(gamma_folder, filename)
            cv2.imwrite(output_path, corrected_img)

        print("✅ Completed gamma = {}, results saved in '{}'".format(gamma, gamma_folder))


if __name__ == "__main__":
    input_folder = "../datasets/filtered_dataset/gray_world_white_balance"                            
    output_folder = "../datasets/filtered_dataset/gamma_correction_n_gray_world_white_balance"         

    # Generate gamma values: 0.5 → 8.0 with step 0.5
    gamma_values = [round(x * 0.5, 1) for x in range(1, 17)]  

    print("Gamma values to process:", gamma_values)

    process_images(input_folder, output_folder, gamma_values)
