import cv2
import os
import glob

def convert_dataset_to_grayscale(input_folder, output_folder, extensions=("*.jpg", "*.png", "*.jpeg")):
    """
    Convert all images in dataset to grayscale and save to output folder.
    
    Args:
        input_folder (str): Path ke folder dataset input.
        output_folder (str): Path ke folder output untuk menyimpan gambar grayscale.
        extensions (tuple): Ekstensi gambar yang akan diproses.
    """
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # Ambil semua file gambar
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    print(f"Jumlah gambar ditemukan: {len(image_paths)}")

    for idx, img_path in enumerate(image_paths, start=1):
        # Baca gambar
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Tidak bisa membaca: {img_path}")
            continue

        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Buat nama file output
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)

        # Simpan gambar grayscale
        cv2.imwrite(save_path, gray)

        print(f"[{idx}/{len(image_paths)}] Disimpan: {save_path}")

    print("Proses selesai ✅")

# Contoh penggunaan
input_dataset = "../datasets/base_dataset/001_cropped/abnormal"   # ganti dengan folder dataset asli
output_dataset = "../datasets/base_dataset/003_grayed/abnormal"   # folder hasil grayscale
convert_dataset_to_grayscale(input_dataset, output_dataset)
