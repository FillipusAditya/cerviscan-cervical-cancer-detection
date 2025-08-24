import os
from PIL import Image
import numpy as np

def min_max_normalize(image_path, feature_range=(0, 255)):
    """
    Melakukan Min-Max normalization pada sebuah gambar grayscale.
    
    Args:
        image_path (str): Jalur lengkap ke file gambar.
        feature_range (tuple): Rentang output normalisasi (default: (0, 255)).

    Returns:
        numpy.ndarray: Gambar yang sudah dinormalisasi dalam bentuk array NumPy, atau None jika terjadi kesalahan.
    """
    try:
        # Buka gambar dan konversi ke grayscale
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)

        min_val, max_val = np.min(img_array), np.max(img_array)
        if max_val - min_val == 0:
            print(f"[PERINGATAN] Gambar '{os.path.basename(image_path)}' memiliki rentang nol. Normalisasi tidak dilakukan.")
            return img_array

        # Min-Max scaling
        scale_min, scale_max = feature_range
        normalized_img_array = (img_array - min_val) / (max_val - min_val)
        normalized_img_array = normalized_img_array * (scale_max - scale_min) + scale_min

        print(f"[INFO] Gambar '{os.path.basename(image_path)}' berhasil dinormalisasi (Min-Max).")
        return normalized_img_array

    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan: {image_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat memproses gambar '{str(image_path)}': {e}")
        return None


def process_dataset(input_dir, output_dir, feature_range=(0, 255)):
    """
    Memproses seluruh dataset gambar dalam sebuah direktori dengan Min-Max normalization.
    
    Args:
        input_dir (str): Direktori yang berisi gambar-gambar mentah.
        output_dir (str): Direktori untuk menyimpan gambar-gambar yang sudah dinormalisasi.
        feature_range (tuple): Rentang output normalisasi (default: (0, 255)).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Direktori output '{output_dir}' berhasil dibuat.")

    print(f"[INFO] Memproses dataset dari '{input_dir}'...")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            normalized_image_array = min_max_normalize(input_path, feature_range)

            if normalized_image_array is not None:
                # Simpan dalam format gambar (uint8 jika range 0-255)
                if feature_range == (0, 255):
                    normalized_img_uint8 = normalized_image_array.astype(np.uint8)
                    Image.fromarray(normalized_img_uint8, 'L').save(os.path.join(output_dir, filename))
                else:
                    # Kalau range bukan 0-255, tetap simpan ke 0-255 agar bisa dilihat sebagai gambar
                    normalized_img_uint8 = ((normalized_image_array - np.min(normalized_image_array)) / 
                                            (np.max(normalized_image_array) - np.min(normalized_image_array)) * 255).astype(np.uint8)
                    Image.fromarray(normalized_img_uint8, 'L').save(os.path.join(output_dir, filename))

                print(f"[INFO] Gambar disimpan: '{os.path.join(output_dir, filename)}'")


# --- Penggunaan Program ---
if __name__ == "__main__":
    # Ganti dengan jalur direktori dataset Anda
    input_directory = '../datasets/base_dataset/001_cropped/abnormal'
    output_directory = '../datasets/base_dataset/004_intensity_normalized_minmax/abnormal'

    # Proses seluruh dataset
    process_dataset(input_directory, output_directory, feature_range=(0, 255))

    # Contoh penggunaan untuk satu gambar
    example_image_path = 'path/to/your/single/image.jpg'
    # normalized_single_image = min_max_normalize(example_image_path, feature_range=(0, 1))
    # if normalized_single_image is not None:
    #     print("\nContoh hasil array normalisasi Min-Max:")
    #     print(normalized_single_image)
