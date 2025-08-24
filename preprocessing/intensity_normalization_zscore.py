import os
from PIL import Image
import numpy as np

def z_score_normalize(image_path):
    """
    Melakukan Z-score normalization pada sebuah gambar.
    
    Args:
        image_path (str): Jalur lengkap ke file gambar.

    Returns:
        numpy.ndarray: Gambar yang sudah dinormalisasi dalam bentuk array NumPy, atau None jika terjadi kesalahan.
    """
    try:
        # Buka gambar dan konversi ke grayscale
        img = Image.open(image_path).convert('L')
        # Konversi gambar ke array NumPy
        img_array = np.array(img, dtype=np.float32)
        
        # Hitung rata-rata dan deviasi standar
        mean = np.mean(img_array)
        std = np.std(img_array)
        
        # Hindari pembagian dengan nol
        if std == 0:
            print(f"[PERINGATAN] Deviasi standar nol pada gambar '{os.path.basename(image_path)}'. Normalisasi tidak dilakukan.")
            return img_array
            
        # Terapkan formula Z-score
        normalized_img_array = (img_array - mean) / std
        
        print(f"[INFO] Gambar '{os.path.basename(image_path)}' berhasil dinormalisasi (Z-score).")
        return normalized_img_array
        
    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan: {image_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat memproses gambar '{str(image_path)}': {e}")
        return None


def process_dataset(input_dir, output_dir):
    """
    Memproses seluruh dataset gambar dalam sebuah direktori.
    
    Args:
        input_dir (str): Direktori yang berisi gambar-gambar mentah.
        output_dir (str): Direktori untuk menyimpan gambar-gambar yang sudah dinormalisasi.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Direktori output '{output_dir}' berhasil dibuat.")
    
    print(f"[INFO] Memproses dataset dari '{input_dir}'...")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            normalized_image_array = z_score_normalize(input_path)
            
            if normalized_image_array is not None:
                # Normalisasi Min-Max agar hasil bisa disimpan sebagai gambar
                min_val, max_val = np.min(normalized_image_array), np.max(normalized_image_array)
                if max_val - min_val == 0:
                    print(f"[PERINGATAN] Gambar '{filename}' memiliki rentang nol setelah normalisasi. Disimpan tanpa scaling.")
                    normalized_img_uint8 = np.zeros_like(normalized_image_array, dtype=np.uint8)
                else:
                    normalized_minmax = (normalized_image_array - min_val) / (max_val - min_val)
                    normalized_img_uint8 = (normalized_minmax * 255).astype(np.uint8)
                
                # Simpan dengan nama asli
                output_path = os.path.join(output_dir, filename)
                Image.fromarray(normalized_img_uint8, 'L').save(output_path)
                print(f"[INFO] Gambar disimpan: '{output_path}'")


# --- Penggunaan Program ---
if __name__ == "__main__":
    # Ganti dengan jalur direktori dataset Anda
    input_directory = '../datasets/base_dataset/001_cropped/normal'
    output_directory = '../datasets/base_dataset/002_intensity_normalized_zscore/normal'
    
    # Proses seluruh dataset
    process_dataset(input_directory, output_directory)
    
    # Contoh penggunaan untuk satu gambar
    example_image_path = 'path/to/your/single/image.jpg'
    # normalized_single_image = z_score_normalize(example_image_path)
    # if normalized_single_image is not None:
    #     print("\nContoh hasil array normalisasi Z-score:")
    #     print(normalized_single_image)
