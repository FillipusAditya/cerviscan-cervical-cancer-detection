import cv2
import os
import glob

def crop_images(folder_path, output_folder_path, max_display_ratio=0.9, extensions=("*.jpg", "*.png", "*.jpeg")):
    """
    Crop multiple images from the given folder using manual ROI selection on resized images.

    Args:
        folder_path (str): Path to the input images.
        output_folder_path (str): Path to save cropped images.
        max_display_ratio (float): Persentase maksimal ukuran layar untuk display (default: 0.9 -> 90%).
        extensions (tuple): Tuple of file extensions to include (default: jpg, png, jpeg).
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Ambil semua file gambar dengan ekstensi tertentu
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        print("⚠️ Tidak ada file gambar ditemukan di folder input!")
        return

    # Ambil resolusi layar (pakai tkinter biar cross platform)
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        screen_width, screen_height = 1280, 720  # fallback default

    success_count = 0

    for idx, image_file in enumerate(image_files, 1):
        image = cv2.imread(image_file)
        if image is None:
            print(f"❌ Gagal membaca gambar: {image_file}")
            continue

        original_height, original_width = image.shape[:2]

        # Tentukan skala supaya gambar muat di layar (90% dari layar)
        scale_w = (screen_width * max_display_ratio) / original_width
        scale_h = (screen_height * max_display_ratio) / original_height
        scale = min(scale_w, scale_h)

        display_width = int(original_width * scale)
        display_height = int(original_height * scale)
        resized_image = cv2.resize(image, (display_width, display_height))

        # Pilih ROI
        print(f"[{idx}/{len(image_files)}] Pilih ROI untuk gambar: {os.path.basename(image_file)}")
        r = cv2.selectROI("Select ROI and press ENTER (or C to cancel)", resized_image, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        # Jika user tidak memilih ROI
        if r[2] == 0 or r[3] == 0:
            print(f"⚠️ ROI dibatalkan untuk gambar: {os.path.basename(image_file)}")
            continue

        # Scale ROI ke ukuran original
        x1 = int(r[0] / scale)
        y1 = int(r[1] / scale)
        x2 = int((r[0] + r[2]) / scale)
        y2 = int((r[1] + r[3]) / scale)

        # Crop gambar original
        cropped_image = image[y1:y2, x1:x2]

        # Simpan hasil crop
        filename = os.path.basename(image_file)
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, cropped_image)
        print(f"✅ Cropped image saved to: {output_path}")
        success_count += 1

    print(f"\n✨ Selesai! {success_count}/{len(image_files)} gambar berhasil dicrop.")

# ==============================
# Contoh penggunaan:
# ==============================
if __name__ == "__main__":
    input_folder = "../datasets/dataset_puskesmas/og/normal"       
    output_folder = "../datasets/dataset_puskesmas/cropped/normal"    
    crop_images(input_folder, output_folder, max_display_ratio=0.8)  # 80% dari layar
