"""
Pipeline A - Classical + Chroma-aware (Batch)
------------------------------------------------
Saves each intermediate result into its own folder and separates final
segmentation masks into Normal / Abnormal according to a CSV file.

Requirements:
    - Python 3.8+
    - OpenCV (cv2)
    - numpy
    - pandas
    - matplotlib (optional, for debug)
    - scikit-image (for remove_small_objects / holes)
    - tqdm

Notes about CSV matching:
    The CSV must contain columns: "image" and "label".
    - label values expected: "Negative" or "Positive" (case-insensitive)
    Matching rule (applied in order):
    1) exact match of filename without extension
    2) filename startswith(csv_name)  (useful for AAE and AAE1)
    3) csv_name substring of filename
    The script selects the longest matching csv_name if multiple matches exist.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import argparse
from skimage.morphology import remove_small_objects, remove_small_holes

# -----------------------
# Utility / Processing Functions
# -----------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_image_files(folder, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]

def copy_originals(src_folder, dst_folder, files):
    ensure_dir(dst_folder)
    for fname in tqdm(files, desc="Copying originals"):
        shutil.copy2(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

# -------- White balance: Shades-of-Gray --------
def shades_of_gray_white_balance_rgb(img_rgb, p=6):
    """
    img_rgb: uint8 RGB image (H,W,3)
    p: Minkowski norm (p=1 -> Gray-World)
    Returns: uint8 RGB image
    """
    img = img_rgb.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    # Minkowski p-norm average per channel
    R_avg = np.power(np.mean(np.power(R, p)), 1.0/p)
    G_avg = np.power(np.mean(np.power(G, p)), 1.0/p)
    B_avg = np.power(np.mean(np.power(B, p)), 1.0/p)
    mean_gray = (R_avg + G_avg + B_avg) / 3.0
    out = np.empty_like(img)
    out[:,:,0] = img[:,:,0] * (mean_gray / (R_avg + 1e-12))
    out[:,:,1] = img[:,:,1] * (mean_gray / (G_avg + 1e-12))
    out[:,:,2] = img[:,:,2] * (mean_gray / (B_avg + 1e-12))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# -------- Glare detection & inpaint --------
def detect_and_inpaint_glare(img_rgb, v_th=220, s_th=30, morph_k=5):
    """
    img_rgb: uint8 RGB image
    returns: (inpainted_rgb_uint8, glare_mask_uint8)
    glare_mask: 0 or 255 single-channel
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:,:,1]
    V = hsv[:,:,2]

    glare_mask = np.zeros_like(S, dtype=np.uint8)
    glare_mask[(V > v_th) & (S < s_th)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)

    # inpaint expects 8-bit single channel mask and BGR image
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(img_bgr, glare_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return inpainted_rgb, glare_mask

# -------- Guided filter fallback --------
def guided_filter_gray(src_float01, radius=5, eps=1e-3):
    """
    Guided smoothing for a single-channel float image in [0,1].
    Tries cv2.ximgproc.guidedFilter if present, otherwise falls back to bilateral filter.
    Returns float in [0,1], same shape.
    """
    try:
        # cv2.ximgproc.guidedFilter(guide, src, radius, eps)
        # convert to correct type
        src_32f = (src_float01).astype(np.float32)
        # Some OpenCV builds require guide to be same number of channels as src.
        gf = cv2.ximgproc.guidedFilter
        res = gf((src_32f*255).astype(np.uint8), (src_32f*255).astype(np.uint8), radius, eps)
        res = (res.astype(np.float32) / 255.0)
        return res
    except Exception:
        # fallback: bilateral filter on 8-bit
        res = cv2.bilateralFilter((src_float01*255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
        return (res.astype(np.float32) / 255.0)

# -------- LAB preprocess (L guided + CLAHE) --------
def lab_preprocess_and_save(img_rgb, radius=5, eps=1e-3, clahe_clip=2.5, clahe_tile=(8,8)):
    """
    Input: RGB uint8 image
    Output:
      - L_clahe_uint8 (single channel)
      - a_uint8, b_uint8 (as returned by OpenCV LAB split)
      - C_ab_norm_uint8  (chroma normalized to 0-255)
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    # convert L to float in [0,1] for guided filter
    Lf = L.astype(np.float32) / 255.0
    Lgf = guided_filter_gray(Lf, radius=radius, eps=eps)
    Lgf_u8 = (np.clip(Lgf*255.0, 0, 255)).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    L_clahe = clahe.apply(Lgf_u8)

    # compute chroma: center a and b at 0 (OpenCV uses 128 offset)
    a_f = a.astype(np.float32) - 128.0
    b_f = b.astype(np.float32) - 128.0
    C_ab = np.sqrt(a_f**2 + b_f**2)
    # normalize chroma to 0-255 to save and threshold easily
    C_ab_norm = cv2.normalize(C_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return L_clahe, a, b, C_ab_norm

# -------- Segmentation: L high & C_ab low intersection --------
def segment_from_L_and_Cab(L_clahe_u8, C_ab_norm_u8, min_size=500):
    """
    Perform Otsu on L_clahe (bright) and Otsu on C_ab_norm (low chroma -> invert)
    Clean with morphological small object removal.
    Returns lesion_mask_uint8 (0/255).
    """
    # Otsu on L
    _, L_mask = cv2.threshold(L_clahe_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu on chroma (invert to get low-chroma)
    _, Cab_inv = cv2.threshold(C_ab_norm_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lesion = cv2.bitwise_and(L_mask, Cab_inv)

    # Clean using skimage (expects boolean)
    lesion_bool = (lesion > 0)
    # remove very small objects and fill small holes
    lesion_bool = remove_small_objects(lesion_bool, min_size=min_size)
    lesion_bool = remove_small_holes(lesion_bool, area_threshold=min_size)
    lesion_mask = (lesion_bool.astype(np.uint8) * 255)
    return lesion_mask, L_mask, Cab_inv

# -----------------------
# CSV label matching
# -----------------------
def build_csv_map(csv_path):
    """
    Reads CSV and returns dict mapping csv_name_lower -> label_lower
    """
    df = pd.read_csv(csv_path, dtype=str)
    if not {'image', 'label'}.issubset(df.columns.str.lower()):
        # try case-insensitive column detection
        cols = {c.lower(): c for c in df.columns}
        if 'image' in cols and 'label' in cols:
            df = df.rename(columns={cols['image']: 'image', cols['label']: 'label'})
        else:
            raise ValueError("CSV must contain columns named 'image' and 'label' (case-insensitive).")
    # normalize
    df['image'] = df['image'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    # accept "negative"/"positive" (map negative->normal, positive->abnormal)
    mapping = dict()
    for _, row in df.iterrows():
        key = row['image'].lower()
        val = row['label']
        mapping[key] = val
    return mapping

def find_label_for_filename(fname_no_ext, csv_map):
    """
    Match filename (no ext) to csv_map keys using:
    1) exact match
    2) csv_key is prefix of fname
    3) csv_key is substring of fname
    Returns value or None
    """
    base = fname_no_ext.lower()
    if base in csv_map:
        return csv_map[base]
    # prefix matches
    prefix_matches = [k for k in csv_map.keys() if base.startswith(k)]
    if prefix_matches:
        # choose longest key (most specific)
        chosen = max(prefix_matches, key=len)
        return csv_map[chosen]
    substr_matches = [k for k in csv_map.keys() if k in base]
    if substr_matches:
        chosen = max(substr_matches, key=len)
        return csv_map[chosen]
    return None

# -----------------------
# Main batch pipeline orchestration
# -----------------------
def run_pipeline_batch(input_dir, output_dir, csv_path=None, min_size=500,
                       v_th=220, s_th=40, wb_p=6, debug=False):
    """
    Orchestrates stages:
      01_original -> 02_wb -> 03_glare_mask (masks saved) -> 04_no_glare ->
      05_L_clahe -> 06_a -> 07_b -> 08_Cab -> 09_segmentation/{Normal,Abnormal,Unknown}
    """
    ensure_dir(output_dir)
    stage_folders = {
        "original": os.path.join(output_dir, "01_original"),
        "wb": os.path.join(output_dir, "02_wb"),
        "glare_mask": os.path.join(output_dir, "03_glare_mask"),
        "no_glare": os.path.join(output_dir, "04_no_glare"),
        "L_clahe": os.path.join(output_dir, "05_L_clahe"),
        "a": os.path.join(output_dir, "06_a"),
        "b": os.path.join(output_dir, "07_b"),
        "C_ab": os.path.join(output_dir, "08_Cab"),
        "segmentation": os.path.join(output_dir, "09_segmentation"),
    }
    for fd in stage_folders.values():
        ensure_dir(fd)
    ensure_dir(os.path.join(stage_folders["segmentation"], "Normal"))
    ensure_dir(os.path.join(stage_folders["segmentation"], "Abnormal"))
    ensure_dir(os.path.join(stage_folders["segmentation"], "Unknown"))

    # list input files
    image_files = list_image_files(input_dir)
    if len(image_files) == 0:
        print("No images found in input folder:", input_dir)
        return

    # Step 0: copy originals
    print("\n[STEP 0] Copy originals to stage folder")
    copy_originals(input_dir, stage_folders["original"], image_files)

    # Build CSV map if provided
    csv_map = {}
    if csv_path:
        print("\nLoading CSV mapping from:", csv_path)
        csv_map = build_csv_map(csv_path)
        # quick stats
        counts = {}
        for v in csv_map.values():
            counts[v] = counts.get(v, 0) + 1
        print("CSV label counts (by csv rows):", counts)

    # Step 1: White balance
    print("\n[STEP 1] White balance (Shades-of-Gray) ->", stage_folders["wb"])
    files = list_image_files(stage_folders["original"])
    for fname in tqdm(files, desc="WB"):
        in_path = os.path.join(stage_folders["original"], fname)
        out_path = os.path.join(stage_folders["wb"], fname)
        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print("Warning: cannot read", in_path); continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_wb_rgb = shades_of_gray_white_balance_rgb(img_rgb, p=wb_p)
        # save using original extension; convert to BGR
        cv2.imwrite(out_path, cv2.cvtColor(img_wb_rgb, cv2.COLOR_RGB2BGR))

    # Step 2: Glare detection & inpaint
    print("\n[STEP 2] Glare detection (HSV) & inpaint -> masks:", stage_folders["glare_mask"],
          "  no-glare:", stage_folders["no_glare"])
    files = list_image_files(stage_folders["wb"])
    for fname in tqdm(files, desc="Glare"):
        in_path = os.path.join(stage_folders["wb"], fname)
        out_noglare = os.path.join(stage_folders["no_glare"], fname)
        out_mask = os.path.join(stage_folders["glare_mask"], fname)
        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print("Warning: cannot read", in_path); continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_inpaint_rgb, glare_mask = detect_and_inpaint_glare(img_rgb, v_th=v_th, s_th=s_th)
        # save results
        # save inpainted
        cv2.imwrite(out_noglare, cv2.cvtColor(img_inpaint_rgb, cv2.COLOR_RGB2BGR))
        # save glare mask (single channel) with same filename/extension
        cv2.imwrite(out_mask, glare_mask)

    # Step 3: LAB preprocess (L_clahe, a, b, C_ab)
    print("\n[STEP 3] LAB preprocessing -> L_clahe, a, b, C_ab")
    files = list_image_files(stage_folders["no_glare"])
    for fname in tqdm(files, desc="LAB"):
        in_path = os.path.join(stage_folders["no_glare"], fname)
        out_L = os.path.join(stage_folders["L_clahe"], fname)
        out_a = os.path.join(stage_folders["a"], fname)
        out_b = os.path.join(stage_folders["b"], fname)
        out_Cab = os.path.join(stage_folders["C_ab"], fname)

        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print("Warning: cannot read", in_path); continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        L_clahe, a_chan, b_chan, C_ab_norm = lab_preprocess_and_save(img_rgb)
        # save L, a, b, Cab as single-channel uint8 using the original filename
        cv2.imwrite(out_L, L_clahe)
        cv2.imwrite(out_a, a_chan)
        cv2.imwrite(out_b, b_chan)
        cv2.imwrite(out_Cab, C_ab_norm)

    # Step 4: Segmentation & separation by CSV label
    print("\n[STEP 4] Segmentation (L high & C_ab low), cleaning, and saving into Normal/Abnormal")
    files = list_image_files(stage_folders["L_clahe"])
    counts_saved = {"Normal":0, "Abnormal":0, "Unknown":0}
    for fname in tqdm(files, desc="Segmentation"):
        L_path = os.path.join(stage_folders["L_clahe"], fname)
        Cab_path = os.path.join(stage_folders["C_ab"], fname)
        # read L and C_ab (they were saved as uint8 single-channel)
        L = cv2.imread(L_path, cv2.IMREAD_GRAYSCALE)
        Cab = cv2.imread(Cab_path, cv2.IMREAD_GRAYSCALE)
        if L is None or Cab is None:
            print("Warning: missing L or C_ab for", fname); continue

        lesion_mask, L_mask, Cab_inv = segment_from_L_and_Cab(L, Cab, min_size=min_size)

        # Determine CSV label
        base_no_ext = os.path.splitext(fname)[0]
        label = find_label_for_filename(base_no_ext, csv_map) if csv_map else None
        if label is None:
            target_folder = os.path.join(stage_folders["segmentation"], "Unknown")
            counts_saved["Unknown"] += 1
        else:
            # mapping label values: negative -> Normal, positive -> Abnormal
            if label.lower().startswith("neg"):
                target_folder = os.path.join(stage_folders["segmentation"], "Normal")
                counts_saved["Normal"] += 1
            elif label.lower().startswith("pos"):
                target_folder = os.path.join(stage_folders["segmentation"], "Abnormal")
                counts_saved["Abnormal"] += 1
            else:
                target_folder = os.path.join(stage_folders["segmentation"], "Unknown")
                counts_saved["Unknown"] += 1

        out_mask_path = os.path.join(target_folder, fname)
        cv2.imwrite(out_mask_path, lesion_mask)

    print("\nSegmentation saved counts:", counts_saved)
    print("\nPipeline completed. Stages saved under:", output_dir)
    if debug:
        print("Debug mode: to visualize specific files open images from stage folders.")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline A - Classical + Chroma-aware (batch)")
    parser.add_argument("--input", "-i", required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", required=True, help="Output folder to save stages")
    parser.add_argument("--csv", "-c", default=None, help="CSV file with columns 'image' and 'label'")
    parser.add_argument("--min_size", type=int, default=500, help="Minimum object size (px) to keep in segmentation")
    parser.add_argument("--v_th", type=int, default=220, help="V threshold for glare detection")
    parser.add_argument("--s_th", type=int, default=40, help="S threshold for glare detection")
    parser.add_argument("--wb_p", type=int, default=6, help="p parameter for Shades-of-Gray WB")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()

    run_pipeline_batch(
        args.input, args.output, csv_path=args.csv,
        min_size=args.min_size, v_th=args.v_th, s_th=args.s_th,
        wb_p=args.wb_p, debug=args.debug)
