import sys
import os
import glob
import pandas as pd
from tqdm import tqdm
from functools import reduce

# Add local feature extraction modules to sys.path
sys.path.append(os.path.abspath('./feature_extraction/'))

# Import feature extraction functions
from rgb_color_moment import get_rgb_color_moment_features, get_rgb_color_moment_feature_names
from yuv_color_moment import get_yuv_color_moment_features, get_yuv_color_moment_feature_names
from lab_color_moment import get_lab_color_moment_features, get_lab_color_moment_feature_names
from glrlm_feature_extraction import get_glrlm_features, get_glrlm_feature_names
from tamura_feature_extraction import get_tamura_features, get_tamura_feature_names
from lbp_feature_extraction import get_lbp_features, get_lbp_feature_names
from glcm_feature_extraction import get_glcm_features, get_glcm_feature_names


class ExtractFeatures:
    """
    Batch feature extraction for image datasets.
    
    Supports multiple root directories, with a single metadata CSV for labels.
    Outputs CSVs separately for each root directory.
    """

    def __init__(self, root_dirs, metadata_csv):
        if isinstance(root_dirs, str):
            self.root_dirs = [root_dirs]
        else:
            self.root_dirs = root_dirs

        self.metadata_csv = metadata_csv

        # Mapping for color moment features
        self.color_moment_features = {
            'RGB': (get_rgb_color_moment_features, get_rgb_color_moment_feature_names),
            'YUV': (get_yuv_color_moment_features, get_yuv_color_moment_feature_names),
            'LAB': (get_lab_color_moment_features, get_lab_color_moment_feature_names),
        }

        # Mapping for texture features
        self.texture_features = {
            'GLRLM': (get_glrlm_features, get_glrlm_feature_names),
            'TAMURA': (get_tamura_features, get_tamura_feature_names),
            'LBP': (get_lbp_features, get_lbp_feature_names),
            'GLCM': (get_glcm_features, get_glcm_feature_names),
        }

        # Load metadata CSV and map labels
        print(f"📥 Loading metadata from {metadata_csv} ...")
        metadata = pd.read_csv(metadata_csv)
        metadata['label'] = metadata['label'].map({'Negative': 'normal', 'Positive': 'abnormal'})
        self.metadata_dict = dict(zip(metadata['image'], metadata['label']))
        print(f"✅ Loaded {len(self.metadata_dict)} entries from metadata.")


    def extract_features(self, color_spaces=None, texture_features=None):
        if color_spaces is None:
            color_spaces = ['RGB', 'YUV', 'LAB']
        if texture_features is None:
            texture_features = ['LBP', 'GLRLM', 'TAMURA']

        # Process each root directory
        for root_dir in self.root_dirs:
            folder_name = os.path.basename(root_dir.rstrip('/\\'))
            output_dir = os.path.join(os.path.dirname(root_dir), f"{folder_name}_features")
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n📁 Processing directory: {root_dir}")
            print(f"🗂 Output folder: {output_dir}")

            # Initialize DataFrames for color moments
            cm_features_df = {}
            for key in color_spaces:
                if key in self.color_moment_features:
                    columns = ['image_id'] + self.color_moment_features[key][1]() + ['label']
                    cm_features_df[key] = pd.DataFrame(columns=columns)

            # Initialize DataFrames for texture features
            texture_features_df = {}
            for key in texture_features:
                if key in self.texture_features:
                    columns = ['image_id'] + self.texture_features[key][1]() + ['label']
                    texture_features_df[key] = pd.DataFrame(columns=columns)

            # Get all images in this directory
            image_paths = glob.glob(os.path.join(root_dir, '*'))
            total_images = len(image_paths)
            print(f"🖼 Total images found: {total_images}")

            # Iterate over images
            with tqdm(total=total_images, desc=f"🔍 Extracting features ({folder_name})", unit="img") as pbar:
                for image_path in image_paths:
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]

                    # Match image ID with metadata
                    matched = next((meta for meta in self.metadata_dict if image_filename.startswith(meta)), None)
                    if matched:
                        label = self.metadata_dict[matched]
                    else:
                        label = 'unknown'
                        print(f"⚠️  Warning: {image_filename} not found in metadata. Using 'unknown' label.")

                    # Extract color moment features
                    for key in color_spaces:
                        if key in self.color_moment_features:
                            get_features = self.color_moment_features[key][0]
                            row = [image_filename] + get_features(image_path) + [label]
                            cm_features_df[key].loc[len(cm_features_df[key])] = row

                    # Extract texture features
                    for key in texture_features:
                        if key in self.texture_features:
                            get_features = self.texture_features[key][0]
                            row = [image_filename] + get_features(image_path) + [label]
                            texture_features_df[key].loc[len(texture_features_df[key])] = row

                    pbar.update(1)

            # Merge texture features if multiple
            df_texture_merged = None
            if texture_features:
                dfs = [texture_features_df[feature] for feature in texture_features]
                df_texture_merged = reduce(lambda left, right: pd.merge(left, right, on=['image_id', 'label']), dfs)

            # Save CSVs per root directory
            for color_space in color_spaces:
                df_combined = cm_features_df[color_space]
                if df_texture_merged is not None:
                    df_combined = pd.merge(df_combined, df_texture_merged, on=['image_id', 'label'])

                filename = f"{color_space}_{'_'.join(texture_features)}.csv" if texture_features else f"{color_space}.csv"
                output_path = os.path.join(output_dir, filename)
                df_combined.to_csv(output_path, index=False)
                print(f"💾 Saved CSV: {output_path}")


if __name__ == "__main__":
    # Example usage with single metadata CSV
    root_dirs = [
        "../datasets/filtered_dataset/multiotsu_bilateral/output_bilateral_d15_sc75_ss75",
        "../datasets/filtered_dataset/multiotsu_bilateral/output_bilateral_d25_sc75_ss75",
        "../datasets/filtered_dataset/multiotsu_white_balanced",
    ]
    metadata_csv = '../datasets/meta_data_IARC.csv'  # single CSV for all directories

    extractor = ExtractFeatures(root_dirs, metadata_csv)
    extractor.extract_features()
