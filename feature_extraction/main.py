import sys
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce

# Add local feature extraction modules to sys.path
sys.path.append(os.path.abspath('./feature_extraction/'))

# Import all feature extraction functions and their name getters
from rgb_color_moment import get_rgb_color_moment_features, get_rgb_color_moment_feature_names
from yuv_color_moment import get_yuv_color_moment_features, get_yuv_color_moment_feature_names
from lab_color_moment import get_lab_color_moment_features, get_lab_color_moment_feature_names
from glrlm_feature_extraction import get_glrlm_features, get_glrlm_feature_names
from tamura_feature_extraction import get_tamura_features, get_tamura_feature_names
from lbp_feature_extraction import get_lbp_features, get_lbp_feature_names
from glcm_feature_extraction import get_glcm_features, get_glcm_feature_names


class ExtractFeatures:
    """
    A class to batch extract multiple color moment and texture features
    from a structured image dataset.

    Attributes:
        root_dir (str): Root directory containing subfolders for each label.
        output_dir (str): Directory to save the extracted feature CSV files.
        color_moment_features (dict): Mapping of color space names to extractor functions.
        texture_features (dict): Mapping of texture feature names to extractor functions.
        cm_features_df (dict): Stores DataFrames for each color moment type.
        texture_features_df (dict): Stores DataFrames for each texture type.
    """

    def __init__(self, root_dir, output_dir):
        """
        Initialize ExtractFeatures with input and output directories.
        Prepares the feature mapping for color moments and texture features.
        """
        self.root_dir = root_dir
        self.output_dir = output_dir

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

        # Containers for DataFrames
        self.cm_features_df = {}
        self.texture_features_df = {}

    def extract_features(self, color_spaces=None, texture_features=None):
        """
        Extract specified color moment and texture features for all images.

        Parameters:
            color_spaces (list or None): Color spaces to extract ['RGB', 'YUV', 'LAB'].
                                         If None, all are used.
            texture_features (list or None): Texture features to extract ['LBP', 'GLRLM', 'TAMURA', 'GLCM'].
                                             If None, defaults to ['LBP', 'GLRLM', 'TAMURA'].
        """
        # Use defaults if not specified
        if color_spaces is None:
            color_spaces = ['RGB', 'YUV', 'LAB']
        if texture_features is None:
            texture_features = ['LBP', 'GLRLM', 'TAMURA']

        # Initialize DataFrames for selected color moments
        for key in color_spaces:
            if key in self.color_moment_features:
                columns = ['image_id'] + self.color_moment_features[key][1]() + ['label']
                self.cm_features_df[key] = pd.DataFrame(columns=columns)

        # Initialize DataFrames for selected texture features
        for key in texture_features:
            if key in self.texture_features:
                columns = ['image_id'] + self.texture_features[key][1]() + ['label']
                self.texture_features_df[key] = pd.DataFrame(columns=columns)

        # Get all subfolders representing class labels
        sub_image_folders = glob.glob(os.path.join(self.root_dir, '*'))
        total_images = sum(len(glob.glob(os.path.join(sub, '*'))) for sub in sub_image_folders)

        print(f"Total images to process: {total_images}")

        # Iterate with progress bar
        with tqdm(total=total_images, desc="Extracting features", unit="img") as pbar:
            for sub_image_folder in sub_image_folders:
                label = os.path.basename(sub_image_folder)
                image_paths = glob.glob(os.path.join(sub_image_folder, '*'))

                for image_path in image_paths:
                    image_id = os.path.splitext(os.path.basename(image_path))[0]

                    # Extract color moment features for each selected color space
                    for key in color_spaces:
                        if key in self.color_moment_features:
                            get_features = self.color_moment_features[key][0]
                            row = [image_id] + get_features(image_path) + [label]
                            self.cm_features_df[key].loc[len(self.cm_features_df[key])] = row

                    # Extract texture features for each selected texture type
                    for key in texture_features:
                        if key in self.texture_features:
                            get_features = self.texture_features[key][0]
                            row = [image_id] + get_features(image_path) + [label]
                            self.texture_features_df[key].loc[len(self.texture_features_df[key])] = row

                    pbar.update(1)

        # If multiple texture features, merge them by image_id and label
        df_texture_merged = None
        if texture_features:
            dfs = [self.texture_features_df[feature] for feature in texture_features]
            df_texture_merged = reduce(
                lambda left, right: pd.merge(left, right, on=['image_id', 'label']), dfs)

        # Save output files for each color moment combined with merged texture features
        os.makedirs(self.output_dir, exist_ok=True)
        for color_space in color_spaces:
            df_combined = self.cm_features_df[color_space]
            if df_texture_merged is not None:
                df_combined = pd.merge(df_combined, df_texture_merged, on=['image_id', 'label'])

            filename = f"{color_space}_{'_'.join(texture_features)}.csv" if texture_features else f"{color_space}.csv"
            output_path = os.path.join(self.output_dir, filename)
            df_combined.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Example usage for batch feature extraction
    root_dir = '../datasets/dataset_k/001_gmm_segmented_images'
    output_dir = '../datasets/dataset_k/features'

    extractor = ExtractFeatures(root_dir, output_dir)
    extractor.extract_features()
