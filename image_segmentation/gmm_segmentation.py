import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

class AcetowhiteSegmenter:
    """
    Class for segmenting acetowhite lesions in cervical images using GMM clustering
    in the L channel of the LAB color space.

    Steps:
        - Converts input RGB image to LAB color space.
        - Enhances L channel with CLAHE and bilateral filtering.
        - Uses Gaussian Mixture Model (GMM) with automatic K selection using BIC and KneeLocator.
        - Extracts the component with the highest mean L value as acetowhite region.
        - Applies thresholding and morphological operations to clean the mask.
        - Crops the region and saves both the cropped lesion and diagnostic plots.
    """

    def __init__(self, input_dir, output_dir, L_thresh=150):
        """
        Initialize the segmenter with input/output directories and L channel threshold.

        Args:
            input_dir (str): Path to input images.
            output_dir (str): Path to save outputs.
            L_thresh (int): L channel threshold to refine acetowhite region.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.L_thresh = L_thresh

    def ensure_dir(self, path):
        """Create directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    def process_image(self, img_path, save_dir_crop, save_dir_summary, filename):
        """
        Process a single image: segment acetowhite region, crop it, save result and plots.

        Args:
            img_path (str): Path to the input image.
            save_dir_crop (str): Directory to save cropped output.
            save_dir_summary (str): Directory to save summary plots.
            filename (str): Image filename for saving outputs.
        """
        # Load image and convert to RGB 
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to LAB and split channels
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # Enhance L channel with CLAHE and bilateral filter
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_clahe = clahe.apply(L)
        L_blur = cv2.bilateralFilter(L_clahe, d=9, sigmaColor=75, sigmaSpace=75)
        X = L_blur.reshape(-1, 1)

        # Select optimal K for GMM using BIC and KneeLocator
        ks = range(1, 10)
        bics, aics = [], []
        for k in ks:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(X)
            bics.append(gmm.bic(X))
            aics.append(gmm.aic(X))

        # Find the knee point for optimal K, fallback to 3 if not found
        knee_bic = KneeLocator(ks, bics, curve='convex', direction='decreasing')
        best_k = knee_bic.knee or 3

        # Fit GMM with best K
        gmm_best = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
        gmm_best.fit(X)
        labels_best = gmm_best.predict(X)
        segmented_best = labels_best.reshape(L_blur.shape)

        # Extract acetowhite region: component with highest mean
        means = gmm_best.means_.flatten()
        acetowhite_label = np.argmax(means)
        acetowhite_mask = (segmented_best == acetowhite_label).astype(np.uint8)
        acetowhite_mask = acetowhite_mask & (L_blur > self.L_thresh)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(acetowhite_mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Crop the region using the clean mask
        acetowhite_mask_uint8 = (mask_clean * 255).astype(np.uint8)
        acetowhite_mask_3ch = cv2.merge([acetowhite_mask_uint8] * 3)
        cropped_lesion = cv2.bitwise_and(img_rgb, acetowhite_mask_3ch)

        # Save the cropped image
        cropped_path = os.path.join(save_dir_crop, filename)
        cv2.imwrite(cropped_path, cv2.cvtColor(cropped_lesion, cv2.COLOR_RGB2BGR))

        # Plot GMM PDF + histogram
        x = np.linspace(0, 255, 256).reshape(-1, 1)
        logprob = gmm_best.score_samples(x)
        pdf = np.exp(logprob)
        responsibilities = gmm_best.predict_proba(x)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        colors = sns.color_palette('tab20', int(best_k)).as_hex()
        cmap = ListedColormap(colors)
        bounds = np.arange(-0.5, int(best_k) + 0.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)

        # Full summary figure
        fig = plt.figure(figsize=(20, 12))

        # BIC + AIC vs K plot
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
        ax1.plot(ks, bics, marker='o', label='BIC')
        ax1.plot(ks, aics, marker='s', label='AIC')
        ax1.vlines(best_k, min(bics + aics), max(bics + aics), colors='blue',
                   linestyles='dashed', label=f'BIC Knee: {best_k}')
        ax1.set_title('BIC & AIC vs Number of Components (K)')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True)

        # Original image panel
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax2.imshow(img_rgb)
        ax2.set_title('Original RGB')
        ax2.axis('off')

        # Processed L channel panel
        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax3.imshow(L_blur, cmap='gray')
        ax3.set_title('L Channel (CLAHE+Bilateral)')
        ax3.axis('off')

        # Segmented mask panel
        ax4 = plt.subplot2grid((2, 3), (1, 2))
        im = ax4.imshow(segmented_best, cmap=cmap, norm=norm)
        ax4.set_title(f'GMM Segmentation (K={best_k})')
        ax4.axis('off')
        cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, ticks=range(int(best_k)))
        cbar.ax.set_yticklabels([f'Component {i}' for i in range(int(best_k))])

        # Save PDF as a separate figure
        fig2, ax5 = plt.subplots(figsize=(8, 4))
        ax5.hist(X.flatten(), bins=50, density=True, alpha=0.5, color='gray',
                 label='Pixel Intensity Histogram')
        ax5.plot(x, pdf, '-k', label='Total GMM PDF', linewidth=2)
        for i in range(best_k):
            ax5.plot(x, pdf_individual[:, i], '--', color=colors[i], linewidth=2,
                     label=f'Component {i + 1}')
        ax5.set_title(f'GMM PDF Fit (K={best_k})')
        ax5.set_xlabel('L Channel Intensity')
        ax5.set_ylabel('Density')
        ax5.legend()

        # Save PDF plot
        pdf_img_path = os.path.join(save_dir_summary, f"{filename}_pdf.png")
        fig2.savefig(pdf_img_path)
        plt.close(fig2)

        # Save main summary figure
        summary_path = os.path.join(save_dir_summary, f"{filename}_summary.png")
        fig.savefig(summary_path)
        plt.close(fig)

    def run(self):
        """
        Run the segmentation for all images in the input directory,
        separated by label ('abnormal', 'normal').

        Creates necessary directories and calls process_image for each file.
        """
        for label in ['abnormal', 'normal']:
            input_subdir = os.path.join(self.input_dir, label)
            output_subdir_crop = os.path.join(self.output_dir, label)
            output_subdir_summary = os.path.join(self.output_dir, 'summary', label)
            self.ensure_dir(output_subdir_crop)
            self.ensure_dir(output_subdir_summary)

            for filename in os.listdir(input_subdir):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(input_subdir, filename)
                    print(f"Processing: {img_path}")
                    self.process_image(img_path, output_subdir_crop, output_subdir_summary, filename)


# Instantiate and run the segmenter
segmenter = AcetowhiteSegmenter(
    input_dir='../datasets/base_dataset_iarc_cropped/001_cropped',
    output_dir='../datasets/dataset_k/001_gmm_segmented_images',
    L_thresh=200
)
segmenter.run()
