"""
============================================
📌 XGBoost Classification Pipeline with RFECV Feature Selection (Python)
============================================

Description:
------------
This program implements a full machine learning pipeline for binary 
classification using the XGBoost algorithm. The pipeline integrates 
automatic feature selection using RFECV (Recursive Feature Elimination 
with Cross-Validation), hyperparameter tuning, model evaluation, and 
comprehensive result export.

The program is designed for datasets containing engineered features 
(e.g., texture, color, statistical descriptors), commonly used in 
computer vision and medical imaging applications such as lesion analysis, 
cervical cancer screening, plant disease detection, and general 
pattern recognition tasks.

Pipeline Features:
------------------
✅ Automatically removes unwanted image IDs  
✅ Performs RFECV feature selection using XGBoost  
✅ Saves RFECV plot as `rfecv_visualization.png`  
✅ Splits dataset into training and testing sets  
✅ Performs GridSearchCV hyperparameter tuning  
✅ Saves the best XGBoost model as `.pkl`  
✅ Computes evaluation metrics:
   - Accuracy  
   - Precision  
   - Recall  
   - Specificity  
   - F1 Score  
✅ Generates:
   - Confusion matrix  
   - Performance bar chart  
   - Feature ranking CSV  
   - Model best parameters CSV  
   - Train/Test accuracy report  

Usage:
------
1. Provide the path to the feature CSV file.
2. Define an output directory to store results.
3. Run the pipeline using:

   run_xgboost_pipeline(csv_file, output_path)

Example:
--------
from xgboost_rfecv_pipeline import run_xgboost_pipeline

csv_path = "dataset/features/TEXTURES.csv"
save_path = "results/xgboost_rfecv/textures"

run_xgboost_pipeline(csv_path, save_path)

Author:
-------
Fillipus Aditya Nugroho

============================================
"""

import os
import glob
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from yellowbrick.model_selection import RFECV

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def split_data(csv_file_path, output_save, remove_ids=None):
    """
    Load dataset, remove unwanted image IDs, perform feature selection using RFECV,
    and split the data into training/testing sets.

    Parameters
    ----------
    csv_file_path : str
        Path to the input CSV file containing features and label.
    output_save : str
        Directory where split and visualization results will be saved.
    remove_ids : list, optional
        List of image IDs to exclude from training (if column 'image_id' exists).

    Returns
    -------
    x_train, x_test, y_train, y_test : DataFrame
        Training and testing data after RFECV feature selection.
    """
    csv_file = pd.read_csv(csv_file_path)

    # Remove unwanted IDs if provided
    if remove_ids is not None and 'image_id' in csv_file.columns:
        csv_file = csv_file[~csv_file['image_id'].isin(remove_ids)]

    if 'label' not in csv_file.columns:
        raise ValueError(f"Column 'label' not found in file {csv_file_path}")

    # Drop image_id column if present
    if 'image_id' in csv_file.columns:
        csv_file.drop(columns=['image_id'], inplace=True)

    x_source = csv_file.drop('label', axis=1)
    y_source = csv_file['label'].replace({'abnormal': 1, 'normal': 0})

    x_train, x_test, y_train, y_test = train_test_split(
        x_source, y_source, test_size=0.2, random_state=123
    )

    # RFECV (recursive feature elimination with cross-validation)
    estimator = xgb.XGBClassifier()
    visualizer = RFECV(estimator=estimator, step=1, cv=5, scoring='recall')
    visualizer.fit(x_train, y_train)
    visualizer.show(outpath=os.path.join(output_save, "rfecv_visualization.png"))
    plt.close()

    # Select best features
    mask = visualizer.support_
    x_train = x_train.loc[:, mask]
    x_test = x_test.loc[:, mask]

    # Save to CSV
    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test


def get_random_grid():
    """
    Generate a predefined hyperparameter grid for XGBoost GridSearchCV.

    Returns
    -------
    dict
        Hyperparameter search space.
    """
    return {
        'learning_rate': np.arange(0.01, 0.2, 0.01),
        'min_child_weight': np.arange(0, 5, 1),
        'min_split_loss': np.arange(0, 5, 1),
        'max_depth': np.arange(3, 10, 1),
        'reg_lambda': [2]
    }


def get_best_model(cv, verbose, n_jobs, random_grid, x_train, y_train, output_save):
    """
    Train XGBoost using GridSearchCV to obtain the best model.

    Parameters
    ----------
    cv : int
        Number of cross-validation folds.
    verbose : int
        Verbosity level.
    n_jobs : int
        Number of parallel jobs.
    random_grid : dict
        Hyperparameter search space.
    x_train : DataFrame
        Training features.
    y_train : DataFrame or Series
        Training labels.
    output_save : str
        Directory to save the trained model.

    Returns
    -------
    object
        Best estimator obtained from GridSearchCV.
    """
    xgb_model = xgb.XGBClassifier()
    xgb_grid_search = GridSearchCV(
        xgb_model, random_grid, cv=cv, verbose=verbose,
        n_jobs=n_jobs, scoring='recall'
    )
    xgb_grid_search.fit(x_train, y_train)

    best_model = xgb_grid_search.best_estimator_
    pickle.dump(best_model, open(os.path.join(output_save, 'xgb_best.pkl'), 'wb'))

    return best_model


def get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_save):
    """
    Generate model evaluation outputs including:
    - best hyperparameters
    - confusion matrix
    - performance metrics (accuracy, precision, specificity, recall, F1)
    - feature importance ranking
    - plots and CSV outputs

    Parameters
    ----------
    best_model : XGBClassifier
        Best model selected from GridSearchCV.
    x_train, y_train, x_test, y_test : DataFrame or Series
        Train/test splits.
    y_predict : array-like
        Model predictions on test data.
    output_save : str
        Directory to save evaluation outputs.
    """
    best_params_df = pd.DataFrame([best_model.get_params()])
    cm = confusion_matrix(y_test, y_predict)

    # Compute metrics manually
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    df_performance = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, specificity, recall, f1]
    })

    # Feature importance
    feature_importances = best_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': x_test.columns.tolist(),
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Save train/test accuracy
    train_accuracy = best_model.score(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test)

    df_train_test_accuracy = pd.DataFrame({
        'Dataset': ['Train Accuracy', 'Test Accuracy', 'Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Accuracy': [train_accuracy, test_accuracy, accuracy, precision, specificity, recall, f1]
    })
    df_train_test_accuracy.to_csv(os.path.join(output_save, 'test_train_accuracy.csv'), index=False)

    # Plot confusion matrix and metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[0],
                xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    axes[0].set_title('Confusion Matrix')

    ax = sns.barplot(x='Metric', y='Value', data=df_performance, palette='Blues_d', ax=axes[1])
    axes[1].set_title('Performance Metrics')
    axes[1].set_ylabel('Value')
    axes[1].set_ylim(0, 1)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9), textcoords='offset points')

    # Save outputs
    best_params_df.to_csv(os.path.join(output_save, 'model_best_params.csv'), index=False)
    features_df.to_csv(os.path.join(output_save, 'features_ranking.csv'), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_save, 'confusion_matrix_and_performance_metrics.png'), dpi=300)
    plt.close()


def run_full_pipeline(csv_file, output_path):
    """
    Execute the full XGBoost training pipeline:
    1. Load and split data
    2. RFECV feature selection
    3. Hyperparameter tuning (GridSearchCV)
    4. Model evaluation and reporting

    Parameters
    ----------
    csv_file : str
        Path to the input CSV file.
    output_path : str
        Directory where all outputs will be stored.
    """
    os.makedirs(output_path, exist_ok=True)

    remove_ids = [
        "AAC1", "AAD1", "AAF1", "AAH1", "AAR1", "ABI1", "ABP1", "ABR1",
        "ACE1", "ACF1", "ADE1", "ADH1", "ADY1", "AEM1", "API1", "AGO1",
        "AOP1", "AIO1", "AHS1", "AFA1", "AEB1", "ADC1", "ADA1"
    ]

    x_train, x_test, y_train, y_test = split_data(csv_file, output_path, remove_ids=remove_ids)

    random_grid = get_random_grid()
    best_model = get_best_model(
        cv=5, verbose=3, random_grid=random_grid, n_jobs=1,
        x_train=x_train, y_train=y_train, output_save=output_path
    )

    y_predict = best_model.predict(x_test)
    get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_path)


def main():
    csv_file = ""
    output_path = ""
    run_full_pipeline(csv_file, output_path)


if __name__ == "__main__":
    main()
