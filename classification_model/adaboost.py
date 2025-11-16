"""
============================================
📌 AdaBoost Classification Pipeline with Feature Evaluation (Python)
============================================

Description:
------------
This program implements a complete machine learning pipeline for binary 
classification using the AdaBoost algorithm. It supports automated data 
preprocessing, hyperparameter tuning, model training, evaluation, and 
export of all results.

The pipeline is designed to work with datasets containing extracted image 
features (e.g., texture, color, morphological features), commonly used in 
computer vision and medical imaging tasks such as cervical cancer detection, 
lesion classification, and pattern recognition.

Pipeline Features:
------------------
✅ Automatically removes unwanted image IDs  
✅ Splits dataset into training and testing sets  
✅ Performs GridSearchCV hyperparameter tuning (AdaBoost)  
✅ Saves the best model as `.pkl`  
✅ Computes classification metrics:
   - Accuracy  
   - Precision  
   - Recall  
   - Specificity  
   - F1 Score  
✅ Generates confusion matrix and performance visualization  
✅ Computes feature importance ranking  
✅ Export results:
   - train/test dataset CSV  
   - model_best_params.csv  
   - features_ranking.csv  
   - test_train_accuracy.csv  
   - confusion_matrix_and_performance_metrics.png  

Usage:
------
1. Provide the path to the input CSV file (containing features + label).
2. Set the output directory for saving the results.
3. Call `run_full_pipeline(csv_file, output_path)` to run the complete flow.

Example:
--------
from adaboost_pipeline import run_full_pipeline

csv_path = "dataset/TEXTURES.csv"
save_path = "results/adaboost/textures"

run_full_pipeline(csv_path, save_path)

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
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def split_data(csv_file_path, output_save, remove_ids=None):
    """
    Load dataset from CSV, optionally remove specific image IDs, 
    split into train and test sets, and save them to output folder.

    Parameters
    ----------
    csv_file_path : str
        Path to the input CSV file.
    output_save : str
        Directory to save train.csv and test.csv.
    remove_ids : list or None
        List of image_id strings to be excluded (optional).

    Returns
    -------
    x_train, x_test, y_train, y_test : pandas DataFrame / Series
        Train-test split of features and labels.
    """
    csv_file = pd.read_csv(csv_file_path)

    if remove_ids is not None and 'image_id' in csv_file.columns:
        csv_file = csv_file[~csv_file['image_id'].isin(remove_ids)]

    if 'label' not in csv_file.columns:
        raise ValueError(f"Column 'label' not found in file {csv_file_path}")

    if 'image_id' in csv_file.columns:
        csv_file.drop(columns=['image_id'], inplace=True)

    x_source = csv_file.drop('label', axis=1)
    y_source = csv_file['label'].replace({'abnormal': 1, 'normal': 0})

    x_train, x_test, y_train, y_test = train_test_split(
        x_source, y_source, test_size=0.2, random_state=123
    )

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test


def get_random_grid():
    """
    Generate random hyperparameter grid for AdaBoost tuning.

    Returns
    -------
    dict
        Hyperparameter search space for GridSearchCV.
    """
    return {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'learning_rate': np.arange(0.01, 2.1, 0.1),
        'estimator': [DecisionTreeClassifier(max_depth=d) for d in range(1, 6)]
    }


def get_best_model(cv, verbose, n_jobs, random_grid, x_train, y_train, output_save):
    """
    Perform hyperparameter tuning using GridSearchCV and save the best model.

    Parameters
    ----------
    cv : int
        Number of cross-validation folds.
    verbose : int
        Verbosity level.
    n_jobs : int
        Number of CPU cores to use.
    random_grid : dict
        Hyperparameter search space.
    x_train, y_train : DataFrame/Series
        Training dataset.
    output_save : str
        Directory to save the trained model.

    Returns
    -------
    best_model : AdaBoostClassifier
        Best estimator obtained from grid search.
    """
    ada_model = AdaBoostClassifier()
    ada_grid_search = GridSearchCV(
        ada_model, random_grid, cv=cv, verbose=verbose,
        n_jobs=n_jobs, scoring='recall'
    )
    ada_grid_search.fit(x_train, y_train)
    best_model = ada_grid_search.best_estimator_

    pickle.dump(best_model, open(os.path.join(output_save, 'adaboost_best.pkl'), 'wb'))
    return best_model


def get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_save):
    """
    Compute evaluation metrics, generate plots, feature importance table,
    and save all results to the output directory.

    Parameters
    ----------
    best_model : AdaBoostClassifier
        Trained model.
    x_train, x_test : pandas DataFrame
        Train and test feature sets.
    y_train, y_test : pandas Series
        Train and test labels.
    y_predict : array-like
        Predicted labels from the model.
    output_save : str
        Directory to save evaluation results.

    Returns
    -------
    None
    """
    best_params_df = pd.DataFrame([best_model.get_params()])
    cm = confusion_matrix(y_test, y_predict)

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

    feature_importances = best_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': x_test.columns.tolist(),
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    train_accuracy = best_model.score(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test)

    df_train_test_accuracy = pd.DataFrame({
        'Dataset': ['Train Accuracy', 'Test Accuracy', 'Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Accuracy': [train_accuracy, test_accuracy, accuracy, precision, specificity, recall, f1]
    })
    df_train_test_accuracy.to_csv(os.path.join(output_save, 'test_train_accuracy.csv'), index=False)

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

    best_params_df.to_csv(os.path.join(output_save, 'model_best_params.csv'), index=False)
    features_df.to_csv(os.path.join(output_save, 'features_ranking.csv'), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_save, 'confusion_matrix_and_performance_metrics.png'), dpi=300)
    plt.close()


def run_full_pipeline(csv_file, output_path):
    """
    Execute the complete classification pipeline:
    - Create output directory
    - Remove specific image IDs
    - Split dataset
    - Hyperparameter tuning (GridSearch)
    - Evaluation + visualization + export results

    Parameters
    ----------
    csv_file : str
        Input CSV file path containing features and labels.
    output_path : str
        Directory to save all outputs.

    Returns
    -------
    None
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
        cv=5, verbose=3, random_grid=random_grid, n_jobs=-1,
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
