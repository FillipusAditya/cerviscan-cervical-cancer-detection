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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from yellowbrick.model_selection import RFECV

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def split_data(csv_file_path, output_save):
    """
    Split the CSV dataset into train and test, apply RFECV feature selection,
    and save the resulting train/test CSVs and RFECV visualization.
    """
    csv_file = pd.read_csv(csv_file_path)
    
    if 'label' not in csv_file.columns:
        raise ValueError(f"Column 'label' not found in file {csv_file_path}")
    
    # Drop 'Image' column if exists
    if 'Image' in csv_file.columns:
        csv_file.drop(columns=['Image'], inplace=True)
        
    x_source = csv_file.drop('label', axis='columns')
    y_source = csv_file['label']
    
    # Map labels to binary
    y_source = y_source.replace({'abnormal': 1, 'normal': 0})
    
    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x_source, y_source, test_size=0.2, random_state=123)
    
    # Recursive Feature Elimination with Cross Validation
    estimator = xgb.XGBClassifier()
    visualizer = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
    visualizer.fit(x_train, y_train)
    visualizer.show(outpath=os.path.join(output_save, "./rfecv_visualization.png"))
    plt.close()
    
    # Keep only selected features
    mask = visualizer.support_
    x_train = x_train.loc[:, mask]
    x_test = x_test.loc[:, mask]
    
    # Save split datasets
    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    
    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)
    
    return x_train, x_test, y_train, y_test


def get_random_grid():
    """
    Return a valid hyperparameter grid for XGBClassifier.
    """
    random_grid = {
        'learning_rate': np.arange(0.01, 0.2, 0.01),
        'min_child_weight': np.arange(0, 5, 1),
        'min_split_loss': np.arange(0, 5, 1),
        'max_depth': np.arange(3, 10, 1),
        'reg_lambda': [2]  # <-- Corrected typo: was 'reg_lamxbda'
    }
    return random_grid


def get_best_model(cv, verbose, n_jobs, random_grid, x_train, y_train, output_save):
    """
    Perform GridSearchCV to find the best hyperparameters for XGBClassifier.
    Save the best model to disk as pickle.
    """
    xgb_model = xgb.XGBClassifier()
    xgb_grid_search = GridSearchCV(
        xgb_model,
        random_grid,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        scoring='accuracy'
    )
    
    xgb_grid_search.fit(x_train, y_train)
    
    best_model = xgb_grid_search.best_estimator_
    filename = os.path.join(output_save, 'xgb_best')
    pickle.dump(best_model, open(filename, 'wb'))
    
    return best_model


def get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_save):
    """
    Save best hyperparameters, feature importance ranking,
    confusion matrix, performance metrics, and related plots.
    """
    # Best Params
    best_params_df = pd.DataFrame([best_model.get_params()])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_predict)

    # Performance Metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    performance_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, specificity, recall, f1]
    }
    df_performance = pd.DataFrame(performance_metrics)

    # Feature Importances
    feature_importances = best_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': x_test.columns.tolist(),
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Train vs Test Accuracy
    train_accuracy = best_model.score(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test)
    
    df_train_test_accuracy = pd.DataFrame({
        'Dataset': ['Train Accuracy', 'Test Accuracy', 'Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Accuracy': [train_accuracy, test_accuracy, accuracy, precision, specificity, recall, f1]
    })
    df_train_test_accuracy.to_csv(os.path.join(output_save, 'test_train_accuracy.csv'), index=False)

    # Plot Confusion Matrix and Performance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[0],
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    axes[0].set_title('Confusion Matrix')

    # Performance Metrics Barplot
    ax = sns.barplot(x='Metric', y='Value', data=df_performance, palette='Blues_d', ax=axes[1])
    axes[1].set_title('Performance Metrics')
    axes[1].set_ylabel('Value')
    axes[1].set_ylim(0, 1)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9), textcoords='offset points')

    # Save files
    best_params_df.to_csv(os.path.join(output_save, 'model_best_params.csv'), index=False)
    features_df.to_csv(os.path.join(output_save, 'features_ranking.csv'), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_save, 'confusion_matrix_and_performance_metrics.png'), dpi=300)
    plt.close()


def otomatis(csv_file, output_path):
    """
    Orchestrate the pipeline: Split, Grid Search, Evaluate, and Save all results.
    """
    os.makedirs(output_path, exist_ok=True)
    x_train, x_test, y_train, y_test = split_data(csv_file, output_path)

    random_grid = get_random_grid()
    best_model = get_best_model(
        cv=5,
        verbose=3,
        random_grid=random_grid,
        n_jobs=1,
        x_train=x_train,
        y_train=y_train,
        output_save=output_path
    )

    y_predict = best_model.predict(x_test)
    get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_path)


def main():
    """
    Process multiple CSV files inside a folder and run the full pipeline for each.
    """
    folder_path = "../result_track/HASIL PERCOBAAN/S1K1PRFE"
    data_path = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for data in tqdm(data_path, desc="Processing files", unit="file"):
        try:
            output_path = os.path.join(folder_path, os.path.basename(data)[:-4])
            otomatis(csv_file=data, output_path=output_path)
        except Exception as e:
            print(f"Error processing {data}: {e}")

main()
