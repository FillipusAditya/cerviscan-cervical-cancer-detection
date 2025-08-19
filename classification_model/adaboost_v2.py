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
    csv_file = pd.read_csv(csv_file_path)

    # Filter rows if image_id is specified and remove_ids is not None
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
    return {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'learning_rate': np.arange(0.01, 2.1, 0.1),
        'estimator': [DecisionTreeClassifier(max_depth=d) for d in range(1, 6)]
    }

def get_best_model(cv, verbose, n_jobs, random_grid, x_train, y_train, output_save):
    ada_model = AdaBoostClassifier()
    ada_grid_search = GridSearchCV(
        ada_model, random_grid, cv=cv, verbose=verbose,
        n_jobs=n_jobs, scoring='accuracy'
    )
    ada_grid_search.fit(x_train, y_train)
    best_model = ada_grid_search.best_estimator_
    pickle.dump(best_model, open(os.path.join(output_save, 'adaboost_best'), 'wb'))
    return best_model

def get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_save):
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

def otomatis(csv_file, output_path):
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
    # csv_file = "../datasets/dataset_g/cropped_image_after_iva_reference_1/features/segmented/LAB_LBP_GLRLM_TAMURA.csv"
    csv_file = "../datasets/dataset_d/features/YUV_LBP_GLRLM_TAMURA.csv"
    # output_path = "../datasets/dataset_g/cropped_image_after_iva_reference_1/classification_result/segmented/lab_lbp_glrlm_tamura_adaboost_norfecv"
    output_path = "../datasets/dataset_d/classification_result/clean_yuv_lbp_glrlm_tamura_adaboost_norfecv"
    otomatis(csv_file, output_path)

if __name__ == "__main__":
    main()
