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
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def split_data(csv_file_path, output_save, remove_ids=None):
    print("[INFO] Membaca dataset...")
    csv_file = pd.read_csv(csv_file_path)

    # Filter rows jika image_id ada dan remove_ids diberikan
    if remove_ids is not None and 'image_id' in csv_file.columns:
        print(f"[INFO] Menghapus {len(remove_ids)} data berdasarkan ID...")
        csv_file = csv_file[~csv_file['image_id'].isin(remove_ids)]

    if 'label' not in csv_file.columns:
        raise ValueError(f"Kolom 'label' tidak ditemukan pada file {csv_file_path}")
    if 'image_id' in csv_file.columns:
        csv_file.drop(columns=['image_id'], inplace=True)

    print("[INFO] Split data train-test...")
    x_source = csv_file.drop('label', axis=1)
    y_source = csv_file['label'].replace({'abnormal': 1, 'normal': 0})

    x_train, x_test, y_train, y_test = train_test_split(
        x_source, y_source, test_size=0.2, random_state=123)

    print("[INFO] Melakukan RFECV untuk feature selection...")
    estimator = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    visualizer = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
    visualizer.fit(x_train, y_train)
    visualizer.show(outpath=os.path.join(output_save, "rfecv_visualization.png"))
    plt.close()

    mask = visualizer.support_
    x_train = x_train.loc[:, mask]
    x_test = x_test.loc[:, mask]

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)

    print(f"[INFO] Data train: {x_train.shape}, Data test: {x_test.shape}")
    return x_train, x_test, y_train, y_test


def get_stacking_model(cv, verbose, n_jobs, x_train, y_train, output_save):
    print("[INFO] Membuat base learners untuk StackingClassifier...")

    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('svc', SVC(probability=True, kernel='rbf', random_state=42))
    ]

    meta_model = LogisticRegression(max_iter=2000, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

    print("[INFO] Training StackingClassifier...")
    stacking_clf.fit(x_train, y_train)
    print("[INFO] Training selesai ✅")

    pickle.dump(stacking_clf, open(os.path.join(output_save, 'stacking_model.pkl'), 'wb'))
    print(f"[INFO] Model disimpan ke {os.path.join(output_save, 'stacking_model.pkl')}")
    return stacking_clf


def get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_save):
    print("[INFO] Menghitung metrik evaluasi...")

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

    feature_importances = None
    if hasattr(best_model, "feature_importances_"):
        feature_importances = best_model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': x_test.columns.tolist(),
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        features_df.to_csv(os.path.join(output_save, 'features_ranking.csv'), index=False)

    train_accuracy = best_model.score(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test)

    df_train_test_accuracy = pd.DataFrame({
        'Dataset': ['Train Accuracy', 'Test Accuracy', 'Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Accuracy': [train_accuracy, test_accuracy, accuracy, precision, specificity, recall, f1]
    })
    df_train_test_accuracy.to_csv(os.path.join(output_save, 'test_train_accuracy.csv'), index=False)

    print("[INFO] Membuat visualisasi hasil...")
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

    plt.tight_layout()
    plt.savefig(os.path.join(output_save, 'confusion_matrix_and_performance_metrics.png'), dpi=300)
    plt.close()

    print("[INFO] Semua hasil evaluasi tersimpan ✅")


def otomatis(csv_file, output_path):
    os.makedirs(output_path, exist_ok=True)

    remove_ids = [
        "AAC1", "AAD1", "AAF1", "AAH1", "AAR1", "ABI1", "ABP1", "ABR1",
        "ACE1", "ACF1", "ADE1", "ADH1", "ADY1", "AEM1", "API1", "AGO1",
        "AOP1", "AIO1", "AHS1", "AFA1", "AEB1", "ADC1", "ADA1"
    ]

    x_train, x_test, y_train, y_test = split_data(csv_file, output_path, remove_ids=remove_ids)

    best_model = get_stacking_model(
        cv=5, verbose=1, random_grid=None, n_jobs=-1,
        x_train=x_train, y_train=y_train, output_save=output_path
    )

    print("[INFO] Prediksi data test...")
    y_predict = best_model.predict(x_test)
    get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_path)


def main():
    # Sesuaikan path dataset dan output
    csv_file = "../datasets/dataset_f/features/LAB_LBP_GLRLM_TAMURA.csv"
    output_path = "../datasets/dataset_f/classification_result/001_run/stacking_rfecv/clean_LAB_lbp_glrlm_tamura"

    otomatis(csv_file, output_path)


if __name__ == "__main__":
    main()
