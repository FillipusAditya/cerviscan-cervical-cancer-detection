import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from yellowbrick.model_selection import RFECV
import optuna

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def split_data(csv_file_path, output_save, remove_ids=None):
    print("[INFO] Membaca dataset...")
    csv_file = pd.read_csv(csv_file_path)

    if remove_ids is not None and 'image_id' in csv_file.columns:
        print(f"[INFO] Menghapus {len(remove_ids)} data berdasarkan image_id...")
        csv_file = csv_file[~csv_file['image_id'].isin(remove_ids)]

    if 'label' not in csv_file.columns:
        raise ValueError(f"Column 'label' tidak ditemukan di file {csv_file_path}")
    if 'image_id' in csv_file.columns:
        csv_file.drop(columns=['image_id'], inplace=True)

    x_source = csv_file.drop('label', axis=1)
    y_source = csv_file['label'].replace({'abnormal': 1, 'normal': 0})

    print("[INFO] Split data train-test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x_source, y_source, test_size=0.2, random_state=123)

    print("[INFO] Melakukan Feature Selection dengan RFECV...")
    estimator = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
    visualizer = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
    visualizer.fit(x_train, y_train)
    visualizer.show(outpath=os.path.join(output_save, "rfecv_visualization.png"))
    plt.close()

    mask = visualizer.support_
    x_train = x_train.loc[:, mask]
    x_test = x_test.loc[:, mask]

    print(f"[INFO] Jumlah fitur terpilih: {x_train.shape[1]}")

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test

# 🔹 Optuna Objective Function
def objective(trial, x_train, y_train):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,  # supaya output dari cross_val_score tidak terlalu bising
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }

    model = xgb.XGBClassifier(**param)
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()

def get_best_model(x_train, y_train, output_save, n_trials=50):
    print(f"[INFO] Memulai Hyperparameter Tuning dengan Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=n_trials)

    print("[INFO] Hyperparameter terbaik ditemukan!")
    print(study.best_trial.params)

    best_params = study.best_trial.params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 1  # tampilkan progres training
    })

    best_model = xgb.XGBClassifier(**best_params)
    print("[INFO] Melatih model terbaik dengan data training...")
    best_model.fit(x_train, y_train, verbose=True)

    pickle.dump(best_model, open(os.path.join(output_save, 'xgb_best.pkl'), 'wb'))
    pd.DataFrame([study.best_trial.params]).to_csv(
        os.path.join(output_save, "model_best_params.csv"), index=False
    )

    return best_model

def get_final_data(best_model, x_train, y_train, x_test, y_test, output_save):
    print("[INFO] Evaluasi model...")
    y_predict = best_model.predict(x_test)
    y_proba = best_model.predict_proba(x_test)[:, 1]

    cm = confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"[INFO] Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    df_performance = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC'],
        'Value': [accuracy, precision, specificity, recall, f1, roc_auc, pr_auc]
    })

    features_df = pd.DataFrame({
        'Feature': x_test.columns.tolist(),
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    df_performance.to_csv(os.path.join(output_save, "performance_metrics.csv"), index=False)
    features_df.to_csv(os.path.join(output_save, 'features_ranking.csv'), index=False)

    # 🔹 Confusion Matrix + Bar Metrics
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[0],
                xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    axes[0].set_title("Confusion Matrix")

    ax = sns.barplot(x='Metric', y='Value', data=df_performance, palette='Blues_d', ax=axes[1])
    axes[1].set_title("Performance Metrics")
    axes[1].set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9), textcoords='offset points')

    # 🔹 ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    axes[2].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set_title("ROC Curve")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_save, 'evaluation_plots.png'), dpi=300)
    plt.close()

    # 🔹 PR Curve separately
    plt.figure(figsize=(8,6))
    plt.plot(rec, prec, label=f"PR-AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(output_save, 'pr_curve.png'), dpi=300)
    plt.close()

def otomatis(csv_file, output_path):
    os.makedirs(output_path, exist_ok=True)
    remove_ids = [
        "AAC1", "AAD1", "AAF1", "AAH1", "AAR1", "ABI1", "ABP1", "ABR1",
        "ACE1", "ACF1", "ADE1", "ADH1", "ADY1", "AEM1", "API1", "AGO1",
        "AOP1", "AIO1", "AHS1", "AFA1", "AEB1", "ADC1", "ADA1"
    ]
    print("[INFO] Memulai pipeline otomatis...")
    x_train, x_test, y_train, y_test = split_data(csv_file, output_path, remove_ids=remove_ids)
    best_model = get_best_model(x_train, y_train, output_path, n_trials=50)
    get_final_data(best_model, x_train, y_train, x_test, y_test, output_path)
    print("[INFO] Proses selesai! Semua hasil disimpan di:", output_path)

def main():
    csv_file = "../datasets/dataset_f/features/LAB_LBP_GLRLM_TAMURA.csv"
    output_path = "../datasets/dataset_f/classification_result/clean_lab_lbp_glrlm_tamura_optuna"
    otomatis(csv_file, output_path)

if __name__ == "__main__":
    main()
