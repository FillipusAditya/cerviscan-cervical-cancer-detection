import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_score

import optuna
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ===============================================================
# 🔹 Split Data
# ===============================================================
def split_data(csv_file_path, output_save, remove_ids=None):
    print("[INFO] Membaca dataset...")
    csv_file = pd.read_csv(csv_file_path)

    if remove_ids is not None and 'image_id' in csv_file.columns:
        csv_file = csv_file[~csv_file['image_id'].isin(remove_ids)]

    if 'label' not in csv_file.columns:
        raise ValueError(f"Column 'label' not ditemukan di file {csv_file_path}")
    if 'image_id' in csv_file.columns:
        csv_file.drop(columns=['image_id'], inplace=True)

    x_source = csv_file.drop('label', axis=1)
    y_source = csv_file['label'].replace({'abnormal': 1, 'normal': 0})

    print("[INFO] Split train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x_source, y_source, test_size=0.2, random_state=123
    )

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test


# ===============================================================
# 🔹 Optuna Objective
# ===============================================================
def objective(trial, x_train, y_train):
    # hyperparam untuk DecisionTree (base learner)
    dt_params = {
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)
    }

    # base estimator
    base_tree = DecisionTreeClassifier(**dt_params)

    # hyperparam untuk AdaBoost
    ab_params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        "estimator": base_tree
    }

    model = AdaBoostClassifier(**ab_params)
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()


# ===============================================================
# 🔹 Dapatkan Best Model
# ===============================================================
def get_best_model(x_train, y_train, output_save, n_trials=50):
    print("[INFO] Hyperparameter tuning dengan Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=n_trials)

    print("[INFO] Best Trial:", study.best_trial.params)

    # Pisahkan parameter untuk DT & AdaBoost
    best_params = study.best_trial.params
    dt_params = {k: best_params[k] for k in ["max_depth", "min_samples_split", "min_samples_leaf"]}
    ab_params = {k: best_params[k] for k in ["n_estimators", "learning_rate"]}

    base_tree = DecisionTreeClassifier(**dt_params)
    best_model = AdaBoostClassifier(estimator=base_tree, **ab_params)

    print("[INFO] Training model terbaik...")
    best_model.fit(x_train, y_train)

    # simpan model & params
    pickle.dump(best_model, open(os.path.join(output_save, 'adaboost_best.pkl'), 'wb'))
    pd.DataFrame([best_params]).to_csv(
        os.path.join(output_save, "model_best_params.csv"), index=False
    )

    return best_model


# ===============================================================
# 🔹 Evaluasi + Simpan Grafik
# ===============================================================
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

    # 🔹 Confusion Matrix & Metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
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

    plt.tight_layout()
    plt.savefig(os.path.join(output_save, 'confusion_matrix_and_metrics.png'), dpi=300)
    plt.close()

    # 🔹 ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_save, 'roc_curve.png'), dpi=300)
    plt.close()

    # 🔹 PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(rec, prec, label=f"PR-AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(output_save, 'pr_curve.png'), dpi=300)
    plt.close()

    print("[INFO] Evaluasi selesai. Hasil disimpan di folder output.")


# ===============================================================
# 🔹 Pipeline
# ===============================================================
def otomatis(csv_file, output_path):
    os.makedirs(output_path, exist_ok=True)

    remove_ids = [
        "AAC1", "AAD1", "AAF1", "AAH1", "AAR1", "ABI1", "ABP1", "ABR1",
        "ACE1", "ACF1", "ADE1", "ADH1", "ADY1", "AEM1", "API1", "AGO1",
        "AOP1", "AIO1", "AHS1", "AFA1", "AEB1", "ADC1", "ADA1"
    ]

    x_train, x_test, y_train, y_test = split_data(csv_file, output_path, remove_ids=remove_ids)
    best_model = get_best_model(x_train, y_train, output_path, n_trials=50)
    get_final_data(best_model, x_train, y_train, x_test, y_test, output_path)


# ===============================================================
# 🔹 Main
# ===============================================================
def main():
    csv_file = "../datasets/dataset_d/features/LAB_LBP_GLRLM_TAMURA.csv"
    output_path = "../datasets/dataset_d/classification_result/001_run/adaboost_optuna/clean_lab_lbp_glrlm_tamura_adaboost_optuna"
    otomatis(csv_file, output_path)


if __name__ == "__main__":
    main()
