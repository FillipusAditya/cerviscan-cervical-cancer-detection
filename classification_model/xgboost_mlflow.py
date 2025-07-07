import os
import pickle
import warnings
import json

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

import mlflow
import mlflow.xgboost

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from yellowbrick.model_selection import RFECV

from tqdm import tqdm

# Suppress future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)


class XGBPipeline:
    """
    An end-to-end pipeline for:
    - Loading dataset from CSV
    - Splitting and feature selecting with RFECV
    - Hyperparameter tuning for XGBoost with GridSearchCV
    - Experiment tracking with MLflow
    - Saving results, metrics, model, and predictions as artifacts
    """

    def __init__(self, csv_file: str, output_path: str,
                 experiment_name: str, run_name: str, artifact_dir: str):
        """
        Initialize pipeline paths, experiment settings, and variables.
        """
        self.csv_file = csv_file
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.y_predict = None

    def split_data(self):
        """
        Step 1:
        - Read dataset
        - Encode labels to binary
        - Train-test split with stratification
        - Apply RFECV for feature selection using XGBoost
        - Save selected splits
        """
        print("Step 1/3: Reading dataset and performing split + RFECV feature selection...")
        df = pd.read_csv(self.csv_file)

        if 'label' not in df.columns:
            raise ValueError(f"Column 'label' not found in file {self.csv_file}")

        # Drop 'image_id' column if exists to avoid leaking IDs
        if 'image_id' in df.columns:
            df = df.sort_values(by='image_id').reset_index(drop=True)
            df.drop(columns=['image_id'], inplace=True)

        X = df.drop('label', axis=1)
        y = df['label'].replace({'abnormal': 1, 'normal': 0})

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=123, stratify=y
        )

        # Use XGBClassifier for RFECV since it supports feature_importances_
        estimator = xgb.XGBClassifier(eval_metric='logloss')
        visualizer = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
        visualizer.fit(self.x_train, self.y_train)
        visualizer.show(outpath=os.path.join(self.artifact_dir, "rfecv_visualization.png"))
        plt.close()

        # Mask features that are selected
        mask = visualizer.support_
        self.x_train = self.x_train.loc[:, mask]
        self.x_test = self.x_test.loc[:, mask]

        # Save train and test sets after feature selection
        pd.concat([self.x_train, self.y_train], axis=1).to_csv(
            os.path.join(self.output_path, 'train.csv'), index=False)
        pd.concat([self.x_test, self.y_test], axis=1).to_csv(
            os.path.join(self.output_path, 'test.csv'), index=False)

        print("Step 1/3 complete: Split data saved.")

    def get_param_grid(self):
        """
        Return hyperparameter grid for XGBoost tuning.
        """
        return {
            'learning_rate': np.arange(0.01, 0.2, 0.01),
            'min_child_weight': np.arange(0, 5, 1),
            'min_split_loss': np.arange(0, 5, 1),
            'max_depth': np.arange(3, 10, 1),
            'reg_lambda': [2]
        }

    def tune_with_gridsearch_mlflow(self):
        """
        Step 2:
        - Perform GridSearchCV tuning on XGBoost
        - Log best model, confusion matrices, metrics, and predictions to MLflow
        """
        print("Step 2/3: Starting GridSearchCV tuning with MLflow tracking...")
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name) as run:
            param_grid = self.get_param_grid()
            model = xgb.XGBClassifier(eval_metric='logloss')

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='accuracy',
                cv=5,
                verbose=0,
                n_jobs=3
            )

            n_candidates = np.prod([len(v) for v in param_grid.values()])
            print(f"Total parameter combinations: {n_candidates}")

            print("Running GridSearchCV...")
            with tqdm(total=n_candidates, desc="GridSearch Progress") as pbar:
                grid_search.fit(self.x_train, self.y_train)
                pbar.update(n_candidates)

            self.best_model = grid_search.best_estimator_

            # Evaluate train and test predictions
            y_train_pred = self.best_model.predict(self.x_train)
            y_test_pred = self.best_model.predict(self.x_test)

            cm_train, train_metrics = self.get_evaluation(self.y_train, y_train_pred)
            cm_test, test_metrics = self.get_evaluation(self.y_test, y_test_pred)

            # Log model to MLflow
            mlflow.xgboost.log_model(self.best_model, name='xgboost_model',
                                     input_example=self.x_train.iloc[0:1])

            # Confusion matrix for training set
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3)
            ax3.set_xlabel("Predicted label")
            ax3.set_ylabel("True label")
            ax3.set_title("Confusion Matrix - Train")
            fig3.savefig(f"{self.artifact_dir}/cm_train.png")
            mlflow.log_artifact(f"{self.artifact_dir}/cm_train.png")

            # Confusion matrix for test set
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
            ax4.set_xlabel("Predicted label")
            ax4.set_ylabel("True label")
            ax4.set_title("Confusion Matrix - Test")
            fig4.savefig(f"{self.artifact_dir}/cm_test.png")
            mlflow.log_artifact(f"{self.artifact_dir}/cm_test.png")

            # Save predictions as CSV
            df_test_pred = pd.DataFrame({"y_test": self.y_test, "y_test_pred": y_test_pred})
            df_test_pred.to_csv(f"{self.artifact_dir}/predictions_test.csv", index=False)
            mlflow.log_artifact(f"{self.artifact_dir}/predictions_test.csv")

            df_train_pred = pd.DataFrame({"y_train": self.y_train, "y_train_pred": y_train_pred})
            df_train_pred.to_csv(f"{self.artifact_dir}/predictions_train.csv", index=False)
            mlflow.log_artifact(f"{self.artifact_dir}/predictions_train.csv")

            # Save best params as JSON
            clean_params = {k: (int(v) if isinstance(v, np.integer)
                                else float(v) if isinstance(v, np.floating)
                                else v)
                            for k, v in grid_search.best_params_.items()}

            with open(f"{self.artifact_dir}/best_params.json", "w") as f:
                json.dump(clean_params, f, indent=4)
            mlflow.log_artifact(f"{self.artifact_dir}/best_params.json")

            # Save model as pickle for offline use
            with open(f"{self.artifact_dir}/best_model.pkl", "wb") as f:
                pickle.dump(self.best_model, f)
            mlflow.log_artifact(f"{self.artifact_dir}/best_model.pkl")

            # Log evaluation metrics to MLflow
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            print(f"Step 2/3 complete: Tuning complete. MLflow Run ID: {run.info.run_id}")

    def get_evaluation(self, y_true, y_pred):
        """
        Compute confusion matrix and core classification metrics.
        """
        cm = confusion_matrix(y_true, y_pred)
        metric = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1-score': f1_score(y_true, y_pred)
        }
        return cm, metric

    def run(self):
        """
        Run the full XGBPipeline: split data, tune, evaluate, and log all artifacts.
        """
        print("XGBPipeline started.")
        self.split_data()
        self.tune_with_gridsearch_mlflow()
        print("XGBPipeline finished. All artifacts saved and logged to MLflow.")


def main():
    """
    Entry point: define paths, set MLflow URI, and run the pipeline.
    """
    csv_file = "../../datasets/dataset_k/features/YUV_LBP_GLRLM_TAMURA.csv"
    output_dir = "./yuv_xgboost_no_deletion"
    experiment_name = "GMM Segmentation Experiments"
    run_name = "20250630_dataset_k_yuv_all_features_gmm_xgboost_no_deletion"
    artifact_dir = os.path.join(output_dir, "artifacts")

    root_dir = r"D:\CerviScan Machine Learning Model\mlruns"
    os.makedirs(root_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{root_dir.replace(os.sep, '/')}")

    pipeline = XGBPipeline(csv_file, output_dir, experiment_name, run_name, artifact_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
