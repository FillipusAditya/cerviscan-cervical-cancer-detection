import os
import pickle
import warnings
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from yellowbrick.model_selection import RFECV

from tqdm import tqdm

# Ignore future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)


class AdaBoostPipeline:
    """
    End-to-end pipeline for:
    - Reading feature dataset
    - Splitting data and applying RFECV for feature selection
    - Hyperparameter tuning of AdaBoostClassifier using GridSearchCV
    - Experiment tracking and artifact logging with MLflow
    - Saving predictions and evaluation results
    """

    def __init__(self, csv_file: str, output_path: str,
                 experiment_name: str, run_name: str, artifact_dir: str):
        """
        Initialize the pipeline with input dataset path, output folders, 
        MLflow experiment and run names.
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

    def split_data(self):
        """
        Step 1:
        - Load dataset from CSV
        - Encode labels as binary
        - Apply train-test split
        - Apply feature selection using RFECV
        - Save selected train and test splits
        """
        print("Step 1/3: Reading dataset and performing train-test split + RFECV feature selection...")
        df = pd.read_csv(self.csv_file)

        if 'label' not in df.columns:
            raise ValueError(f"Column 'label' not found in file {self.csv_file}")

        # Drop unused ID column if exists
        if 'image_id' in df.columns:
            df = df.sort_values(by='image_id').reset_index(drop=True)
            df.drop(columns=['image_id'], inplace=True)

        X = df.drop('label', axis=1)
        y = df['label'].replace({'abnormal': 1, 'normal': 0})

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=123, stratify=y
        )

        # Use XGBClassifier for RFECV as it provides feature_importances_
        from xgboost import XGBClassifier
        estimator = XGBClassifier(eval_metric='logloss')
        visualizer = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
        visualizer.fit(self.x_train, self.y_train)
        visualizer.show(outpath=os.path.join(self.artifact_dir, "rfecv_visualization.png"))
        plt.close()

        # Mask only selected features
        mask = visualizer.support_
        self.x_train = self.x_train.loc[:, mask]
        self.x_test = self.x_test.loc[:, mask]

        pd.concat([self.x_train, self.y_train], axis=1).to_csv(
            os.path.join(self.output_path, 'train.csv'), index=False)
        pd.concat([self.x_test, self.y_test], axis=1).to_csv(
            os.path.join(self.output_path, 'test.csv'), index=False)

        print("Step 1/3 complete: Data split and saved.")

    def get_param_grid(self):
        """
        Define parameter grid for AdaBoost hyperparameter tuning.
        """
        return {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
            'estimator__max_depth': [1, 2, 3]
        }

    def tune_with_gridsearch_mlflow(self):
        """
        Step 2:
        - Perform GridSearchCV for AdaBoost hyperparameter tuning.
        - Log all artifacts (model, confusion matrix, predictions) to MLflow.
        - Log best params and metrics.
        """
        print("Step 2/3: Starting hyperparameter tuning with GridSearchCV + MLflow tracking...")
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name) as run:
            param_grid = self.get_param_grid()
            base_estimator = DecisionTreeClassifier(random_state=42)
            model = AdaBoostClassifier(estimator=base_estimator, random_state=42)

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

            # Evaluate predictions for train and test sets
            y_train_pred = self.best_model.predict(self.x_train)
            y_test_pred = self.best_model.predict(self.x_test)

            cm_train, train_metrics = self.get_evaluation(self.y_train, y_train_pred)
            cm_test, test_metrics = self.get_evaluation(self.y_test, y_test_pred)

            # Log model to MLflow
            mlflow.sklearn.log_model(self.best_model, "adaboost_model")

            # Log confusion matrix plots
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3)
            ax3.set_xlabel("Predicted label")
            ax3.set_ylabel("True label")
            ax3.set_title("Confusion Matrix - Train")
            fig3.savefig(f"{self.artifact_dir}/cm_train.png")
            mlflow.log_artifact(f"{self.artifact_dir}/cm_train.png")

            fig4, ax4 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
            ax4.set_xlabel("Predicted label")
            ax4.set_ylabel("True label")
            ax4.set_title("Confusion Matrix - Test")
            fig4.savefig(f"{self.artifact_dir}/cm_test.png")
            mlflow.log_artifact(f"{self.artifact_dir}/cm_test.png")

            # Save predictions
            df_test_pred = pd.DataFrame({"y_test": self.y_test, "y_test_pred": y_test_pred})
            df_test_pred.to_csv(f"{self.artifact_dir}/predictions_test.csv", index=False)
            mlflow.log_artifact(f"{self.artifact_dir}/predictions_test.csv")

            df_train_pred = pd.DataFrame({"y_train": self.y_train, "y_train_pred": y_train_pred})
            df_train_pred.to_csv(f"{self.artifact_dir}/predictions_train.csv", index=False)
            mlflow.log_artifact(f"{self.artifact_dir}/predictions_train.csv")

            # Save best parameters as JSON
            clean_params = {k: (int(v) if isinstance(v, np.integer)
                                else float(v) if isinstance(v, np.floating)
                                else v)
                            for k, v in grid_search.best_params_.items()}

            with open(f"{self.artifact_dir}/best_params.json", "w") as f:
                json.dump(clean_params, f, indent=4)
            mlflow.log_artifact(f"{self.artifact_dir}/best_params.json")

            # Save best model as pickle
            with open(f"{self.artifact_dir}/best_model.pkl", "wb") as f:
                pickle.dump(self.best_model, f)
            mlflow.log_artifact(f"{self.artifact_dir}/best_model.pkl")

            # Log metrics
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            print(f"Step 2/3 complete: Tuning finished. MLflow Run ID: {run.info.run_id}")

    def get_evaluation(self, y_true, y_pred):
        """
        Helper function: Compute confusion matrix and key metrics.
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
        Run the entire pipeline end-to-end.
        """
        print("AdaBoostPipeline started.")
        self.split_data()
        self.tune_with_gridsearch_mlflow()
        print("AdaBoostPipeline completed. All artifacts saved and logged to MLflow.")


def main():
    """
    Main entry point for running the pipeline.
    """
    csv_file = "../../datasets/dataset_k/features/LAB_LBP_GLRLM_TAMURA.csv"
    output_dir = "./lab_adaboost_no_deletion"
    experiment_name = "GMM Segmentation Experiments"
    run_name = "20250701_dataset_k_lab_all_features_gmm_adaboost_no_deletion"
    artifact_dir = os.path.join(output_dir, "artifacts")

    root_dir = r"D:\CerviScan Machine Learning Model\mlruns"
    os.makedirs(root_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{root_dir.replace(os.sep, '/')}")

    pipeline = AdaBoostPipeline(csv_file, output_dir, experiment_name, run_name, artifact_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
