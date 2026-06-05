import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Heart Disease Classification - MLflow Project"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="heart_preprocessing/heart_preprocessed.csv",
        help="Path ke dataset preprocessed CSV"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Jumlah estimator Random Forest"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Kedalaman maksimum pohon"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporsi data test"
    )
    return parser.parse_args()

def load_data(data_file, target_col="target"):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_file}")

    df = pd.read_csv(data_file)
    print(f"✅ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"   Distribusi target: {df[target_col].value_counts().to_dict()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1_score" : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_pred_proba)
    }

def main():
    args = parse_args()

    print("=" * 55)
    print("  Heart Disease - MLflow Project Training")
    print(f"  data_file    : {args.data_file}")
    print(f"  n_estimators : {args.n_estimators}")
    print(f"  max_depth    : {args.max_depth}")
    print(f"  test_size    : {args.test_size}")
    print("=" * 55)

    # Load data
    X, y = load_data(args.data_file)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = args.test_size,
        random_state = 42,
        stratify     = y
    )
    print(f"\n📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    with mlflow.start_run():

        # Log parameter
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth",    args.max_depth)
        mlflow.log_param("test_size",    args.test_size)
        mlflow.log_param("model_type",   "RandomForestClassifier")
        mlflow.log_param("data_file",    args.data_file)

        # Training
        model = RandomForestClassifier(
            n_estimators = args.n_estimators,
            max_depth    = args.max_depth,
            random_state = 42,
            n_jobs       = -1
        )
        model.fit(X_train, y_train)

        # Evaluasi
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1_score",  metrics["f1_score"])
        mlflow.log_metric("roc_auc",   metrics["roc_auc"])

        # Log model sebagai artefak
        mlflow.sklearn.log_model(
            sk_model      = model,
            artifact_path = "model",
            registered_model_name="heart-disease-rf"
        )

        # Print hasil
        print("\n📊 Hasil Evaluasi:")
        for k, v in metrics.items():
            print(f"   {k:<12}: {v:.4f}")

        print(f"\n✅ Run selesai! Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
