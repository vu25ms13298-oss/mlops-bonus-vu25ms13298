"""
Automated MLflow Experiment Matrix Runner.

Executes a grid of 10 distinct model configurations across 4 model families
(Logistic Regression, Random Forest, Gradient Boosting, SVM), logging all parameters,
metrics, and artifacts to MLflow, and automatically promotes the best performer to Production.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure S3/MinIO environment defaults if not provided
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "miniopassword")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")

from src.ingestion import ingest_data
from src.validation import validate_dataset
from src.preprocessing import preprocess_data
from src.train import train_and_track_experiment, promote_best_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ExperimentRunner")

EXPERIMENT_MATRIX: List[Dict[str, Any]] = [
    {
        "name": "LogisticRegression_Baseline",
        "algo": "logistic_regression",
        "params": {"C": 1.0, "max_iter": 500, "random_state": 42}
    },
    {
        "name": "LogisticRegression_L2_Strong",
        "algo": "logistic_regression",
        "params": {"C": 0.1, "max_iter": 500, "random_state": 42}
    },
    {
        "name": "LogisticRegression_L2_Weak",
        "algo": "logistic_regression",
        "params": {"C": 10.0, "max_iter": 500, "random_state": 42}
    },
    {
        "name": "RandomForest_Shallow_50Trees",
        "algo": "random_forest",
        "params": {"n_estimators": 50, "max_depth": 5, "min_samples_split": 4, "random_state": 42}
    },
    {
        "name": "RandomForest_Medium_100Trees",
        "algo": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "random_state": 42}
    },
    {
        "name": "RandomForest_Deep_200Trees",
        "algo": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2, "random_state": 42}
    },
    {
        "name": "GradientBoosting_Slow_50Trees",
        "algo": "gradient_boosting",
        "params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}
    },
    {
        "name": "GradientBoosting_Default_100Trees",
        "algo": "gradient_boosting",
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42}
    },
    {
        "name": "GradientBoosting_Fast_150Trees",
        "algo": "gradient_boosting",
        "params": {"n_estimators": 150, "learning_rate": 0.2, "max_depth": 5, "random_state": 42}
    },
    {
        "name": "SVM_RBF_Kernel",
        "algo": "svm",
        "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42}
    },
]


def resolve_tracking_uri(candidate_uri: str | None) -> str:
    """Validate if remote MLflow server is active; fallback to local mlruns if offline."""
    import requests
    target = candidate_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    if target.startswith("http"):
        try:
            resp = requests.get(f"{target}/health", timeout=1.5)
            if resp.status_code == 200:
                logger.info("Connected to remote MLflow server at: %s", target)
                return target
        except Exception:
            pass
        local_uri = f"file://{PROJECT_ROOT}/mlruns"
        logger.info("Remote MLflow server (%s) offline. Using local directory tracking: %s", target, local_uri)
        return local_uri
    return target


def run_experiment_matrix(
    data_path: str | Path | None = None,
    tracking_uri: str | None = None,
    register_best: bool = True
) -> pd.DataFrame:
    """Run full pipeline and execute all planned configurations."""
    raw_csv = Path(data_path or PROJECT_ROOT / "data" / "raw" / "winequality-red.csv")
    uri = resolve_tracking_uri(tracking_uri)

    logger.info("=== STEP 1: Ingestion & Validation ===")
    raw_df, _ = ingest_data(raw_csv, output_path=PROJECT_ROOT / "data" / "processed" / "raw_snapshot.csv")
    clean_df, _ = validate_dataset(raw_df, quarantine_dir=PROJECT_ROOT / "data" / "processed" / "validation")

    logger.info("=== STEP 2: Preprocessing ===")
    X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(
        clean_df, output_dir=PROJECT_ROOT / "data" / "processed"
    )

    logger.info("=== STEP 3: Executing %d Experiments ===", len(EXPERIMENT_MATRIX))
    results = []

    for idx, exp in enumerate(EXPERIMENT_MATRIX, start=1):
        logger.info("[%d/%d] Running: %s (%s)", idx, len(EXPERIMENT_MATRIX), exp["name"], exp["algo"])
        try:
            _, metrics, run_id = train_and_track_experiment(
                algo_name=exp["algo"],
                params=exp["params"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                run_name=exp["name"],
                tracking_uri=uri,
                register_model_name="wine_quality_model" if register_best else None,
                artifact_dir=PROJECT_ROOT / "data" / "processed" / "artifacts"
            )
            row = {
                "experiment_name": exp["name"],
                "algorithm": exp["algo"],
                "run_id": run_id[:8],
                **metrics
            }
            results.append(row)
        except Exception as exc:
            logger.error("Experiment %s failed: %s", exp["name"], exc)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        summary_path = PROJECT_ROOT / "data" / "processed" / "experiment_results.csv"
        results_df.to_csv(summary_path, index=False)

        logger.info("\n=== EXPERIMENT RESULTS MATRIX ===")
        print(results_df.to_string(index=False))

        best = results_df.iloc[0]
        logger.info("\n🏆 Best Performer: %s (F1: %.4f, Accuracy: %.4f)",
                    best["experiment_name"], best["f1_score"], best["accuracy"])

        if register_best:
            logger.info("=== STEP 4: Promoting Best Model to Production ===")
            promote_best_model(
                model_name="wine_quality_model",
                target_stage="Production",
                metric_name="f1_score",
                tracking_uri=uri
            )

    return results_df


if __name__ == "__main__":
    run_experiment_matrix()
