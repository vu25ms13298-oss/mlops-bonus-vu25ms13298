"""
Model Training and MLflow Experiment Tracking Module.

Supports multiple scikit-learn architectures, parameter/metric logging,
artifact logging, and automated MLflow Model Registry promotion.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_estimator(algo_name: str, params: Dict[str, Any]):
    """Instantiate model architecture based on algorithm name and params."""
    algo = algo_name.lower()
    if algo == "random_forest":
        return RandomForestClassifier(**params)
    elif algo == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    elif algo == "logistic_regression":
        return LogisticRegression(**params)
    elif algo in ("svm", "svc"):
        return SVC(probability=True, **params)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")


def train_and_track_experiment(
    algo_name: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "wine_quality_experiment",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    register_model_name: str | None = None,
    artifact_dir: str | Path = "data/processed/artifacts"
) -> Tuple[Any, Dict[str, float], str]:
    """Train estimator, log to MLflow, and return trained model with metrics.

    Args:
        algo_name: Name of algorithm.
        params: Model hyperparameters.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        experiment_name: MLflow experiment name.
        run_name: Custom run name for MLflow.
        tracking_uri: Optional MLflow server tracking URI.
        register_model_name: Optional model registry name.
        artifact_dir: Directory to save plots before logging.

    Returns:
        Tuple of (model, metrics_dict, run_id).
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow Tracking URI set to: %s", tracking_uri)

    mlflow.set_experiment(experiment_name)

    model = build_estimator(algo_name, params)
    model.fit(X_train, y_train)

    run_title = run_name or f"{algo_name}_{params.get('random_state', 42)}"

    with mlflow.start_run(run_name=run_title) as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run %s (ID: %s)", run_title, run_id)

        # 1. Log hyperparameters and tags
        mlflow.log_params(params)
        mlflow.set_tag("algorithm", algo_name)
        mlflow.set_tag("dataset", "winequality-red")

        # 2. Evaluate and log metrics
        metrics, artifacts = evaluate_model(model, X_test, y_test, artifact_dir=artifact_dir)
        mlflow.log_metrics(metrics)

        # 3. Log visual artifacts
        if "confusion_matrix_path" in artifacts:
            mlflow.log_artifact(artifacts["confusion_matrix_path"])

        # 4. Infer signature and log model
        signature = infer_signature(X_train[:5], model.predict(X_train[:5]))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature
        )

        if register_model_name:
            try:
                model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri=model_uri, name=register_model_name)
                logger.info("Registered model %s from %s", register_model_name, model_uri)
            except Exception as reg_exc:
                logger.warning("Could not register model: %s", reg_exc)

        logger.info("Completed MLflow run %s: F1=%.4f, Acc=%.4f",
                    run_title, metrics["f1_score"], metrics["accuracy"])

    return model, metrics, run_id


def promote_best_model(
    model_name: str,
    target_stage: str = "Production",
    metric_name: str = "f1_score",
    experiment_name: str = "wine_quality_experiment",
    tracking_uri: str | None = None
) -> str | None:
    """Find the best run for an experiment and promote its registered version.

    Args:
        model_name: Registered model name.
        target_stage: Stage to promote to (Production, Staging).
        metric_name: Primary metric for comparison.
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI.

    Returns:
        Version number promoted, or None if failed.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    try:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            logger.warning("Experiment %s not found.", experiment_name)
            return None

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        if not runs:
            logger.warning("No runs found in experiment %s", experiment_name)
            return None

        best_run = runs[0]
        best_run_id = best_run.info.run_id
        best_score = best_run.data.metrics.get(metric_name, 0.0)
        logger.info("Best run %s has %s = %.4f", best_run_id, metric_name, best_score)

        # Find registered model version matching best_run_id
        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = None
        for v in versions:
            if v.run_id == best_run_id:
                target_version = v.version
                break

        if target_version:
            try:
                client.set_registered_model_alias(
                    name=model_name,
                    alias=target_stage.lower(),
                    version=target_version
                )
                logger.info("Set model alias '%s' -> v%s for %s", target_stage.lower(), target_version, model_name)
            except Exception as e:
                logger.debug("Model alias set error (ignored): %s", e)

            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=target_version,
                    stage=target_stage,
                    archive_existing_versions=True
                )
                logger.info("Promoted %s v%s to stage '%s'", model_name, target_version, target_stage)
            except Exception as e:
                logger.debug("Stage transition notice: %s", e)

            return str(target_version)
        else:
            logger.warning("No registered version found for run %s", best_run_id)
            return None

    except Exception as exc:
        logger.warning("Model promotion encountered an error: %s", exc)
        return None
