"""
Data Preprocessing and Feature Engineering Module.

Handles feature selection, train-test splitting with stratified sampling,
standardization scaling, and reproducibility artifact persistence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol"
]
TARGET_COL = "quality"


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str | Path | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Dict[str, Any]]:
    """Transform raw DataFrame into scaled train and test arrays.

    Args:
        df: Input clean DataFrame.
        test_size: Fraction of data reserved for evaluation.
        random_state: Random seed for deterministic reproducibility.
        output_dir: Optional path to save processed artifacts.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler, metadata).
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # 1. Feature extraction
    X = df[FEATURE_NAMES].copy()
    
    # 2. Binary target: 1 = Good wine (quality >= 6), 0 = Normal wine (quality < 6)
    y = (df[TARGET_COL] >= 6).astype(int).values

    logger.info("Dataset shape: %d samples, %d features", X.shape[0], X.shape[1])
    logger.info("Target distribution: Class 1 (Good): %d, Class 0 (Normal): %d",
                int(np.sum(y == 1)), int(np.sum(y == 0)))

    # 3. Stratified split for balanced target ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4. Standard scaling (fit on train only to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metadata: Dict[str, Any] = {
        "train_samples": int(len(X_train_scaled)),
        "test_samples": int(len(X_test_scaled)),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "test_size": test_size,
        "random_state": random_state,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "X_train.npy", X_train_scaled)
        np.save(out_path / "X_test.npy", X_test_scaled)
        np.save(out_path / "y_train.npy", y_train)
        np.save(out_path / "y_test.npy", y_test)
        joblib.dump(scaler, out_path / "scaler.joblib")
        logger.info("Saved preprocessed artifacts to: %s", out_path)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, metadata


if __name__ == "__main__":
    from src.ingestion import ingest_data
    from src.validation import validate_dataset
    raw_csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "winequality-red.csv"
    raw_df, _ = ingest_data(raw_csv)
    clean_df, _ = validate_dataset(raw_df, max_bad_fraction=0.3)
    X_tr, X_te, y_tr, y_te, sc, meta = preprocess_data(clean_df, output_dir="data/processed")
    print(f"Preprocessing completed: train={len(X_tr)}, test={len(X_te)}")
