"""
Data Validation and Quality Gates Module.

Implements automated schema validation, range bounding checks, outlier and
corrupted data isolation, and quarantine reporting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "fixed_acidity": (2.0, 20.0),
    "volatile_acidity": (0.05, 2.0),
    "citric_acid": (0.0, 1.5),
    "residual_sugar": (0.5, 30.0),
    "chlorides": (0.005, 1.0),
    "free_sulfur_dioxide": (1.0, 100.0),
    "total_sulfur_dioxide": (5.0, 350.0),
    "density": (0.97, 1.05),
    "ph": (2.5, 4.5),
    "sulphates": (0.2, 2.5),
    "alcohol": (7.0, 17.0),
    "quality": (3.0, 10.0),
}


class DataValidationError(Exception):
    """Raised when data fails quality gates threshold."""
    pass


def validate_dataset(
    df: pd.DataFrame,
    max_bad_fraction: float = 0.20,
    quarantine_dir: str | Path | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate DataFrame against domain boundaries and schema rules.

    Args:
        df: Input DataFrame to validate.
        max_bad_fraction: Maximum allowed bad row fraction before raising error.
        quarantine_dir: Optional path to save clean/rejected data and report.

    Returns:
        Tuple containing the clean DataFrame and a dictionary validation report.

    Raises:
        DataValidationError: If bad row fraction exceeds max_bad_fraction.
    """
    total_rows = len(df)
    if total_rows == 0:
        raise DataValidationError("Input dataset is empty.")

    issues = pd.DataFrame(index=df.index)

    # 1. Null check
    issues["has_null"] = df.isna().any(axis=1)

    # 2. Check each feature bounds
    for col, (min_val, max_val) in FEATURE_BOUNDS.items():
        if col in df.columns:
            issues[f"{col}_out_of_bounds"] = (df[col] < min_val) | (df[col] > max_val)
        else:
            logger.warning("Feature %s missing during validation.", col)
            issues[f"{col}_missing"] = True

    # 3. Duplicate check
    issues["is_duplicate"] = df.duplicated(keep="first")

    # Flag all bad rows
    is_bad = issues.any(axis=1)
    bad_count = int(is_bad.sum())
    bad_fraction = float(bad_count / total_rows)

    clean_df = df[~is_bad].copy()
    rejected_df = df[is_bad].copy()

    counts = {col: int(issues[col].sum()) for col in issues.columns if issues[col].sum() > 0}

    report: Dict[str, Any] = {
        "total_rows": total_rows,
        "clean_rows": len(clean_df),
        "rejected_rows": bad_count,
        "bad_fraction": round(bad_fraction, 4),
        "max_allowed_bad_fraction": max_bad_fraction,
        "issues_breakdown": counts,
        "status": "PASSED" if bad_fraction <= max_bad_fraction else "FAILED"
    }

    logger.info("Validation completed: %d clean, %d rejected (%.2f%% bad)",
                len(clean_df), bad_count, bad_fraction * 100)

    if quarantine_dir:
        q_path = Path(quarantine_dir)
        q_path.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(q_path / "clean_data.csv", index=False)
        rejected_df.to_csv(q_path / "rejected_data.csv", index=False)
        with open(q_path / "validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Quarantine artifacts saved to: %s", q_path)

    if bad_fraction > max_bad_fraction:
        raise DataValidationError(
            f"Dataset failed quality gate: {bad_fraction:.2%} rejected "
            f"(threshold: {max_bad_fraction:.2%})"
        )

    return clean_df, report


if __name__ == "__main__":
    from src.ingestion import ingest_data
    raw_csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "winequality-red.csv"
    data, _ = ingest_data(raw_csv)
    clean, rep = validate_dataset(data, quarantine_dir="data/processed/validation")
    print(rep)
