"""
Data Ingestion Module.

Handles loading raw wine quality datasets, normalizing schema columns,
computing cryptographic integrity hashes (SHA-256), and storing snapshots.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_FEATURES = [
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
    "alcohol",
    "quality"
]


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for data versioning and traceability.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def ingest_data(
    raw_path: str | Path,
    output_path: str | Path | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load raw dataset, clean column naming, snapshot, and calculate metadata.

    Args:
        raw_path: Path to raw CSV file.
        output_path: Optional path to save normalized snapshot.

    Returns:
        Tuple containing the processed DataFrame and ingestion metadata dictionary.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_file}")

    file_hash = compute_file_hash(raw_file)
    logger.info("Ingesting raw data from %s (SHA-256: %s)", raw_file, file_hash[:12])

    # Semicolon delimiter used by UCI Wine Quality
    try:
        df = pd.read_csv(raw_file, sep=";")
        if df.shape[1] == 1:
            # Fallback to comma if semicolon produced a single column
            df = pd.read_csv(raw_file, sep=",")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}") from exc

    # Clean and normalize column names
    df.columns = [
        col.strip().strip('"').replace(" ", "_").lower()
        for col in df.columns
    ]

    metadata: Dict[str, Any] = {
        "source_path": str(raw_file),
        "sha256": file_hash,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns)
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        metadata["snapshot_path"] = str(out_file)
        logger.info("Saved normalized snapshot to %s", out_file)

    return df, metadata


if __name__ == "__main__":
    import sys
    raw_csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "winequality-red.csv"
    staging_csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "raw_snapshot.csv"
    data, meta = ingest_data(raw_csv, staging_csv)
    print(f"Successfully ingested {meta['rows']} rows.")
