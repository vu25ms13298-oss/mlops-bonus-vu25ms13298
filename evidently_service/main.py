"""
Evidently AI Drift Detection and Monitoring Service.

Provides:
  - Data drift calculation comparing production stream against baseline reference
  - Data quality metrics calculation
  - Prometheus metrics exposure for Grafana visualization
  - HTML & JSON report generation and archiving
"""
from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
except (ImportError, ModuleNotFoundError):
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.legacy.metrics import DatasetDriftMetric, ColumnDriftMetric

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvidentlyService")

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", str(BASE_DIR / "reference")))

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = float(os.getenv("EVIDENTLY_DRIFT_THRESHOLD", "0.1"))
MIN_SAMPLES_FOR_ANALYSIS = int(os.getenv("EVIDENTLY_MIN_SAMPLES", "30"))

# ============================================
# PROMETHEUS METRICS
# ============================================
DRIFT_DETECTED = Gauge(
    "evidently_data_drift_detected",
    "Overall data drift detected status (1=drifted, 0=no drift)"
)

DRIFT_SCORE = Gauge(
    "evidently_drift_score",
    "Proportion of features showing statistical drift"
)

FEATURE_DRIFT = Gauge(
    "evidently_feature_drift",
    "Individual feature drift status (1=drifted, 0=normal)",
    ["feature_name"]
)

DRIFTED_FEATURES_COUNT = Gauge(
    "evidently_drifted_features_count",
    "Total count of drifted features"
)

ANALYSIS_COUNT = Counter(
    "evidently_analysis_total",
    "Total drift calculations executed"
)

ANALYSIS_DURATION = Histogram(
    "evidently_analysis_duration_seconds",
    "Time taken to execute drift calculation",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

MISSING_VALUES = Gauge(
    "evidently_missing_values_ratio",
    "Missing values ratio per feature in production stream",
    ["feature_name"]
)


# ============================================
# PYDANTIC SCHEMAS
# ============================================
class SingleCapture(BaseModel):
    features: Dict[str, float]
    prediction: Optional[int] = None
    timestamp: Optional[str] = None
    model_version: Optional[str] = None


class BatchCapture(BaseModel):
    data: List[Dict[str, Any]]


class ReferenceUpload(BaseModel):
    data: List[Dict[str, Any]]
    feature_names: Optional[List[str]] = None
    description: Optional[str] = None


class AnalysisRequest(BaseModel):
    window_size: Optional[int] = Field(default=100, description="Window of recent samples to analyze")
    threshold: Optional[float] = Field(default=DRIFT_THRESHOLD, description="Drift threshold")


# ============================================
# DATA STORE
# ============================================
class DataStore:
    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
        self.production_samples: List[Dict[str, Any]] = []
        self.last_analysis_time: Optional[str] = None
        self._load_reference()

    def _load_reference(self):
        ref_file = REFERENCE_DIR / "reference_data.csv"
        if ref_file.exists():
            try:
                self.reference_data = pd.read_csv(ref_file)
                logger.info("Loaded reference data (%d rows) from %s", len(self.reference_data), ref_file)
            except Exception as exc:
                logger.error("Failed loading reference data: %s", exc)

    def set_reference(self, df: pd.DataFrame, description: str = ""):
        self.reference_data = df.copy()
        ref_file = REFERENCE_DIR / "reference_data.csv"
        self.reference_data.to_csv(ref_file, index=False)
        with open(REFERENCE_DIR / "metadata.json", "w") as f:
            json.dump({
                "description": description,
                "rows": len(df),
                "features": list(df.columns),
                "timestamp": datetime.utcnow().isoformat()
            }, f, indent=2)
        logger.info("Saved new reference baseline with %d samples.", len(df))

    def add_production(self, row: Dict[str, Any]):
        self.production_samples.append(row)
        # Ring buffer: cap at 10,000 to manage RAM
        if len(self.production_samples) > 10000:
            self.production_samples = self.production_samples[-10000:]

    def get_production_df(self, window_size: Optional[int] = None) -> pd.DataFrame:
        if not self.production_samples:
            return pd.DataFrame()
        recent = self.production_samples[-window_size:] if window_size else self.production_samples
        return pd.DataFrame(recent)


store = DataStore()
app = FastAPI(
    title="Evidently AI Drift Monitoring Service",
    description="Real-time statistical drift analysis, quality tracking, and report generation.",
    version="2.0.0"
)


# ============================================
# CORE DRIFT COMPUTATION
# ============================================
def run_drift_calculation(window_size: int, threshold: float) -> Dict[str, Any]:
    """Execute Evidently report comparing production against baseline reference."""
    start_time = time.time()

    if store.reference_data is None:
        raise HTTPException(status_code=400, detail="Reference baseline dataset has not been initialized.")

    prod_df = store.get_production_df(window_size=window_size)
    if len(prod_df) < MIN_SAMPLES_FOR_ANALYSIS:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient production samples ({len(prod_df)}). Minimum required is {MIN_SAMPLES_FOR_ANALYSIS}"
        )

    # Align columns to numerical features present in reference
    common_cols = [c for c in store.reference_data.columns if c in prod_df.columns and c != "quality"]
    if len(common_cols) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No matching features between reference ({list(store.reference_data.columns)[:5]}...) and production ({list(prod_df.columns)[:5]}...)"
        )
    ref_sub = store.reference_data[common_cols].dropna()
    prod_sub = prod_df[common_cols].dropna()

    report = Report(metrics=[
        DataDriftPreset(drift_share=threshold),
        DataQualityPreset()
    ])
    report.run(reference_data=ref_sub, current_data=prod_sub)

    duration = time.time() - start_time
    ANALYSIS_COUNT.inc()
    ANALYSIS_DURATION.observe(duration)

    dict_report = report.as_dict()

    # Extract drift metrics
    metrics_list = dict_report.get("metrics", [])
    dataset_drift_val = 0
    drift_score_val = 0.0
    drifted_count = 0
    feature_drift_dict = {}

    for m in metrics_list:
        if m.get("metric") == "DatasetDriftMetric":
            res = m.get("result", {})
            dataset_drift_val = 1 if res.get("dataset_drift") else 0
            drift_score_val = float(res.get("drift_share", 0.0))
            drifted_count = int(res.get("number_of_drifted_columns", 0))
            drift_by_col = res.get("drift_by_columns", {})
            for col_name, info in drift_by_col.items():
                is_drifted = 1 if info.get("drift_detected") else 0
                feature_drift_dict[col_name] = is_drifted
                FEATURE_DRIFT.labels(feature_name=col_name).set(is_drifted)

    DRIFT_DETECTED.set(dataset_drift_val)
    DRIFT_SCORE.set(drift_score_val)
    DRIFTED_FEATURES_COUNT.set(drifted_count)

    # Save HTML & JSON report artifacts
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    json_path = REPORTS_DIR / f"drift_report_{ts}.json"
    report.save_html(str(html_path))
    with open(json_path, "w") as jf:
        json.dump(dict_report, jf, indent=2)

    store.last_analysis_time = ts

    return {
        "timestamp": ts,
        "dataset_drift_detected": bool(dataset_drift_val),
        "drift_share": drift_score_val,
        "number_of_drifted_columns": drifted_count,
        "drift_by_columns": feature_drift_dict,
        "html_report": html_path.name,
        "duration_seconds": round(duration, 3)
    }


# ============================================
# ENDPOINTS
# ============================================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "reference_loaded": store.reference_data is not None,
        "reference_samples": len(store.reference_data) if store.reference_data is not None else 0,
        "production_samples": len(store.production_samples),
        "last_analysis": store.last_analysis_time,
        "reports_count": len(list(REPORTS_DIR.glob("*.html")))
    }


@app.post("/reference")
def set_reference(payload: ReferenceUpload):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list is empty.")
    df = pd.DataFrame(payload.data)
    store.set_reference(df, description=payload.description or "User uploaded reference")
    return {"message": "Reference baseline updated successfully.", "samples": len(df), "features": list(df.columns)}


@app.get("/reference")
def get_reference():
    if store.reference_data is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "samples": len(store.reference_data),
        "columns": list(store.reference_data.columns)
    }


@app.post("/capture")
def capture(payload: SingleCapture):
    row = dict(payload.features)
    if payload.prediction is not None:
        row["prediction"] = payload.prediction
    row["captured_at"] = payload.timestamp or datetime.utcnow().isoformat()
    store.add_production(row)
    return {"status": "captured", "total_samples": len(store.production_samples)}


@app.post("/capture/batch")
def capture_batch(payload: BatchCapture):
    for r in payload.data:
        store.add_production(r)
    return {"status": "batch_captured", "total_samples": len(store.production_samples)}


@app.post("/analyze")
def trigger_analysis(req: AnalysisRequest):
    result = run_drift_calculation(
        window_size=req.window_size or 100,
        threshold=req.threshold or DRIFT_THRESHOLD
    )
    return result


@app.get("/reports")
def list_reports():
    reports = sorted([f.name for f in REPORTS_DIR.glob("*.html")], reverse=True)
    return {"reports": reports}


@app.get("/reports/{report_name}")
def get_report(report_name: str):
    target = REPORTS_DIR / report_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return HTMLResponse(content=target.read_text(encoding="utf-8"))


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
