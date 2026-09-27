"""
FastAPI Model Serving API with Prometheus Metrics and Evidently Drift Hook.

Endpoints:
  - GET  /health      : Service and model readiness status
  - POST /predict     : Single prediction with real-time metric collection
  - POST /predict/batch: Batch prediction
  - GET  /metrics     : Prometheus scrape endpoint
  - GET  /model/info  : Current deployed model metadata
"""
from __future__ import annotations

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelAPI")

# ============================================
# CONFIGURATION
# ============================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "wine_quality_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
EVIDENTLY_SERVICE_URL = os.getenv("EVIDENTLY_SERVICE_URL", "http://evidently:8001")

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

# ============================================
# PROMETHEUS METRICS
# ============================================
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total inference predictions completed",
    ["model_name", "model_version"]
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model inference computation latency",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5]
)

PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of output prediction values (0=Normal, 1=Good)",
    ["model_name"],
    buckets=[0.0, 0.5, 1.0]
)

FEATURE_VALUE = Histogram(
    "model_feature_value",
    "Real-time distribution of input feature values",
    ["feature_name"],
    buckets=[-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
)

CURRENT_MODEL_VERSION = Gauge(
    "model_version_info",
    "Current active model version in production",
    ["model_name", "version"]
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time taken to load model into memory",
    ["model_name"]
)

PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total prediction exceptions encountered",
    ["model_name", "error_type"]
)

# ============================================
# PYDANTIC SCHEMAS
# ============================================
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="List of 11 wine features in order"
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Optional feature names matching the 11 feature inputs"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
                "feature_names": FEATURE_NAMES
            }
        }


class BatchPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(..., description="List of sample feature vectors")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 1 (Good Wine) or 0 (Normal)")
    confidence: Optional[float] = Field(None, description="Prediction probability")
    model_name: str
    model_version: str
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float


# ============================================
# MODEL MANAGER
# ============================================
class ModelManager:
    """Manages model loading from MLflow Registry with fallback support."""

    def __init__(self):
        self.model = None
        self.model_name = MODEL_NAME
        self.model_version = "none"
        self.load_time = 0.0
        self.load_model()

    def load_model(self) -> bool:
        """Attempt to load model from MLflow registry; fallback to baseline dummy if uninitialized."""
        start = time.time()
        
        # Fast check if remote server is responding
        server_alive = False
        if MLFLOW_TRACKING_URI.startswith("http"):
            try:
                resp = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=1.0)
                if resp.status_code == 200:
                    server_alive = True
            except Exception:
                server_alive = False
        else:
            server_alive = True

        if server_alive:
            try:
                import mlflow
                import mlflow.pyfunc

                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                uri = f"models:/{self.model_name}/{MODEL_STAGE}"
                logger.info("Attempting to load model from URI: %s", uri)

                self.model = mlflow.pyfunc.load_model(uri)

                client = mlflow.tracking.MlflowClient()
                versions = client.get_latest_versions(self.model_name, stages=[MODEL_STAGE])
                self.model_version = str(versions[0].version) if versions else "1"

                self.load_time = time.time() - start
                CURRENT_MODEL_VERSION.labels(model_name=self.model_name, version=self.model_version).set(1)
                MODEL_LOAD_TIME.labels(model_name=self.model_name).set(self.load_time)
                logger.info("Model %s v%s successfully loaded in %.2fs!", self.model_name, self.model_version, self.load_time)
                return True
            except Exception as exc:
                logger.warning("Could not load from MLflow (%s). Falling back.", exc)

        logger.info("MLflow server offline or uninitialized. Initializing baseline heuristic estimator.")
        # Heuristic fallback: wine with alcohol > 10.5 and volatile_acidity < 0.5 is good
        class HeuristicModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return ((X[:, 10] >= 10.5) & (X[:, 1] <= 0.55)).astype(int)

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                preds = self.predict(X)
                probas = np.zeros((len(X), 2))
                for i, p in enumerate(preds):
                    probas[i] = [0.2, 0.8] if p == 1 else [0.8, 0.2]
                return probas

        self.model = HeuristicModel()
        self.model_version = "fallback-heuristic"
        self.load_time = time.time() - start
        return True

    def predict(self, features: List[float]) -> Tuple[int, Optional[float], float]:
        """Perform single prediction and return (class, probability, latency_seconds)."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        arr = np.array(features, dtype=float).reshape(1, -1)
        start = time.time()
        raw_pred = self.model.predict(arr)
        pred_class = int(raw_pred[0])

        proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = float(self.model.predict_proba(arr)[0][1])
            except Exception:
                proba = None

        latency = time.time() - start

        # Metrics updates
        PREDICTION_COUNT.labels(model_name=self.model_name, model_version=self.model_version).inc()
        PREDICTION_LATENCY.labels(model_name=self.model_name).observe(latency)
        PREDICTION_VALUE.labels(model_name=self.model_name).observe(pred_class)

        return pred_class, proba, latency


# ============================================
# FASTAPI APP & MIDDLEWARE
# ============================================
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Production-grade ML serving API with real-time Prometheus monitoring and Evidently AI drift integration.",
    version="2.0.0"
)

manager = ModelManager()
app_start_time = time.time()


@app.on_event("startup")
def startup_event():
    """Initialize model during server startup."""
    manager.load_model()


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """Intercept HTTP calls and record request metrics."""
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    status = str(response.status_code)
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(latency)
    return response


def forward_sample_to_evidently(features: List[float], pred: int):
    """Background task: push live prediction samples to Evidently drift monitor."""
    try:
        sample_dict = {name: val for name, val in zip(FEATURE_NAMES, features)}
        payload = {
            "features": sample_dict,
            "prediction": pred,
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": manager.model_version
        }
        requests.post(f"{EVIDENTLY_SERVICE_URL}/capture", json=payload, timeout=2)
    except Exception as exc:
        logger.debug("Failed pushing sample to Evidently: %s", exc)


# ============================================
# ROUTE HANDLERS
# ============================================
@app.get("/")
def index():
    return {
        "service": "Wine Quality ML Inference Service",
        "status": "operational",
        "model": manager.model_name,
        "version": manager.model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health readiness check for orchestrators and container healthchecks."""
    is_healthy = manager.model is not None
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=is_healthy,
        model_name=manager.model_name,
        model_version=manager.model_version,
        uptime_seconds=round(time.time() - app_start_time, 2)
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, background_tasks: BackgroundTasks):
    """Perform single model prediction on 11 input features."""
    if len(payload.features) != len(FEATURE_NAMES):
        PREDICTION_ERRORS.labels(model_name=manager.model_name, error_type="invalid_feature_count").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(FEATURE_NAMES)} features, received {len(payload.features)}"
        )

    # Record feature distribution values
    f_names = payload.feature_names or FEATURE_NAMES
    for name, val in zip(f_names, payload.features):
        FEATURE_VALUE.labels(feature_name=name).observe(val)

    try:
        pred_class, proba, latency = manager.predict(payload.features)
    except Exception as exc:
        PREDICTION_ERRORS.labels(model_name=manager.model_name, error_type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failure: {exc}")

    # Forward sample to Evidently in background
    background_tasks.add_task(forward_sample_to_evidently, payload.features, pred_class)

    return PredictionResponse(
        prediction=pred_class,
        confidence=proba,
        model_name=manager.model_name,
        model_version=manager.model_version,
        latency_ms=round(latency * 1000, 2),
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictionRequest):
    """Batch inference endpoint."""
    predictions = []
    for sample in payload.samples:
        if len(sample) != len(FEATURE_NAMES):
            raise HTTPException(status_code=422, detail="Each sample must contain exactly 11 features.")
        pred_class, proba, _ = manager.predict(sample)
        predictions.append({"prediction": pred_class, "confidence": proba})
    return {"batch_size": len(predictions), "results": predictions}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
def model_info():
    """Model metadata endpoint."""
    return {
        "model_name": manager.model_name,
        "model_version": manager.model_version,
        "features": FEATURE_NAMES,
        "load_time_seconds": manager.load_time,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_TRACKING_URI
    }
