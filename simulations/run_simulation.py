"""
End-to-End Simulation Runner.

Simulates production traffic:
  Phase 1: Normal traffic -> verifies baseline stability
  Phase 2: Drift traffic  -> triggers drift detection and Prometheus alerts
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulations.scenarios import NormalTrafficScenario, SevereDriftScenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SimulationRunner")

API_URL = os.getenv("API_URL", "http://localhost:8000")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")


def check_services() -> bool:
    try:
        r1 = requests.get(f"{API_URL}/health", timeout=2)
        r2 = requests.get(f"{EVIDENTLY_URL}/health", timeout=2)
        return r1.status_code == 200 and r2.status_code == 200
    except Exception:
        return False


def main():
    logger.info("Checking service health (API: %s, Evidently: %s)...", API_URL, EVIDENTLY_URL)
    if not check_services():
        logger.warning("Services are not reachable at localhost:8000 / localhost:8001.")
        logger.info("Make sure docker compose is running: docker compose up -d")
        return

    logger.info("=== PHASE 1: Simulating Normal Traffic (50 requests) ===")
    normal = NormalTrafficScenario(api_url=f"{API_URL}/predict")
    res1 = normal.run(count=50, delay_sec=0.01)
    logger.info("Phase 1 finished: %s", res1)

    logger.info("Triggering Evidently Analysis (Normal)...")
    try:
        drift_res1 = requests.post(f"{EVIDENTLY_URL}/analyze", json={"window_size": 50}, timeout=10).json()
        logger.info("Evidently Result 1: Drift Detected = %s, Score = %s",
                    drift_res1.get("dataset_drift_detected"), drift_res1.get("drift_share"))
    except Exception as e:
        logger.warning("Analysis 1 warning: %s", e)

    time.sleep(1)

    logger.info("=== PHASE 2: Simulating Severe Drift Traffic (60 requests) ===")
    drifted = SevereDriftScenario(api_url=f"{API_URL}/predict")
    res2 = drifted.run(count=60, delay_sec=0.01)
    logger.info("Phase 2 finished: %s", res2)

    logger.info("Triggering Evidently Analysis (Drifted)...")
    try:
        drift_res2 = requests.post(f"{EVIDENTLY_URL}/analyze", json={"window_size": 60}, timeout=10).json()
        logger.info("Evidently Result 2: Drift Detected = %s, Score = %s, Drifted Features = %s",
                    drift_res2.get("dataset_drift_detected"), drift_res2.get("drift_share"),
                    drift_res2.get("number_of_drifted_columns"))
        logger.info("Report generated: %s", drift_res2.get("html_report"))
    except Exception as e:
        logger.warning("Analysis 2 warning: %s", e)


if __name__ == "__main__":
    main()
