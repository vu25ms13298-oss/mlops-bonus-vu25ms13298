"""
Simulation Scenarios for Observability & Alerting Validation.

Defines scenarios:
  1. NormalTrafficScenario : Healthy baseline stream (no drift)
  2. SlightDriftScenario   : Mild shift in 2 features (sub-threshold)
  3. SevereDriftScenario   : High distribution drift across 4+ features
"""
from __future__ import annotations

import time
import logging
from typing import List, Dict, Any
import requests

from simulations.data_generator import WineDataGenerator

logger = logging.getLogger("Scenarios")


class BaseScenario:
    def __init__(self, api_url: str = "http://localhost:8000/predict"):
        self.api_url = api_url
        self.generator = WineDataGenerator()

    def run(self, count: int, delay_sec: float = 0.05) -> Dict[str, Any]:
        raise NotImplementedError


class NormalTrafficScenario(BaseScenario):
    """Sends baseline samples without distribution drift."""

    def run(self, count: int = 50, delay_sec: float = 0.02) -> Dict[str, Any]:
        logger.info("Executing NormalTrafficScenario (%d requests)...", count)
        successes = 0
        latencies = []

        for _ in range(count):
            sample = self.generator.generate_normal_sample()
            payload = {
                "features": list(sample.values()),
                "feature_names": list(sample.keys())
            }
            try:
                t0 = time.time()
                resp = requests.post(self.api_url, json=payload, timeout=2)
                latencies.append(time.time() - t0)
                if resp.status_code == 200:
                    successes += 1
            except Exception:
                pass
            time.sleep(delay_sec)

        return {
            "scenario": "NormalTraffic",
            "total_sent": count,
            "successes": successes,
            "avg_latency_ms": round(float(sum(latencies) / len(latencies) * 1000), 2) if latencies else 0.0
        }


class SevereDriftScenario(BaseScenario):
    """Sends strongly drifted samples to trigger Evidently drift alerts."""

    def run(self, count: int = 60, delay_sec: float = 0.02) -> Dict[str, Any]:
        logger.info("Executing SevereDriftScenario (%d requests)...", count)
        successes = 0
        latencies = []

        for _ in range(count):
            sample = self.generator.generate_drifted_sample(
                drift_factor=1.75,
                target_features=["alcohol", "volatile_acidity", "sulphates", "chlorides", "total_sulfur_dioxide"]
            )
            payload = {
                "features": list(sample.values()),
                "feature_names": list(sample.keys())
            }
            try:
                t0 = time.time()
                resp = requests.post(self.api_url, json=payload, timeout=2)
                latencies.append(time.time() - t0)
                if resp.status_code == 200:
                    successes += 1
            except Exception:
                pass
            time.sleep(delay_sec)

        return {
            "scenario": "SevereDrift",
            "total_sent": count,
            "successes": successes,
            "avg_latency_ms": round(float(sum(latencies) / len(latencies) * 1000), 2) if latencies else 0.0
        }
