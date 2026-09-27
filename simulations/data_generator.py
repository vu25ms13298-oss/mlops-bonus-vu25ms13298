"""
Data Generator for Real-Time Prediction & Drift Simulation.

Generates realistic feature vectors based on UCI wine quality distribution,
with programmatic drift injection (mean shift, variance expansion).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class WineDataGenerator:
    """Generates synthetic wine quality observations for testing and drift simulation."""

    def __init__(self, config_path: Path | str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.features_spec = self.config["features"]
        self.feature_names = list(self.features_spec.keys())

    def generate_normal_sample(self) -> Dict[str, float]:
        """Generate a single sample aligned with the baseline distribution."""
        sample: Dict[str, float] = {}
        for feature, params in self.features_spec.items():
            val = np.random.normal(params["mean"], params["std"])
            clipped = np.clip(val, params["min"], params["max"])
            sample[feature] = round(float(clipped), 4)
        return sample

    def generate_drifted_sample(
        self,
        drift_factor: float = 1.6,
        target_features: Optional[List[str]] = None,
        noise_level: float = 0.2
    ) -> Dict[str, float]:
        """Generate a sample with injected distribution shift."""
        sample = self.generate_normal_sample()
        drift_cols = target_features or ["alcohol", "volatile_acidity", "sulphates", "chlorides"]

        for col in drift_cols:
            if col in sample:
                params = self.features_spec[col]
                shifted_mean = params["mean"] * drift_factor
                shifted_std = params["std"] * (1.0 + noise_level)
                val = np.random.normal(shifted_mean, shifted_std)
                # Expand upper limit by 30% for severe drift
                sample[col] = round(float(np.clip(val, params["min"], params["max"] * 1.3)), 4)

        return sample

    def generate_batch(self, count: int, drift: bool = False, drift_factor: float = 1.5) -> List[Dict[str, float]]:
        """Generate a batch of normal or drifted samples."""
        if drift:
            return [self.generate_drifted_sample(drift_factor=drift_factor) for _ in range(count)]
        return [self.generate_normal_sample() for _ in range(count)]
