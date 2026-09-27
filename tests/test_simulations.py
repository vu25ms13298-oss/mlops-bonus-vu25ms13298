"""
Unit Tests for Simulation Data Generator Module.
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

from simulations.data_generator import WineDataGenerator

CONFIG_PATH = Path(__file__).resolve().parents[1] / "simulations" / "config.yaml"


@pytest.fixture
def generator():
    return WineDataGenerator(config_path=CONFIG_PATH)


def test_normal_sample_has_all_features(generator):
    sample = generator.generate_normal_sample()
    assert len(sample) == 11
    for name in generator.feature_names:
        assert name in sample


def test_normal_sample_within_bounds(generator):
    np.random.seed(42)
    for _ in range(50):
        sample = generator.generate_normal_sample()
        for name, val in sample.items():
            spec = generator.features_spec[name]
            assert spec["min"] <= val <= spec["max"], (
                f"{name}={val} out of [{spec['min']}, {spec['max']}]"
            )


def test_drifted_sample_has_all_features(generator):
    sample = generator.generate_drifted_sample(drift_factor=1.5)
    assert len(sample) == 11
    for name in generator.feature_names:
        assert name in sample


def test_drifted_sample_shifts_target_features(generator):
    np.random.seed(42)
    targets = ["alcohol", "volatile_acidity"]
    normal_means = {t: [] for t in targets}
    drifted_means = {t: [] for t in targets}

    for _ in range(200):
        n = generator.generate_normal_sample()
        d = generator.generate_drifted_sample(drift_factor=1.6, target_features=targets)
        for t in targets:
            normal_means[t].append(n[t])
            drifted_means[t].append(d[t])

    for t in targets:
        normal_avg = np.mean(normal_means[t])
        drifted_avg = np.mean(drifted_means[t])
        assert drifted_avg != pytest.approx(normal_avg, rel=0.1), (
            f"Drifted mean for {t} should differ from normal"
        )


def test_batch_generation(generator):
    normal_batch = generator.generate_batch(count=20, drift=False)
    assert len(normal_batch) == 20

    drifted_batch = generator.generate_batch(count=15, drift=True, drift_factor=1.5)
    assert len(drifted_batch) == 15
