"""Tests for BenchmarkHarness base class."""
from __future__ import annotations

import pytest
import numpy as np

from edge_tpu_v6_bench import BenchmarkHarness, BenchmarkReport


# --- Minimal concrete implementation for testing ----------------------------- #

class _DummyBackend(BenchmarkHarness):
    """Trivial backend that returns fixed timing values."""

    def __init__(self, latency_ms: float = 5.0, qps: float = 200.0):
        self._latency_ms = latency_ms
        self._qps = qps

    def run_inference(self, model_params, input_data):
        return {
            "latency_ms": self._latency_ms,
            "throughput_qps": self._qps,
            "backend": "dummy",
        }

    def get_backend_name(self):
        return "DummyBackend"

    def get_specs(self):
        return {"tops": 1.0}


# ----------------------------------------------------------------------------- #

class TestBenchmarkHarnessInterface:
    def test_get_backend_name(self):
        assert _DummyBackend().get_backend_name() == "DummyBackend"

    def test_get_specs_returns_dict(self):
        specs = _DummyBackend().get_specs()
        assert isinstance(specs, dict)

    def test_run_inference_returns_required_keys(self):
        result = _DummyBackend().run_inference({}, np.zeros(4))
        assert "latency_ms" in result
        assert "throughput_qps" in result
        assert "backend" in result


class TestBenchmarkMethod:
    def setup_method(self):
        self.backend = _DummyBackend(latency_ms=10.0, qps=100.0)
        self.model_params = {"num_params": 1000, "input_size": 16}
        self.input_data = np.ones(16, dtype=np.float32)

    def test_returns_benchmark_report(self):
        report = self.backend.benchmark(self.model_params, self.input_data)
        assert isinstance(report, BenchmarkReport)

    def test_num_runs_recorded(self):
        report = self.backend.benchmark(self.model_params, self.input_data, num_runs=7)
        assert report.num_runs == 7

    def test_mean_latency_equals_fixed_value(self):
        report = self.backend.benchmark(self.model_params, self.input_data, num_runs=5)
        assert pytest.approx(report.mean_latency_ms, rel=1e-6) == 10.0

    def test_std_latency_zero_for_constant_backend(self):
        report = self.backend.benchmark(self.model_params, self.input_data, num_runs=5)
        assert pytest.approx(report.std_latency_ms, abs=1e-9) == 0.0

    def test_mean_throughput_equals_fixed_value(self):
        report = self.backend.benchmark(self.model_params, self.input_data, num_runs=5)
        assert pytest.approx(report.mean_throughput_qps, rel=1e-6) == 100.0

    def test_backend_name_in_report(self):
        report = self.backend.benchmark(self.model_params, self.input_data)
        assert report.backend_name == "DummyBackend"

    def test_invalid_num_runs_raises(self):
        with pytest.raises(ValueError):
            self.backend.benchmark(self.model_params, self.input_data, num_runs=0)

    def test_single_run_std_zero(self):
        report = self.backend.benchmark(self.model_params, self.input_data, num_runs=1)
        assert report.std_latency_ms == 0.0
