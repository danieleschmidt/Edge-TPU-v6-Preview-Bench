"""Tests for EdgeTPUv5eBackend and EdgeTPUv6Backend."""
from __future__ import annotations

import pytest
import numpy as np

from edge_tpu_v6_bench import EdgeTPUv5eBackend, EdgeTPUv6Backend, BenchmarkReport


MODEL_PARAMS = {"num_params": 10_000, "input_size": 64}
INPUT = np.random.default_rng(0).standard_normal(64).astype(np.float32)


class TestEdgeTPUv5eBackend:
    def setup_method(self):
        self.backend = EdgeTPUv5eBackend()

    def test_get_backend_name(self):
        assert "v5e" in self.backend.get_backend_name().lower()

    def test_get_specs_keys(self):
        specs = self.backend.get_specs()
        assert "tops" in specs
        assert "power_w" in specs
        assert "tops_per_watt" in specs

    def test_get_specs_values(self):
        specs = self.backend.get_specs()
        assert specs["tops"] == 4.0
        assert specs["power_w"] == 2.0
        assert specs["tops_per_watt"] == 2.0

    def test_run_inference_returns_dict(self):
        result = self.backend.run_inference(MODEL_PARAMS, INPUT)
        assert isinstance(result, dict)

    def test_run_inference_keys(self):
        result = self.backend.run_inference(MODEL_PARAMS, INPUT)
        assert "latency_ms" in result
        assert "throughput_qps" in result
        assert "backend" in result

    def test_run_inference_positive_latency(self):
        result = self.backend.run_inference(MODEL_PARAMS, INPUT)
        assert result["latency_ms"] > 0

    def test_run_inference_positive_throughput(self):
        result = self.backend.run_inference(MODEL_PARAMS, INPUT)
        assert result["throughput_qps"] > 0

    def test_benchmark_returns_report(self):
        report = self.backend.benchmark(MODEL_PARAMS, INPUT, num_runs=3)
        assert isinstance(report, BenchmarkReport)

    def test_benchmark_mean_latency_positive(self):
        report = self.backend.benchmark(MODEL_PARAMS, INPUT, num_runs=3)
        assert report.mean_latency_ms > 0


class TestEdgeTPUv6Backend:
    def setup_method(self):
        self.backend = EdgeTPUv6Backend()

    def test_get_backend_name_mentions_projected(self):
        name = self.backend.get_backend_name().lower()
        assert "projected" in name or "v6" in name

    def test_get_specs_note_present(self):
        specs = self.backend.get_specs()
        assert "note" in specs
        assert "PROJECTED" in specs["note"]

    def test_get_specs_values(self):
        specs = self.backend.get_specs()
        assert specs["tops"] == 8.0
        assert specs["power_w"] == 2.5

    def test_run_inference_returns_dict(self):
        result = self.backend.run_inference(MODEL_PARAMS, INPUT)
        assert isinstance(result, dict)

    def test_run_inference_positive_latency(self):
        result = self.backend.run_inference(MODEL_PARAMS, INPUT)
        assert result["latency_ms"] > 0

    def test_v6_faster_than_v5e(self):
        """v6 should consistently report lower latency due to 2x speedup factor."""
        v5e = EdgeTPUv5eBackend()
        v6 = EdgeTPUv6Backend()
        # Use many runs to average out noise
        r5e = v5e.benchmark(MODEL_PARAMS, INPUT, num_runs=20)
        rv6 = v6.benchmark(MODEL_PARAMS, INPUT, num_runs=20)
        assert rv6.mean_latency_ms < r5e.mean_latency_ms

    def test_speedup_factor_constant(self):
        """v6 backend must advertise a 2x speedup vs v5e via its class constant."""
        assert EdgeTPUv6Backend.SPEEDUP_VS_V5E == 2.0

    def test_v6_tops_double_v5e(self):
        """v6 projected TOPS should be double the v5e TOPS."""
        v5e_tops = EdgeTPUv5eBackend().get_specs()["tops"]
        v6_tops = EdgeTPUv6Backend().get_specs()["tops"]
        assert v6_tops == pytest.approx(v5e_tops * 2.0)
