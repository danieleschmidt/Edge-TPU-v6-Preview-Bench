"""Tests for BenchmarkReport."""
from __future__ import annotations

import json
import os
import tempfile
import pytest

from edge_tpu_v6_bench import BenchmarkReport


def _sample_report(**kwargs) -> BenchmarkReport:
    defaults = dict(
        backend_name="TestBackend",
        model_params={"num_params": 100},
        num_runs=5,
        mean_latency_ms=3.5,
        std_latency_ms=0.2,
        mean_throughput_qps=285.7,
    )
    defaults.update(kwargs)
    return BenchmarkReport(**defaults)


class TestBenchmarkReportFields:
    def test_backend_name(self):
        r = _sample_report()
        assert r.backend_name == "TestBackend"

    def test_num_runs(self):
        r = _sample_report(num_runs=10)
        assert r.num_runs == 10

    def test_timestamp_auto_set(self):
        r = _sample_report()
        assert r.timestamp is not None
        assert len(r.timestamp) > 0

    def test_custom_timestamp(self):
        r = _sample_report()
        r2 = BenchmarkReport(
            backend_name="X",
            model_params={},
            num_runs=1,
            mean_latency_ms=1.0,
            std_latency_ms=0.0,
            mean_throughput_qps=1000.0,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert r2.timestamp == "2024-01-01T00:00:00+00:00"


class TestBenchmarkReportSerialisation:
    def test_to_dict_type(self):
        r = _sample_report()
        d = r.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_keys(self):
        r = _sample_report()
        d = r.to_dict()
        for key in ["backend_name", "model_params", "num_runs",
                    "mean_latency_ms", "std_latency_ms",
                    "mean_throughput_qps", "timestamp"]:
            assert key in d

    def test_to_json_valid(self):
        r = _sample_report()
        s = r.to_json()
        parsed = json.loads(s)
        assert parsed["backend_name"] == "TestBackend"

    def test_to_json_roundtrip(self):
        r = _sample_report()
        d = json.loads(r.to_json())
        assert d["num_runs"] == 5
        assert d["mean_latency_ms"] == pytest.approx(3.5)

    def test_save_creates_file(self):
        r = _sample_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            r.save(path)
            assert os.path.exists(path)
            with open(path) as fh:
                data = json.load(fh)
            assert data["backend_name"] == "TestBackend"
        finally:
            os.unlink(path)
