"""End-to-end integration tests."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from edge_tpu_v6_bench import (
    EdgeTPUv5eBackend,
    EdgeTPUv6Backend,
    QuantizationRecipe,
    BenchmarkReport,
)


MODEL_PARAMS = {"num_params": 5_000, "input_size": 32}


def _make_input(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(MODEL_PARAMS["input_size"]).astype(np.float32)


class TestEndToEnd:
    def test_v5e_full_pipeline(self):
        recipe = QuantizationRecipe.int8()
        backend = EdgeTPUv5eBackend()
        data = recipe.apply(_make_input()).astype(np.float32)
        report = backend.benchmark(MODEL_PARAMS, data, num_runs=5)
        assert isinstance(report, BenchmarkReport)
        assert report.mean_latency_ms > 0
        assert report.num_runs == 5

    def test_v6_full_pipeline(self):
        recipe = QuantizationRecipe.int4()
        backend = EdgeTPUv6Backend()
        data = recipe.apply(_make_input(seed=1)).astype(np.float32)
        report = backend.benchmark(MODEL_PARAMS, data, num_runs=5)
        assert isinstance(report, BenchmarkReport)
        assert report.mean_latency_ms > 0

    def test_report_save_and_reload(self):
        backend = EdgeTPUv5eBackend()
        report = backend.benchmark(MODEL_PARAMS, _make_input(), num_runs=3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.save(path)
            with open(path) as fh:
                data = json.load(fh)
            assert data["num_runs"] == 3
            assert data["backend_name"] == report.backend_name
        finally:
            os.unlink(path)

    def test_v6_specs_have_projected_note(self):
        specs = EdgeTPUv6Backend().get_specs()
        assert "note" in specs
        assert "PROJECTED" in specs["note"].upper()

    def test_quantization_reduces_precision(self):
        recipe = QuantizationRecipe.int4()
        data = np.linspace(-5, 5, 50).astype(np.float32)
        q = recipe.apply(data)
        # INT4 range is [-8, 7]
        assert q.min() >= -8 and q.max() <= 7
