"""Abstract BenchmarkHarness base class."""
from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from .report import BenchmarkReport


class BenchmarkHarness(ABC):
    """Abstract base class that every backend must implement.

    Subclasses must override:
    - run_inference
    - get_backend_name
    - get_specs
    """

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def run_inference(self, model_params: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
        """Execute a single forward pass and return timing/throughput metrics.

        Parameters
        ----------
        model_params: Dict describing the model (e.g. 'num_params', 'input_size').
        input_data: Input tensor/array to feed through the model.

        Returns
        -------
        dict with at minimum:
            latency_ms  – wall-clock time for this inference (milliseconds)
            throughput_qps – queries per second for this inference
            backend     – string name identifying the backend
        """

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return a human-readable name for this backend."""

    @abstractmethod
    def get_specs(self) -> Dict[str, Any]:
        """Return hardware / chip specifications as a plain dict."""

    # ------------------------------------------------------------------ #
    # Concrete orchestration                                               #
    # ------------------------------------------------------------------ #

    def benchmark(
        self,
        model_params: Dict[str, Any],
        input_data: Any,
        num_runs: int = 10,
    ) -> BenchmarkReport:
        """Run *num_runs* inferences and aggregate the results.

        Parameters
        ----------
        model_params: Forwarded to :meth:`run_inference`.
        input_data: Forwarded to :meth:`run_inference`.
        num_runs: How many inference passes to execute (default 10).

        Returns
        -------
        :class:`~edge_tpu_v6_bench.report.BenchmarkReport` with aggregated
        mean / std statistics over all runs.
        """
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")

        latencies: list[float] = []
        throughputs: list[float] = []

        for _ in range(num_runs):
            result = self.run_inference(model_params, input_data)
            latencies.append(float(result["latency_ms"]))
            throughputs.append(float(result["throughput_qps"]))

        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        mean_throughput = statistics.mean(throughputs)

        return BenchmarkReport(
            backend_name=self.get_backend_name(),
            model_params=model_params,
            num_runs=num_runs,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            mean_throughput_qps=mean_throughput,
        )
