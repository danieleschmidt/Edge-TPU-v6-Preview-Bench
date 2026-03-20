"""BenchmarkReport dataclass for storing and serializing benchmark results."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class BenchmarkReport:
    """Stores the aggregated results from a benchmark run.

    Attributes
    ----------
    backend_name: Name of the backend that was benchmarked.
    model_params: Dict of model parameters used (e.g. num_params, input_size).
    num_runs: Number of inference runs that were executed.
    mean_latency_ms: Mean latency across all runs (milliseconds).
    std_latency_ms: Standard deviation of latency (milliseconds).
    mean_throughput_qps: Mean queries-per-second across all runs.
    timestamp: ISO-8601 UTC timestamp of when the report was created.
    """

    backend_name: str
    model_params: Dict[str, Any]
    num_runs: int
    mean_latency_ms: float
    std_latency_ms: float
    mean_throughput_qps: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a plain Python dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Persist the report to *path* as JSON.

        Parameters
        ----------
        path: File system path where the JSON report will be written.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
