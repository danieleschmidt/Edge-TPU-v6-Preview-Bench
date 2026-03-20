"""Concrete backend implementations for Edge TPU v5e and v6.

⚠️  IMPORTANT DISCLAIMER
   EdgeTPUv6Backend performance numbers are **projected / estimated**.
   They are NOT measured on real Edge TPU v6 hardware (which is not yet
   publicly available).  All v6 figures are derived by applying a 2× speedup
   multiplier to measured v5e simulation results and should be treated as
   indicative only.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from .harness import BenchmarkHarness


# --------------------------------------------------------------------------- #
# Helper                                                                        #
# --------------------------------------------------------------------------- #

def _simulate_inference(
    model_params: Dict[str, Any],
    input_data: np.ndarray,
    speedup: float = 1.0,
) -> tuple[float, float]:
    """Core numpy simulation shared by both backends.

    Uses a random weight matrix of shape (input_size, num_params) and performs
    a matrix-multiply against *input_data* to generate realistic wall-clock
    timings.  The *speedup* factor scales the observed latency.

    Returns
    -------
    (latency_ms, throughput_qps)
    """
    num_params: int = int(model_params.get("num_params", 1_000_000))
    input_size: int = int(model_params.get("input_size", 256))

    rng = np.random.default_rng(seed=42)
    weights = rng.standard_normal((input_size, num_params)).astype(np.float32)

    # Ensure input is 1-D of the right size
    flat_input = np.asarray(input_data, dtype=np.float32).flatten()
    if flat_input.shape[0] != input_size:
        flat_input = flat_input[:input_size] if flat_input.shape[0] >= input_size else np.pad(
            flat_input, (0, input_size - flat_input.shape[0])
        )

    t0 = time.perf_counter()
    _ = flat_input @ weights  # the matmul "inference"
    elapsed_s = (time.perf_counter() - t0) / speedup  # apply speedup factor

    latency_ms = elapsed_s * 1_000.0
    throughput_qps = 1.0 / elapsed_s if elapsed_s > 0 else float("inf")
    return latency_ms, throughput_qps


# --------------------------------------------------------------------------- #
# v5e backend                                                                   #
# --------------------------------------------------------------------------- #

class EdgeTPUv5eBackend(BenchmarkHarness):
    """Simulated Edge TPU v5e backend using numpy matrix-multiply.

    This backend mimics v5e inference by measuring wall-clock time of a
    numpy matmul operation.  It does **not** run on real hardware.
    """

    def get_backend_name(self) -> str:
        return "EdgeTPU-v5e"

    def get_specs(self) -> Dict[str, Any]:
        """Return approximate Edge TPU v5e chip specifications."""
        return {
            "tops": 4.0,
            "power_w": 2.0,
            "tops_per_watt": 2.0,
        }

    def run_inference(
        self,
        model_params: Dict[str, Any],
        input_data: Any,
    ) -> Dict[str, Any]:
        """Simulate a single v5e inference pass.

        Parameters
        ----------
        model_params: Dict with 'num_params' (int) and 'input_size' (int).
        input_data: Array-like input data.

        Returns
        -------
        dict with keys: latency_ms, throughput_qps, backend.
        """
        latency_ms, throughput_qps = _simulate_inference(
            model_params, np.asarray(input_data, dtype=np.float32), speedup=1.0
        )
        return {
            "latency_ms": latency_ms,
            "throughput_qps": throughput_qps,
            "backend": "v5e",
        }


# --------------------------------------------------------------------------- #
# v6 backend                                                                    #
# --------------------------------------------------------------------------- #

class EdgeTPUv6Backend(BenchmarkHarness):
    """⚠️ PROJECTED / ESTIMATED Edge TPU v6 backend — NOT measured on real hardware.

    All performance figures reported by this backend are derived from applying
    a 2× speedup multiplier to the v5e simulation results.  This is a
    theoretical projection based on anticipated architectural improvements and
    should NOT be cited as measured benchmarks.

    The Edge TPU v6 is not yet publicly released; this class exists solely
    for exploratory / comparative modelling purposes.
    """

    #: Theoretical speedup factor over v5e (projected, not measured).
    SPEEDUP_VS_V5E: float = 2.0

    def get_backend_name(self) -> str:
        return "EdgeTPU-v6 (PROJECTED)"

    def get_specs(self) -> Dict[str, Any]:
        """Return projected Edge TPU v6 chip specifications.

        ⚠️ All values are PROJECTED / ESTIMATED — not measured on real hardware.
        """
        return {
            "tops": 8.0,
            "power_w": 2.5,
            "tops_per_watt": 3.2,
            "note": "PROJECTED - not measured on real hardware",
        }

    def run_inference(
        self,
        model_params: Dict[str, Any],
        input_data: Any,
    ) -> Dict[str, Any]:
        """Simulate a single projected v6 inference pass.

        ⚠️ Performance is estimated via a 2× speedup over the v5e numpy
        simulation.  These numbers are NOT from real Edge TPU v6 hardware.

        Parameters
        ----------
        model_params: Dict with 'num_params' (int) and 'input_size' (int).
        input_data: Array-like input data.

        Returns
        -------
        dict with keys: latency_ms, throughput_qps, backend.
        """
        latency_ms, throughput_qps = _simulate_inference(
            model_params,
            np.asarray(input_data, dtype=np.float32),
            speedup=self.SPEEDUP_VS_V5E,
        )
        return {
            "latency_ms": latency_ms,
            "throughput_qps": throughput_qps,
            "backend": "v6 (projected)",
        }
