#!/usr/bin/env python3
"""Demo: Compare Edge TPU v5e vs v6 (projected) with INT8 and INT4 quantization."""
from __future__ import annotations

import json
import sys

import numpy as np

from edge_tpu_v6_bench import (
    EdgeTPUv5eBackend,
    EdgeTPUv6Backend,
    QuantizationRecipe,
)

# ---- Configuration -------------------------------------------------------- #

MODEL_PARAMS = {"num_params": 500_000, "input_size": 256}
NUM_RUNS = 5

RECIPES = {
    "INT8": QuantizationRecipe.int8(),
    "INT4": QuantizationRecipe.int4(),
}

BACKENDS = {
    "v5e": EdgeTPUv5eBackend(),
    "v6 (projected ⚠️)": EdgeTPUv6Backend(),
}

# --------------------------------------------------------------------------- #


def divider(char: str = "-", width: int = 60) -> None:
    print(char * width)


def main() -> None:
    rng = np.random.default_rng(seed=0)
    raw_input = rng.standard_normal(MODEL_PARAMS["input_size"]).astype(np.float32)

    print()
    divider("=")
    print("  Edge TPU v6 Preview Benchmark  —  Demo")
    print()
    print("  ⚠️  v6 numbers are PROJECTED/ESTIMATED.")
    print("      NOT measured on real hardware.")
    divider("=")

    results: dict[str, dict] = {}

    for recipe_name, recipe in RECIPES.items():
        print(f"\n── Quantization: {recipe_name} ──────────────────────────────")
        quantized_input = recipe.apply(raw_input).astype(np.float32)

        for backend_label, backend in BACKENDS.items():
            report = backend.benchmark(MODEL_PARAMS, quantized_input, num_runs=NUM_RUNS)
            key = f"{backend_label}/{recipe_name}"
            results[key] = report.to_dict()

            print(f"  {backend_label:30s}")
            print(f"    mean latency  : {report.mean_latency_ms:.4f} ms")
            print(f"    std  latency  : {report.std_latency_ms:.4f} ms")
            print(f"    mean QPS      : {report.mean_throughput_qps:.1f}")

    divider("=")
    print("\n  Full JSON summary:\n")
    print(json.dumps(results, indent=2))
    print()
    divider("=")
    print("\n  ⚠️  Reminder: EdgeTPUv6Backend is PROJECTED — not real hardware.\n")


if __name__ == "__main__":
    main()
