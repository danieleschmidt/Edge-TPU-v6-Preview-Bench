# Edge TPU v6 Preview Benchmark

> **⚠️ DISCLAIMER — PROJECTED PERFORMANCE ONLY**
>
> All Edge TPU **v6** numbers in this repository are **estimated / projected** values.
> They are **NOT** measured on real Edge TPU v6 hardware (which is not yet
> publicly available).  The `EdgeTPUv6Backend` derives its figures by applying a
> theoretical 2× speedup multiplier to numpy-simulated v5e results.
>
> **Do not cite these numbers as measured benchmarks.**

---

## Overview

`edge_tpu_v6_bench` is a Python package for comparative benchmarking of
simulated Edge TPU v5e and (projected) Edge TPU v6 inference workloads.
It supports configurable quantization recipes (INT8 / INT4) and produces
structured JSON reports.

---

## Package Structure

```
edge_tpu_v6_bench/
├── harness.py         # Abstract BenchmarkHarness base class
├── backends.py        # EdgeTPUv5eBackend + EdgeTPUv6Backend (projected)
├── quantization.py    # QuantizationRecipe (INT8, INT4)
├── report.py          # BenchmarkReport dataclass (JSON/save)
└── __init__.py        # Public API exports

demo.py                # Demo script comparing backends + recipes
tests/                 # Pytest test suite (10+ tests)
```

---

## Installation

Requires Python 3.8+ and NumPy.

```bash
pip install -e .
# or
pip install numpy
```

---

## Quick Start

```python
import numpy as np
from edge_tpu_v6_bench import EdgeTPUv5eBackend, EdgeTPUv6Backend, QuantizationRecipe

model_params = {"num_params": 500_000, "input_size": 256}
raw_input    = np.random.randn(256).astype(np.float32)

recipe = QuantizationRecipe.int8()
data   = recipe.apply(raw_input).astype(np.float32)

# Benchmark v5e
v5e_report = EdgeTPUv5eBackend().benchmark(model_params, data, num_runs=10)
print(v5e_report.to_json())

# Benchmark v6 (⚠️ projected)
v6_report  = EdgeTPUv6Backend().benchmark(model_params, data, num_runs=10)
print(v6_report.to_json())
```

Run the demo:

```bash
python demo.py
```

---

## API Reference

### `BenchmarkHarness` (abstract)

| Method | Description |
|---|---|
| `run_inference(model_params, input_data) -> dict` | Single inference pass |
| `get_backend_name() -> str` | Backend identifier |
| `get_specs() -> dict` | Hardware spec dict |
| `benchmark(model_params, input_data, num_runs=10) -> BenchmarkReport` | Aggregated N-run benchmark |

### `QuantizationRecipe`

| Class method | Precision | Range |
|---|---|---|
| `int8()` | INT8 | [-128, 127] |
| `int4()` | INT4 | [-8, 7] |

`recipe.apply(data: np.ndarray) -> np.ndarray` — quantize and clip.

### `BenchmarkReport`

Fields: `backend_name`, `model_params`, `num_runs`, `mean_latency_ms`,
`std_latency_ms`, `mean_throughput_qps`, `timestamp`.

Methods: `to_dict()`, `to_json()`, `save(path)`.

---

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

---

## Specs Summary

| Chip | TOPS | Power (W) | TOPS/W | Note |
|---|---|---|---|---|
| Edge TPU v5e | 4.0 | 2.0 | 2.0 | Simulated (numpy) |
| Edge TPU v6 | 8.0 | 2.5 | 3.2 | ⚠️ PROJECTED ONLY |

---

## License

MIT
