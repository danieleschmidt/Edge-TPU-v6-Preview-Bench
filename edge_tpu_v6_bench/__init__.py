"""Edge TPU v6 Preview Benchmark package.

⚠️  DISCLAIMER: EdgeTPUv6Backend performance figures are PROJECTED / ESTIMATED
    and are NOT based on measurements from real Edge TPU v6 hardware.
"""

from .harness import BenchmarkHarness
from .backends import EdgeTPUv5eBackend, EdgeTPUv6Backend
from .quantization import QuantizationRecipe
from .report import BenchmarkReport

__all__ = [
    "BenchmarkHarness",
    "EdgeTPUv5eBackend",
    "EdgeTPUv6Backend",
    "QuantizationRecipe",
    "BenchmarkReport",
]

__version__ = "0.1.0"
