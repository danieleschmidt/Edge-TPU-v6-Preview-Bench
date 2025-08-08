"""
Benchmark suite implementations for Edge TPU v6
"""

from .micro import MicroBenchmarkSuite
from .standard import StandardBenchmark
# from .applications import ApplicationBenchmark  # TODO: Implement application benchmarks

__all__ = [
    'MicroBenchmarkSuite',
    'StandardBenchmark',
    # 'ApplicationBenchmark',
]