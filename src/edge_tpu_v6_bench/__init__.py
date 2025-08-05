"""
Edge TPU v6 Preview Benchmark Suite
Future-ready benchmarking for next-generation Edge AI acceleration
"""

__version__ = '0.1.0'
__author__ = 'Daniel Schmidt'
__email__ = 'daniel@terragonlabs.com'
__license__ = 'Apache 2.0'

# Core public API
from .core.benchmark import EdgeTPUBenchmark
from .quantization.auto_quantizer import AutoQuantizer
from .core.device_manager import DeviceManager
from .core.metrics import BenchmarkMetrics

# Convenience imports for common workflows
from .benchmarks.micro import MicroBenchmarkSuite
from .benchmarks.standard import StandardBenchmark
from .benchmarks.applications import ApplicationBenchmark

# Analysis and visualization
from .analysis.profiler import LatencyProfiler
from .analysis.power import PowerAnalyzer
from .analysis.thermal import ThermalAnalyzer
from .analysis.visualizer import PerformanceVisualizer

# Migration and compatibility
from .compatibility.migration import MigrationAssistant
from .compatibility.features import FeatureDetector

__all__ = [
    # Core API
    'EdgeTPUBenchmark',
    'AutoQuantizer', 
    'DeviceManager',
    'BenchmarkMetrics',
    
    # Benchmark suites
    'MicroBenchmarkSuite',
    'StandardBenchmark', 
    'ApplicationBenchmark',
    
    # Analysis tools
    'LatencyProfiler',
    'PowerAnalyzer', 
    'ThermalAnalyzer',
    'PerformanceVisualizer',
    
    # Compatibility
    'MigrationAssistant',
    'FeatureDetector',
]

# Version info tuple
VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version():
    """Get the current version string."""
    return __version__

def get_device_compatibility():
    """Get supported device compatibility info."""
    return {
        'edge_tpu_v6': 'Primary target (preview)',
        'edge_tpu_v5e': 'Full compatibility with fallback',
        'coral_dev_board': 'Supported via compatibility layer',
        'usb_accelerator': 'Basic functionality',
    }

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())