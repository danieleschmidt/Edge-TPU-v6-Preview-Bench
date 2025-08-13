"""
Core benchmarking engine components for Edge TPU v6
"""

# Conditional imports based on dependencies availability
try:
    from .benchmark import EdgeTPUBenchmark
    from .device_manager import DeviceManager
    from .metrics import BenchmarkMetrics
    from .power import PowerMonitor
    ADVANCED_FEATURES = True
except ImportError:
    # Basic mode - only simple benchmark available
    try:
        from .simple_benchmark import SimpleEdgeTPUBenchmark, SimpleAutoQuantizer
        EdgeTPUBenchmark = SimpleEdgeTPUBenchmark
        AutoQuantizer = SimpleAutoQuantizer
        ADVANCED_FEATURES = False
        
        # Mock missing classes
        class DeviceManager:
            pass
        class BenchmarkMetrics:
            pass
        class PowerMonitor:
            pass
    except ImportError:
        ADVANCED_FEATURES = False

__all__ = [
    'EdgeTPUBenchmark',
    'DeviceManager', 
    'BenchmarkMetrics',
    'PowerMonitor',
]