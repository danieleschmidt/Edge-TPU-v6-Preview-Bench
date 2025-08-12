"""
Edge TPU v6 Preview Benchmark Suite
Future-ready benchmarking for next-generation Edge AI acceleration
"""

__version__ = '0.1.0'
__author__ = 'Daniel Schmidt'
__email__ = 'daniel@terragonlabs.com'
__license__ = 'Apache 2.0'

# Try to import full components, fallback to simple versions for Generation 1
try:
    # Core public API
    from .core.benchmark import EdgeTPUBenchmark
    from .quantization.auto_quantizer import AutoQuantizer
    from .core.device_manager import DeviceManager
    from .core.metrics import BenchmarkMetrics
    
    # Convenience imports for common workflows
    from .benchmarks.micro import MicroBenchmarkSuite
    from .benchmarks.standard import StandardBenchmark
    
    # Quantum Planner
    from .quantum_planner import QuantumTaskEngine, QuantumScheduler, QuantumOptimizer, TaskGraph, QuantumTask, QuantumHeuristics, PerformanceOptimizer
    
    FULL_FEATURES = True
    
except ImportError as e:
    # Fallback to simple implementations for Generation 1
    from .core.simple_benchmark import SimpleEdgeTPUBenchmark, SimpleAutoQuantizer
    
    # Aliases for compatibility
    EdgeTPUBenchmark = SimpleEdgeTPUBenchmark
    AutoQuantizer = SimpleAutoQuantizer
    
    # Mock other classes to avoid import errors
    class DeviceManager:
        def __init__(self):
            pass
    
    class BenchmarkMetrics:
        def __init__(self):
            pass
    
    class MicroBenchmarkSuite:
        def __init__(self):
            pass
    
    class StandardBenchmark:
        def __init__(self):
            pass
    
    # Mock quantum planner components
    class QuantumTaskEngine:
        def __init__(self):
            pass
    
    class QuantumScheduler:
        def __init__(self):
            pass
    
    class QuantumOptimizer:
        def __init__(self):
            pass
    
    class TaskGraph:
        def __init__(self):
            pass
    
    class QuantumTask:
        def __init__(self):
            pass
    
    class QuantumHeuristics:
        def __init__(self):
            pass
    
    class PerformanceOptimizer:
        def __init__(self):
            pass
    
    FULL_FEATURES = False

__all__ = [
    # Core API
    'EdgeTPUBenchmark',
    'AutoQuantizer', 
    'DeviceManager',
    'BenchmarkMetrics',
    
    # Benchmark suites
    'MicroBenchmarkSuite',
    'StandardBenchmark', 
    # 'ApplicationBenchmark',
    
    # Quantum Planner
    'QuantumTaskEngine',
    'QuantumScheduler',
    'QuantumOptimizer', 
    'TaskGraph',
    'QuantumTask',
    'QuantumHeuristics',
    'PerformanceOptimizer',
    
    # Analysis tools (commented out until modules are created)
    # 'LatencyProfiler',
    # 'PowerAnalyzer', 
    # 'ThermalAnalyzer',
    # 'PerformanceVisualizer',
    
    # Compatibility (commented out until modules are created)
    # 'MigrationAssistant',
    # 'FeatureDetector',
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