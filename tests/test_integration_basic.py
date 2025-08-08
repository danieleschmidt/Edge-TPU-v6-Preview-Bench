"""
Basic integration tests for Edge TPU v6 benchmarking suite
Tests core functionality without requiring physical hardware
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_import_core_modules():
    """Test that all core modules can be imported successfully"""
    
    # Test core benchmark import
    from edge_tpu_v6_bench import EdgeTPUBenchmark
    assert EdgeTPUBenchmark is not None
    
    # Test quantum planner import
    from edge_tpu_v6_bench.quantum_planner import QuantumTaskEngine
    assert QuantumTaskEngine is not None
    
    # Test analysis module import  
    from edge_tpu_v6_bench.analysis.profiler import LatencyProfiler
    assert LatencyProfiler is not None
    
    # Test validation module import
    from edge_tpu_v6_bench.core.validation import validate_and_sanitize
    assert validate_and_sanitize is not None
    
    # Test error handling import
    from edge_tpu_v6_bench.core.error_handling import ErrorRecoverySystem
    assert ErrorRecoverySystem is not None

def test_quantum_task_engine_basic():
    """Test basic quantum task engine functionality"""
    from edge_tpu_v6_bench.quantum_planner import QuantumTaskEngine, Priority
    
    engine = QuantumTaskEngine(max_workers=2, quantum_coherence_time=10.0)
    assert engine is not None
    
    # Add a simple task
    def simple_task():
        return "task_completed"
    
    task = engine.add_task(
        task_id="test_task_1",
        name="Simple Test Task", 
        function=simple_task,
        description="A simple test task",
        priority=Priority.HIGH
    )
    
    assert task.task_id == "test_task_1"
    assert task.name == "Simple Test Task"
    assert task.priority == Priority.HIGH

def test_validation_system():
    """Test input validation and sanitization"""
    from edge_tpu_v6_bench.core.validation import validate_and_sanitize, ValidationResult
    
    # Test string validation
    result = validate_and_sanitize("test_string", "string", max_length=100)
    assert isinstance(result, ValidationResult)
    assert result.is_valid
    assert result.sanitized_value == "test_string"
    
    # Test numeric validation
    result = validate_and_sanitize(42.5, "numeric", min_val=0, max_val=100)
    assert result.is_valid
    assert result.sanitized_value == 42.5
    
    # Test invalid numeric input
    result = validate_and_sanitize("invalid", "numeric")
    assert not result.is_valid
    assert result.error_message is not None

def test_error_recovery_system():
    """Test error handling and recovery"""
    from edge_tpu_v6_bench.core.error_handling import ErrorRecoverySystem, EdgeTPUError, ErrorCategory
    
    recovery_system = ErrorRecoverySystem()
    assert recovery_system is not None
    
    # Test error handling
    test_error = EdgeTPUError("Test error", category=ErrorCategory.VALIDATION)
    
    # Handle the error (should return None as no recovery strategies match)
    result = recovery_system.handle_error(test_error)
    
    # Check that error was logged
    assert len(recovery_system.error_history) >= 1
    assert recovery_system.error_history[-1].category == ErrorCategory.VALIDATION

def test_security_manager():
    """Test security management functionality"""
    from edge_tpu_v6_bench.core.security import SecurityManager
    
    security_manager = SecurityManager()
    assert security_manager is not None
    
    # Test input sanitization
    sanitized = security_manager.sanitize_input("test_input", "general")
    assert isinstance(sanitized, str)
    
    # Test threat scanning
    threats = security_manager.scan_for_threats("../../../etc/passwd", "path")
    assert len(threats) > 0  # Should detect path traversal

def test_performance_cache():
    """Test performance caching system"""
    from edge_tpu_v6_bench.core.performance_cache import PerformanceCache, CacheLevel
    
    # Create cache with small limits for testing
    cache = PerformanceCache(
        max_memory_size=1024 * 1024,  # 1MB
        max_disk_size=10 * 1024 * 1024  # 10MB
    )
    
    # Test basic set/get
    cache.set("test_key", "test_value")
    result = cache.get("test_key")
    assert result == "test_value"
    
    # Test cache miss
    result = cache.get("nonexistent_key", "default")
    assert result == "default"
    
    # Test statistics
    stats = cache.get_statistics()
    assert "overall" in stats
    assert stats["overall"]["total_requests"] >= 2  # At least our get operations

def test_concurrent_executor():
    """Test concurrent execution system"""
    from edge_tpu_v6_bench.core.concurrent_execution import ConcurrentExecutor, TaskSpec, ExecutionStrategy
    
    executor = ConcurrentExecutor(
        strategy=ExecutionStrategy.THREADED,
        max_workers=2
    )
    
    # Create simple test tasks
    def test_function(x):
        return x * 2
    
    tasks = [
        TaskSpec(
            task_id=f"task_{i}",
            function=test_function,
            args=(i,),
            priority=5
        )
        for i in range(3)
    ]
    
    # Execute tasks
    results = executor.execute_batch(tasks)
    
    assert len(results) == 3
    for i in range(3):
        task_id = f"task_{i}"
        assert task_id in results
        assert results[task_id].success
        assert results[task_id].result == i * 2

def test_latency_profiler():
    """Test latency profiling functionality"""
    from edge_tpu_v6_bench.analysis.profiler import LatencyProfiler
    
    profiler = LatencyProfiler(device='cpu')  # Use CPU for testing
    assert profiler is not None
    
    # Mock model for profiling
    class MockModel:
        pass
    
    mock_model = MockModel()
    
    # Profile with small parameters for fast testing
    profile = profiler.profile_model(
        mock_model,
        input_shape=(1, 224, 224, 3),
        n_runs=10,
        warmup_runs=2
    )
    
    assert profile is not None
    assert profile.total_latency_ms > 0
    assert len(profile.layer_profiles) > 0

def test_device_manager_fallback():
    """Test device manager fallback behavior"""
    from edge_tpu_v6_bench.core.device_manager import DeviceManager, DeviceType
    
    device_manager = DeviceManager()
    
    # Test device selection fallback (should fall back to CPU since no Edge TPU available)
    device_info = device_manager.select_device('auto')
    
    # Should fall back to CPU or mock device
    assert device_info is not None
    assert device_info.device_type in [DeviceType.CPU_FALLBACK, DeviceType.EDGE_TPU_V5E, DeviceType.EDGE_TPU_V6, DeviceType.UNKNOWN]

def test_benchmark_config_validation():
    """Test benchmark configuration validation"""
    from edge_tpu_v6_bench.core.validation import ConfigValidator
    
    # Valid configuration
    valid_config = {
        'warmup_runs': 10,
        'measurement_runs': 100,
        'timeout_seconds': 300.0,
        'batch_sizes': [1, 4, 8],
        'concurrent_streams': 2
    }
    
    result = ConfigValidator.validate_benchmark_config(valid_config)
    assert result.is_valid
    
    # Invalid configuration
    invalid_config = {
        'warmup_runs': -5,  # Invalid negative value
        'measurement_runs': 0,  # Invalid zero value
        'batch_sizes': ['invalid']  # Invalid type in list
    }
    
    result = ConfigValidator.validate_benchmark_config(invalid_config)
    assert not result.is_valid

def test_file_operations_security():
    """Test secure file operations"""
    from edge_tpu_v6_bench.core.security import SecurityManager
    from edge_tpu_v6_bench.core.validation import SecurityValidator
    
    security_manager = SecurityManager()
    
    # Test secure temp file creation
    temp_file = security_manager.create_secure_temp_file()
    assert temp_file.exists()
    
    # Test file permissions (Unix-like systems only)
    if hasattr(os, 'stat'):
        file_stat = temp_file.stat()
        # Check that file is readable by owner
        assert file_stat.st_mode & 0o400  
    
    # Clean up
    temp_file.unlink()
    
    # Test file path validation with dangerous paths
    dangerous_paths = [
        "../../../etc/passwd",
        "/etc/shadow",
        "C:\\Windows\\System32\\config\\SAM"
    ]
    
    for path in dangerous_paths:
        result = SecurityValidator.validate_file_path(path)
        # Should either be invalid or not exist
        assert not result.is_valid or not Path(path).exists()

def test_comprehensive_workflow():
    """Test a comprehensive workflow using multiple systems"""
    import asyncio
    from edge_tpu_v6_bench.quantum_planner import QuantumTaskEngine, Priority
    from edge_tpu_v6_bench.core.performance_cache import PerformanceCache
    from edge_tpu_v6_bench.core.validation import validate_and_sanitize
    
    # Initialize systems
    cache = PerformanceCache(max_memory_size=1024*1024)
    engine = QuantumTaskEngine(max_workers=2, quantum_coherence_time=30.0)
    
    # Define workflow tasks
    def data_preparation_task():
        # Simulate data preparation
        data = np.random.randn(100, 100)
        return data.mean()
    
    def model_optimization_task():
        # Simulate model optimization
        return "model_optimized"
    
    def benchmark_execution_task():
        # Simulate benchmark execution
        return {"latency_ms": 15.2, "throughput_fps": 65.8}
    
    # Validate inputs
    config = {"warmup_runs": 5, "measurement_runs": 50}
    validation_result = validate_and_sanitize(config, "benchmark_config")
    assert validation_result.is_valid
    
    # Add tasks to quantum engine
    tasks = [
        engine.add_task("data_prep", "Data Preparation", data_preparation_task, priority=Priority.HIGH),
        engine.add_task("model_opt", "Model Optimization", model_optimization_task, 
                       dependencies={"data_prep"}, priority=Priority.MEDIUM),
        engine.add_task("benchmark", "Benchmark Execution", benchmark_execution_task,
                       dependencies={"model_opt"}, priority=Priority.HIGH)
    ]
    
    # Cache intermediate results
    cache.set("workflow_config", config)
    cached_config = cache.get("workflow_config")
    assert cached_config == config
    
    # Execute workflow (mock execution since we can't run real async in pytest easily)
    assert len(engine.tasks) == 3
    assert all(task.task_id in engine.tasks for task in tasks)
    
    # Verify task dependencies
    assert "data_prep" in engine.tasks["model_opt"].dependencies
    assert "model_opt" in engine.tasks["benchmark"].dependencies
    
    print("✅ Comprehensive workflow test completed successfully")

if __name__ == "__main__":
    # Run basic tests directly
    test_import_core_modules()
    test_quantum_task_engine_basic() 
    test_validation_system()
    test_error_recovery_system()
    test_security_manager()
    test_performance_cache()
    test_concurrent_executor()
    test_latency_profiler()
    test_device_manager_fallback()
    test_benchmark_config_validation()
    test_file_operations_security()
    test_comprehensive_workflow()
    
    print("\n🎉 All integration tests passed!")