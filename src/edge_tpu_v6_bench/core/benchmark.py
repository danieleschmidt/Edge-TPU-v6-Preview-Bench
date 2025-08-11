"""
Core benchmarking engine for Edge TPU v6
Main orchestration class that coordinates device management, model execution, and metrics collection
"""

import time
import logging
import statistics
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
import queue

import numpy as np
import tensorflow as tf

from .device_manager import DeviceManager, DeviceInfo, DeviceType
from .metrics import BenchmarkMetrics, MetricType
from .power import PowerMonitor
from .validation import SecurityValidator, ConfigValidator, validate_and_sanitize
from .security import SecurityManager, global_security_manager
from .error_handling import ErrorRecoverySystem, handle_errors, global_recovery_system, EdgeTPUError, ValidationError
from .performance_cache import PerformanceCache, global_performance_cache
from .concurrent_execution import ConcurrentExecutor, TaskSpec, ExecutionStrategy, global_concurrent_executor
from .auto_scaler import AutoScaler, ScalingTrigger, global_auto_scaler

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    warmup_runs: int = 10
    measurement_runs: int = 100
    timeout_seconds: float = 300.0
    measure_power: bool = False
    measure_thermal: bool = False
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    input_shapes: Optional[Dict[str, tuple]] = None
    target_latency_ms: Optional[float] = None
    target_throughput_fps: Optional[float] = None
    concurrent_streams: int = 1
    output_precision: int = 4  # Decimal places for results

@dataclass 
class BenchmarkResult:
    """Results from a benchmark execution"""
    device_info: DeviceInfo
    model_info: Dict[str, Any]
    config: BenchmarkConfig
    metrics: Dict[str, Any]
    raw_measurements: Dict[str, List[float]]
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class EdgeTPUBenchmark:
    """
    Main benchmark orchestration class for Edge TPU v6 devices
    
    Provides comprehensive benchmarking capabilities including:
    - Automatic device detection and selection
    - Model loading and compilation
    - Performance measurement (latency, throughput, accuracy)
    - Power and thermal monitoring
    - Multi-batch and concurrent execution
    - Comprehensive security validation
    - Robust error handling and recovery
    """
    
    def __init__(self, 
                 device: Union[str, DeviceType, int] = 'auto',
                 power_monitoring: bool = False,
                 thermal_monitoring: bool = False):
        """
        Initialize Edge TPU benchmark
        
        Args:
            device: Device specification for auto-selection
            power_monitoring: Enable power measurement
            thermal_monitoring: Enable thermal monitoring
        """
        # Validate device specification
        device_validation = validate_and_sanitize(device, 'device_spec')
        if not device_validation.is_valid:
            raise ValidationError(f"Invalid device specification: {device_validation.error_message}")
        device = device_validation.sanitized_value or device
        
        self.device_manager = DeviceManager()
        self.metrics = BenchmarkMetrics()
        self.power_monitor = PowerMonitor() if power_monitoring else None
        self.security_manager = global_security_manager
        self.recovery_system = global_recovery_system
        self.performance_cache = global_performance_cache
        self.concurrent_executor = global_concurrent_executor
        self.auto_scaler = global_auto_scaler
        
        # Device selection with error recovery
        try:
            self.device_info = self.device_manager.select_device(device)
        except Exception as e:
            recovery_result = self.recovery_system.handle_error(e, {'device': device})
            if recovery_result and isinstance(recovery_result, dict) and 'device' in recovery_result:
                self.device_info = self.device_manager.select_device(recovery_result['device'])
            else:
                raise EdgeTPUError(f"Failed to select device after recovery: {e}")
        
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self.model_info: Dict[str, Any] = {}
        
        # Monitoring flags
        self.power_monitoring = power_monitoring
        self.thermal_monitoring = thermal_monitoring
        
        # Start auto-scaling for dynamic performance optimization
        self.auto_scaler.start()
        
        logger.info(f"Initialized EdgeTPUBenchmark with {self.device_info.device_type.value}, "
                   f"security, caching, concurrent execution, and auto-scaling enabled")
        
    @handle_errors(global_recovery_system, reraise=True)
    def load_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and prepare model for benchmarking with security validation
        
        Args:
            model_path: Path to TensorFlow Lite model file
            
        Returns:
            Model information dictionary
            
        Raises:
            ValidationError: If file validation fails
            EdgeTPUError: If model loading fails
        """
        # Security validation of file path
        path_validation = validate_and_sanitize(model_path, 'file_path')
        if not path_validation.is_valid:
            raise ValidationError(f"Model path validation failed: {path_validation.error_message}")
        
        model_path = path_validation.sanitized_value or Path(model_path)
        
        # Additional security checks
        if not self.security_manager.validate_access(str(model_path)):
            raise ValidationError("Access to model file denied by security policy")
        
        # Verify file integrity
        if not self.security_manager.verify_file_integrity(model_path):
            raise ValidationError("Model file integrity verification failed")
        
        # Scan for threats
        threats = self.security_manager.scan_for_threats(str(model_path), "path")
        if threats:
            raise ValidationError(f"Security threats detected in model path: {threats}")
        
        logger.info(f"Loading model: {model_path} (security validation passed)")
        
        try:
            # Create interpreter for the selected device
            self.interpreter = self.device_manager.create_interpreter(
                str(model_path), self.device_info
            )
            
            # Extract model information
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Calculate model size
            model_size_bytes = model_path.stat().st_size
            
            self.model_info = {
                'path': str(model_path),
                'size_bytes': model_size_bytes,
                'size_mb': model_size_bytes / (1024 * 1024),
                'input_details': input_details,
                'output_details': output_details,
                'num_inputs': len(input_details),
                'num_outputs': len(output_details),
                'input_shapes': [detail['shape'].tolist() for detail in input_details],
                'output_shapes': [detail['shape'].tolist() for detail in output_details],
                'input_dtypes': [detail['dtype'].__name__ for detail in input_details],
                'output_dtypes': [detail['dtype'].__name__ for detail in output_details],
            }
            
            logger.info(f"Model loaded successfully: "
                       f"{self.model_info['size_mb']:.1f} MB, "
                       f"{self.model_info['num_inputs']} inputs, "
                       f"{self.model_info['num_outputs']} outputs")
            
            return self.model_info
            
        except Exception as e:
            raise EdgeTPUError(f"Failed to load model {model_path}: {e}")
    
    @handle_errors(global_recovery_system, reraise=False, default_return=None)
    def benchmark(self, 
                  model_path: Optional[Union[str, Path]] = None,
                  test_data: Optional[np.ndarray] = None,
                  config: Optional[BenchmarkConfig] = None,
                  metrics: Optional[List[str]] = None) -> BenchmarkResult:
        """
        Run comprehensive benchmark on the loaded model with validation and security
        
        Args:
            model_path: Path to model (if not already loaded)
            test_data: Input data for benchmarking 
            config: Benchmark configuration
            metrics: List of metrics to measure ['latency', 'throughput', 'accuracy', 'power']
            
        Returns:
            Comprehensive benchmark results or None if critical error
        """
        # Validate configuration
        if config is None:
            config = BenchmarkConfig()
        else:
            config_dict = {
                'warmup_runs': config.warmup_runs,
                'measurement_runs': config.measurement_runs,
                'timeout_seconds': config.timeout_seconds,
                'batch_sizes': config.batch_sizes,
                'concurrent_streams': config.concurrent_streams
            }
            config_validation = validate_and_sanitize(config_dict, 'benchmark_config')
            if not config_validation.is_valid:
                raise ValidationError(f"Configuration validation failed: {config_validation.error_message}")
            
        if metrics is None:
            metrics = ['latency', 'throughput']
            if self.power_monitoring:
                metrics.append('power')
        
        # Load model if not already loaded
        if model_path and not self.interpreter:
            self.load_model(model_path)
            
        if not self.interpreter:
            raise RuntimeError("No model loaded for benchmarking")
        
        logger.info(f"Starting benchmark with config: warmup={config.warmup_runs}, "
                   f"runs={config.measurement_runs}, metrics={metrics}")
        
        try:
            # Prepare test data with validation
            if test_data is None:
                test_data = self._generate_dummy_data()
            else:
                # Validate input data
                data_validation = validate_and_sanitize(test_data, 'model_input')
                if not data_validation.is_valid:
                    raise ValidationError(f"Input data validation failed: {data_validation.error_message}")
                test_data = data_validation.sanitized_value or test_data
            
            # Initialize result structure
            result = BenchmarkResult(
                device_info=self.device_info,
                model_info=self.model_info,
                config=config,
                metrics={},
                raw_measurements={},
                success=False
            )
            
            # Start power monitoring if enabled
            if self.power_monitoring and self.power_monitor:
                self.power_monitor.start_monitoring()
            
            # Run warmup
            logger.info(f"Running {config.warmup_runs} warmup iterations...")
            self._run_warmup(test_data, config.warmup_runs)
            
            # Create concurrent benchmark tasks for each batch size
            benchmark_tasks = []
            for i, batch_size in enumerate(config.batch_sizes):
                batch_data = self._prepare_batch_data(test_data, batch_size)
                
                task = TaskSpec(
                    task_id=f"benchmark_batch_{batch_size}_{i}",
                    function=self._benchmark_batch,
                    args=(batch_data, batch_size, config),
                    priority=10 - i,  # Higher priority for smaller batch sizes
                    estimated_duration=config.measurement_runs * 0.01,  # Rough estimate
                    resource_requirements={ResourceType.TPU: 1.0, ResourceType.CPU: 0.5}
                )
                benchmark_tasks.append(task)
            
            # Execute batch benchmarks concurrently with caching
            cache_key = self._generate_cache_key(config, metrics)
            cached_results = self.performance_cache.get(cache_key)
            
            if cached_results:
                logger.info("Using cached benchmark results")
                batch_execution_results = cached_results
            else:
                batch_execution_results = self.concurrent_executor.execute_batch(
                    benchmark_tasks,
                    strategy_override=ExecutionStrategy.THREADED
                )
                
                # Cache successful results
                successful_results = {k: v.result for k, v in batch_execution_results.items() if v.success}
                if successful_results:
                    self.performance_cache.set(
                        cache_key, 
                        successful_results, 
                        ttl=3600.0,  # 1 hour cache
                        tags=['benchmark', 'performance']
                    )
            
            # Process execution results into raw measurements
            for task_id, execution_result in batch_execution_results.items():
                if execution_result.success and execution_result.result:
                    batch_results = execution_result.result
                    batch_size = int(task_id.split('_')[2])  # Extract batch size from task_id
                    
                    # Store batch-specific results  
                    for metric_name, values in batch_results.items():
                        key = f"{metric_name}_batch_{batch_size}"
                        result.raw_measurements[key] = values
                    
                    # Update auto-scaler metrics
                    if values:
                        avg_latency = sum(values) / len(values)
                        self.auto_scaler.add_metric(ScalingTrigger.RESPONSE_TIME, avg_latency)
                        self.auto_scaler.add_metric(ScalingTrigger.THROUGHPUT, 1000.0 / avg_latency if avg_latency > 0 else 0)
            
            # Calculate aggregate metrics
            result.metrics = self._calculate_metrics(result.raw_measurements, metrics)
            
            # Stop power monitoring
            if self.power_monitoring and self.power_monitor:
                power_data = self.power_monitor.stop_monitoring()
                result.metrics.update(self._process_power_data(power_data))
            
            result.success = True
            logger.info("Benchmark completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                device_info=self.device_info,
                model_info=self.model_info,
                config=config,
                metrics={},
                raw_measurements={},
                success=False,
                error_message=str(e)
            )
    
    def _generate_dummy_data(self) -> np.ndarray:
        """Generate dummy input data matching model requirements"""
        if not self.interpreter:
            raise RuntimeError("No interpreter available")
            
        input_details = self.interpreter.get_input_details()
        dummy_inputs = []
        
        for detail in input_details:
            shape = detail['shape']
            dtype = detail['dtype']
            
            # Generate appropriate dummy data based on dtype
            if dtype == np.float32:
                data = np.random.normal(0.0, 1.0, shape).astype(dtype)
            elif dtype == np.uint8:
                data = np.random.randint(0, 256, shape, dtype=dtype)
            elif dtype == np.int8:
                data = np.random.randint(-128, 128, shape, dtype=dtype)
            else:
                # Default to zeros for unknown dtypes
                data = np.zeros(shape, dtype=dtype)
                
            dummy_inputs.append(data)
        
        return dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs
    
    def _prepare_batch_data(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Prepare data for batch inference"""
        if batch_size == 1:
            return data
            
        # Replicate data for batch processing
        if isinstance(data, list):
            return [np.repeat(d[np.newaxis, ...], batch_size, axis=0) for d in data]
        else:
            return np.repeat(data[np.newaxis, ...], batch_size, axis=0)
    
    def _run_warmup(self, data: np.ndarray, warmup_runs: int):
        """Run warmup iterations to stabilize performance"""
        for _ in range(warmup_runs):
            self._single_inference(data)
    
    def _benchmark_batch(self, data: np.ndarray, batch_size: int, config: BenchmarkConfig) -> Dict[str, List[float]]:
        """Run benchmark measurements for a specific batch size"""
        latency_measurements = []
        
        logger.info(f"Measuring batch_size={batch_size} for {config.measurement_runs} runs...")
        
        for run in range(config.measurement_runs):
            start_time = time.perf_counter()
            
            try:
                outputs = self._single_inference(data)
                end_time = time.perf_counter()
                
                # Record latency in milliseconds
                latency_ms = (end_time - start_time) * 1000.0
                latency_measurements.append(latency_ms)
                
                # Check timeout
                if time.perf_counter() - start_time > config.timeout_seconds:
                    logger.warning(f"Benchmark timeout reached at run {run}")
                    break
                    
            except Exception as e:
                logger.error(f"Inference failed at run {run}: {e}")
                break
        
        return {
            'latency_ms': latency_measurements,
        }
    
    def _single_inference(self, data: np.ndarray) -> List[np.ndarray]:
        """Run single inference and return outputs"""
        if not self.interpreter:
            raise RuntimeError("No interpreter available")
        
        # Set input data
        input_details = self.interpreter.get_input_details()
        
        if isinstance(data, list):
            for i, input_data in enumerate(data):
                self.interpreter.set_tensor(input_details[i]['index'], input_data)
        else:
            self.interpreter.set_tensor(input_details[0]['index'], data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        output_details = self.interpreter.get_output_details()
        outputs = []
        
        for detail in output_details:
            output = self.interpreter.get_tensor(detail['index'])
            outputs.append(output.copy())  # Copy to avoid memory issues
        
        return outputs
    
    def _calculate_metrics(self, raw_measurements: Dict[str, List[float]], requested_metrics: List[str]) -> Dict[str, Any]:
        """Calculate aggregate metrics from raw measurements"""
        metrics = {}
        
        # Process latency metrics
        all_latencies = []
        for key, values in raw_measurements.items():
            if 'latency_ms' in key:
                all_latencies.extend(values)
        
        if all_latencies and 'latency' in requested_metrics:
            metrics.update({
                'latency_mean_ms': statistics.mean(all_latencies),
                'latency_median_ms': statistics.median(all_latencies),
                'latency_p50_ms': np.percentile(all_latencies, 50),
                'latency_p95_ms': np.percentile(all_latencies, 95),
                'latency_p99_ms': np.percentile(all_latencies, 99),
                'latency_std_ms': statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0,
                'latency_min_ms': min(all_latencies),
                'latency_max_ms': max(all_latencies),
            })
        
        # Calculate throughput metrics
        if all_latencies and 'throughput' in requested_metrics:
            mean_latency_s = statistics.mean(all_latencies) / 1000.0
            metrics.update({
                'throughput_fps': 1.0 / mean_latency_s if mean_latency_s > 0 else 0.0,
                'throughput_ips': 1.0 / mean_latency_s if mean_latency_s > 0 else 0.0,  # Inferences per second
            })
        
        # Add device and model metadata
        metrics.update({
            'device_type': self.device_info.device_type.value,
            'device_id': self.device_info.device_id,
            'model_size_mb': self.model_info.get('size_mb', 0),
            'total_measurements': len(all_latencies),
        })
        
        return metrics
    
    def _process_power_data(self, power_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process power monitoring data into metrics"""
        if not power_data:
            return {}
        
        power_metrics = {}
        
        if 'power_samples' in power_data:
            power_samples = power_data['power_samples']
            power_metrics.update({
                'power_mean_w': statistics.mean(power_samples),
                'power_max_w': max(power_samples),
                'power_min_w': min(power_samples),
                'power_std_w': statistics.stdev(power_samples) if len(power_samples) > 1 else 0.0,
            })
            
            # Calculate energy metrics
            if 'duration_s' in power_data:
                duration_s = power_data['duration_s']
                total_energy_j = statistics.mean(power_samples) * duration_s
                power_metrics.update({
                    'energy_total_j': total_energy_j,
                    'energy_per_inference_mj': (total_energy_j * 1000) / max(1, self.model_info.get('total_measurements', 1)),
                })
        
        return power_metrics
    
    def get_device_info(self) -> DeviceInfo:
        """Get information about the active device"""
        return self.device_info
    
    def _generate_cache_key(self, config: BenchmarkConfig, metrics: List[str]) -> str:
        """Generate cache key for benchmark results"""
        import hashlib
        
        key_data = {
            'device': self.device_info.device_type.value,
            'model_path': self.model_info.get('path', ''),
            'model_size': self.model_info.get('size_bytes', 0),
            'config': {
                'warmup_runs': config.warmup_runs,
                'measurement_runs': config.measurement_runs,
                'batch_sizes': config.batch_sizes,
            },
            'metrics': sorted(metrics)
        }
        
        key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics from all optimization components"""
        return {
            'cache_stats': self.performance_cache.get_statistics(),
            'concurrent_execution_stats': self.concurrent_executor.get_performance_stats(),
            'auto_scaler_stats': self.auto_scaler.get_statistics(),
            'security_stats': self.security_manager.get_security_report(),
            'error_recovery_stats': self.recovery_system.get_error_statistics()
        }
    
    def optimize_performance(self):
        """Trigger performance optimization across all components"""
        logger.info("Starting comprehensive performance optimization...")
        
        # Optimize cache performance
        self.performance_cache.optimize()
        
        # Clear old cache entries to free memory
        self.performance_cache.clear()
        
        # Get current metrics for auto-scaler
        cache_stats = self.performance_cache.get_statistics()
        concurrent_stats = self.concurrent_executor.get_performance_stats()
        
        # Update auto-scaler with current performance metrics
        if concurrent_stats.get('average_execution_time'):
            self.auto_scaler.add_metric(
                ScalingTrigger.RESPONSE_TIME, 
                concurrent_stats['average_execution_time'] * 1000  # Convert to ms
            )
        
        if cache_stats.get('overall', {}).get('overall_hit_rate'):
            cache_hit_rate = cache_stats['overall']['overall_hit_rate'] * 100
            self.auto_scaler.add_metric(ScalingTrigger.THROUGHPUT, cache_hit_rate)
        
        logger.info("Performance optimization completed")
    
    def shutdown(self):
        """Graceful shutdown of all benchmark components"""
        logger.info("Shutting down EdgeTPUBenchmark...")
        
        # Stop auto-scaler
        self.auto_scaler.stop()
        
        # Shutdown concurrent executor
        self.concurrent_executor.shutdown(wait=True)
        
        # Clear caches
        self.performance_cache.clear()
        
        logger.info("EdgeTPUBenchmark shutdown completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return self.model_info.copy()
    
    def list_available_devices(self) -> List[Dict[str, Any]]:
        """List all available devices"""
        return self.device_manager.list_devices()
    
    def switch_device(self, device: Union[str, DeviceType, int]) -> DeviceInfo:
        """
        Switch to a different device
        
        Args:
            device: Device specification
            
        Returns:
            New active device information
        """
        old_device = self.device_info.device_type.value
        self.device_info = self.device_manager.select_device(device)
        
        # Reload model if it was already loaded
        if self.interpreter and self.model_info:
            model_path = self.model_info['path']
            logger.info(f"Switching from {old_device} to {self.device_info.device_type.value}")
            self.load_model(model_path)
        
        return self.device_info