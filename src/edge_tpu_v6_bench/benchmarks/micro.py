"""
Micro-benchmarks for individual operations and kernels
High-performance, concurrent micro-benchmarking with optimization
"""

import logging
import time
import statistics
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..core.device_manager import DeviceManager, DeviceInfo
from ..core.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)

@dataclass
class MicroBenchmarkConfig:
    """Configuration for micro-benchmarks"""
    warmup_runs: int = 20
    measurement_runs: int = 200
    concurrent_streams: int = 4
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    timeout_seconds: float = 120.0
    enable_profiling: bool = True
    output_precision: int = 6

@dataclass
class OperationResult:
    """Results from a single operation benchmark"""
    operation_name: str
    input_shape: Tuple[int, ...]
    parameters: Dict[str, Any]
    latency_stats: Dict[str, float]
    throughput_ops_per_sec: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None

class MicroBenchmarkSuite:
    """
    High-performance micro-benchmark suite for Edge TPU operations
    
    Provides detailed benchmarking of individual operations:
    - Convolution operations (1D, 2D, 3D)
    - Matrix multiplication variants
    - Element-wise operations
    - Pooling operations
    - Activation functions
    - Normalization layers
    
    Features:
    - Concurrent execution for throughput testing
    - Memory usage profiling
    - Performance optimization analysis
    - Cross-device comparison
    """
    
    def __init__(self, 
                 device: Union[str, DeviceInfo] = 'auto',
                 config: Optional[MicroBenchmarkConfig] = None):
        """
        Initialize micro-benchmark suite
        
        Args:
            device: Target device for benchmarking
            config: Benchmark configuration
        """
        self.device_manager = DeviceManager()
        self.device_info = (device if isinstance(device, DeviceInfo) 
                           else self.device_manager.select_device(device))
        self.config = config or MicroBenchmarkConfig()
        self.metrics = BenchmarkMetrics()
        
        # Thread pool for concurrent benchmarks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.concurrent_streams
        )
        
        logger.info(f"Initialized MicroBenchmarkSuite on {self.device_info.device_type.value}")
    
    def benchmark_convolutions(self,
                             input_shapes: List[Tuple[int, ...]] = None,
                             filter_sizes: List[int] = None,
                             strides: List[int] = None,
                             quantization: str = 'int8') -> Dict[str, List[OperationResult]]:
        """
        Benchmark convolution operations with various configurations
        
        Args:
            input_shapes: List of input tensor shapes
            filter_sizes: List of filter/kernel sizes to test
            strides: List of stride values
            quantization: Quantization type ('int8', 'uint8', 'float32')
            
        Returns:
            Dictionary of convolution benchmark results
        """
        if input_shapes is None:
            input_shapes = [
                (1, 224, 224, 3),   # ImageNet input
                (1, 112, 112, 32),  # Mid-layer feature map
                (1, 56, 56, 64),    # Deeper layer
                (1, 28, 28, 128),   # High-level features
            ]
        
        if filter_sizes is None:
            filter_sizes = [1, 3, 5, 7]
        
        if strides is None:
            strides = [1, 2]
        
        logger.info(f"Benchmarking convolutions: {len(input_shapes)} shapes, "
                   f"{len(filter_sizes)} filter sizes, {len(strides)} strides")
        
        results = {
            'conv2d': [],
            'depthwise_conv2d': [],
            'separable_conv2d': []
        }
        
        # Generate all parameter combinations
        test_configs = []
        for input_shape in input_shapes:
            for filter_size in filter_sizes:
                for stride in strides:
                    for conv_type in ['conv2d', 'depthwise_conv2d', 'separable_conv2d']:
                        test_configs.append({
                            'conv_type': conv_type,
                            'input_shape': input_shape,
                            'filter_size': filter_size,
                            'stride': stride,
                            'quantization': quantization
                        })
        
        # Run benchmarks concurrently
        futures = []
        for config in test_configs:
            future = self.thread_pool.submit(self._benchmark_single_convolution, config)
            futures.append((future, config))
        
        # Collect results
        for future, config in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                results[config['conv_type']].append(result)
            except Exception as e:
                logger.error(f"Convolution benchmark failed for {config}: {e}")
                
                # Add failed result
                failed_result = OperationResult(
                    operation_name=f"{config['conv_type']}_{config['filter_size']}x{config['filter_size']}",
                    input_shape=config['input_shape'],
                    parameters=config,
                    latency_stats={},
                    throughput_ops_per_sec=0.0,
                    memory_usage_mb=0.0,
                    success=False,
                    error_message=str(e)
                )
                results[config['conv_type']].append(failed_result)
        
        # Log summary
        total_successful = sum(len([r for r in results[k] if r.success]) for k in results)
        total_tests = sum(len(results[k]) for k in results)
        logger.info(f"Convolution benchmarks completed: {total_successful}/{total_tests} successful")
        
        return results
    
    def _benchmark_single_convolution(self, config: Dict[str, Any]) -> OperationResult:
        """Benchmark a single convolution configuration"""
        
        conv_type = config['conv_type']
        input_shape = config['input_shape']
        filter_size = config['filter_size']
        stride = config['stride']
        quantization = config['quantization']
        
        operation_name = f"{conv_type}_{filter_size}x{filter_size}_s{stride}"
        
        try:
            # Create test model
            model = self._create_convolution_model(config)
            
            # Convert to TFLite for device testing
            tflite_model = self._convert_model_to_tflite(model, quantization)
            
            # Create interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Generate test data
            test_data = self._generate_test_data(input_shape, quantization)
            
            # Benchmark the operation
            latency_measurements = []
            
            # Warmup
            for _ in range(self.config.warmup_runs):
                self._single_inference(interpreter, test_data)
            
            # Measurements
            for _ in range(self.config.measurement_runs):
                start_time = time.perf_counter()
                self._single_inference(interpreter, test_data)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000.0
                latency_measurements.append(latency_ms)
            
            # Calculate statistics
            latency_stats = {
                'mean_ms': statistics.mean(latency_measurements),
                'median_ms': statistics.median(latency_measurements),
                'p95_ms': np.percentile(latency_measurements, 95),
                'p99_ms': np.percentile(latency_measurements, 99),
                'std_ms': statistics.stdev(latency_measurements),
                'min_ms': min(latency_measurements),
                'max_ms': max(latency_measurements),
            }
            
            # Calculate throughput
            mean_latency_s = latency_stats['mean_ms'] / 1000.0
            throughput_ops_per_sec = 1.0 / mean_latency_s if mean_latency_s > 0 else 0.0
            
            # Estimate memory usage
            memory_usage_mb = self._estimate_memory_usage(model)
            
            return OperationResult(
                operation_name=operation_name,
                input_shape=input_shape,
                parameters=config,
                latency_stats=latency_stats,
                throughput_ops_per_sec=throughput_ops_per_sec,
                memory_usage_mb=memory_usage_mb,
                success=True
            )
            
        except Exception as e:
            return OperationResult(
                operation_name=operation_name,
                input_shape=input_shape,
                parameters=config,
                latency_stats={},
                throughput_ops_per_sec=0.0,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_matmul(self,
                        m_sizes: List[int] = None,
                        n_sizes: List[int] = None,
                        k_sizes: List[int] = None,
                        quantization: str = 'int8') -> List[OperationResult]:
        """
        Benchmark matrix multiplication operations
        
        Args:
            m_sizes: List of M dimensions for matrices
            n_sizes: List of N dimensions for matrices  
            k_sizes: List of K dimensions for matrices
            quantization: Quantization type
            
        Returns:
            List of matrix multiplication benchmark results
        """
        if m_sizes is None:
            m_sizes = [64, 128, 256, 512, 1024]
        if n_sizes is None:
            n_sizes = [64, 128, 256, 512, 1024]
        if k_sizes is None:
            k_sizes = [64, 128, 256, 512, 1024]
        
        logger.info(f"Benchmarking matrix multiplication: {len(m_sizes)}x{len(n_sizes)}x{len(k_sizes)} configurations")
        
        results = []
        
        # Generate test configurations
        test_configs = []
        for m in m_sizes:
            for n in n_sizes:
                for k in k_sizes:
                    test_configs.append({
                        'operation': 'matmul',
                        'm': m, 'n': n, 'k': k,
                        'quantization': quantization
                    })
        
        # Run benchmarks concurrently
        futures = []
        for config in test_configs:
            future = self.thread_pool.submit(self._benchmark_single_matmul, config)
            futures.append((future, config))
        
        # Collect results
        for future, config in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                results.append(result)
            except Exception as e:
                logger.error(f"MatMul benchmark failed for {config}: {e}")
                
                failed_result = OperationResult(
                    operation_name=f"matmul_{config['m']}x{config['n']}x{config['k']}",
                    input_shape=(config['m'], config['k']),
                    parameters=config,
                    latency_stats={},
                    throughput_ops_per_sec=0.0,
                    memory_usage_mb=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)
        
        successful_results = [r for r in results if r.success]
        logger.info(f"MatMul benchmarks completed: {len(successful_results)}/{len(results)} successful")
        
        return results
    
    def _benchmark_single_matmul(self, config: Dict[str, Any]) -> OperationResult:
        """Benchmark a single matrix multiplication configuration"""
        
        m, n, k = config['m'], config['n'], config['k']
        quantization = config['quantization']
        operation_name = f"matmul_{m}x{n}x{k}"
        
        try:
            # Create test model with matrix multiplication
            input_a = tf.keras.Input(shape=(m, k), name='input_a')
            input_b = tf.keras.Input(shape=(k, n), name='input_b')
            output = tf.linalg.matmul(input_a, input_b)
            model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
            
            # Convert to TFLite
            tflite_model = self._convert_model_to_tflite(model, quantization)
            
            # Create interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Generate test data
            if quantization == 'int8':
                test_a = np.random.randint(-128, 127, (1, m, k), dtype=np.int8)
                test_b = np.random.randint(-128, 127, (1, k, n), dtype=np.int8)
            elif quantization == 'uint8':
                test_a = np.random.randint(0, 255, (1, m, k), dtype=np.uint8)
                test_b = np.random.randint(0, 255, (1, k, n), dtype=np.uint8)
            else:
                test_a = np.random.normal(0, 1, (1, m, k)).astype(np.float32)
                test_b = np.random.normal(0, 1, (1, k, n)).astype(np.float32)
            
            # Benchmark
            latency_measurements = []
            
            # Warmup
            for _ in range(self.config.warmup_runs):
                self._single_inference_multi_input(interpreter, [test_a, test_b])
            
            # Measurements
            for _ in range(self.config.measurement_runs):
                start_time = time.perf_counter()
                self._single_inference_multi_input(interpreter, [test_a, test_b])
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000.0
                latency_measurements.append(latency_ms)
            
            # Calculate statistics
            latency_stats = {
                'mean_ms': statistics.mean(latency_measurements),
                'median_ms': statistics.median(latency_measurements),
                'p95_ms': np.percentile(latency_measurements, 95),
                'p99_ms': np.percentile(latency_measurements, 99),
                'std_ms': statistics.stdev(latency_measurements),
                'min_ms': min(latency_measurements),
                'max_ms': max(latency_measurements),
            }
            
            # Calculate FLOPS and throughput
            flops_per_matmul = 2 * m * n * k  # 2 operations per multiply-add
            mean_latency_s = latency_stats['mean_ms'] / 1000.0
            throughput_ops_per_sec = 1.0 / mean_latency_s if mean_latency_s > 0 else 0.0
            
            # Estimate memory usage
            element_size = 1 if 'int8' in quantization else 4  # bytes
            memory_usage_mb = ((m * k + k * n + m * n) * element_size) / (1024 * 1024)
            
            return OperationResult(
                operation_name=operation_name,
                input_shape=(m, k),
                parameters=config,
                latency_stats=latency_stats,
                throughput_ops_per_sec=throughput_ops_per_sec,
                memory_usage_mb=memory_usage_mb,
                success=True
            )
            
        except Exception as e:
            return OperationResult(
                operation_name=operation_name,
                input_shape=(m, k),
                parameters=config,
                latency_stats={},
                throughput_ops_per_sec=0.0,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_elementwise(self,
                            operations: List[str] = None,
                            tensor_sizes: List[Tuple[int, ...]] = None,
                            quantization: str = 'int8') -> List[OperationResult]:
        """
        Benchmark element-wise operations
        
        Args:
            operations: List of operations to benchmark
            tensor_sizes: List of tensor shapes to test
            quantization: Quantization type
            
        Returns:
            List of element-wise operation benchmark results
        """
        if operations is None:
            operations = ['add', 'multiply', 'relu', 'sigmoid', 'tanh', 'maximum']
        
        if tensor_sizes is None:
            tensor_sizes = [
                (1, 1000),           # Vector
                (1, 32, 32, 128),    # Feature map
                (1, 224, 224, 3),    # Image
                (1, 64, 64, 256),    # Mid-size tensor
            ]
        
        logger.info(f"Benchmarking element-wise operations: {len(operations)} ops, {len(tensor_sizes)} sizes")
        
        results = []
        
        # Generate test configurations
        test_configs = []
        for operation in operations:
            for tensor_size in tensor_sizes:
                test_configs.append({
                    'operation': operation,
                    'tensor_size': tensor_size,
                    'quantization': quantization
                })
        
        # Run benchmarks concurrently
        futures = []
        for config in test_configs:
            future = self.thread_pool.submit(self._benchmark_single_elementwise, config)
            futures.append((future, config))
        
        # Collect results
        for future, config in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                results.append(result)
            except Exception as e:
                logger.error(f"Element-wise benchmark failed for {config}: {e}")
                
                failed_result = OperationResult(
                    operation_name=f"{config['operation']}_{config['tensor_size']}",
                    input_shape=config['tensor_size'],
                    parameters=config,
                    latency_stats={},
                    throughput_ops_per_sec=0.0,
                    memory_usage_mb=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)
        
        successful_results = [r for r in results if r.success]
        logger.info(f"Element-wise benchmarks completed: {len(successful_results)}/{len(results)} successful")
        
        return results
    
    def _benchmark_single_elementwise(self, config: Dict[str, Any]) -> OperationResult:
        """Benchmark a single element-wise operation"""
        
        operation = config['operation']
        tensor_size = config['tensor_size']
        quantization = config['quantization']
        operation_name = f"{operation}_{tensor_size}"
        
        try:
            # Create test model with element-wise operation
            model = self._create_elementwise_model(operation, tensor_size)
            
            # Convert to TFLite
            tflite_model = self._convert_model_to_tflite(model, quantization)
            
            # Create interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Generate test data
            test_data = self._generate_test_data(tensor_size, quantization)
            
            # Benchmark
            latency_measurements = []
            
            # Warmup
            for _ in range(self.config.warmup_runs):
                self._single_inference(interpreter, test_data)
            
            # Measurements
            for _ in range(self.config.measurement_runs):
                start_time = time.perf_counter()
                self._single_inference(interpreter, test_data)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000.0
                latency_measurements.append(latency_ms)
            
            # Calculate statistics
            latency_stats = {
                'mean_ms': statistics.mean(latency_measurements),
                'median_ms': statistics.median(latency_measurements),
                'p95_ms': np.percentile(latency_measurements, 95),
                'p99_ms': np.percentile(latency_measurements, 99),
                'std_ms': statistics.stdev(latency_measurements),
                'min_ms': min(latency_measurements),
                'max_ms': max(latency_measurements),
            }
            
            # Calculate throughput
            mean_latency_s = latency_stats['mean_ms'] / 1000.0
            throughput_ops_per_sec = 1.0 / mean_latency_s if mean_latency_s > 0 else 0.0
            
            # Estimate memory usage
            num_elements = np.prod(tensor_size)
            element_size = 1 if 'int8' in quantization else 4
            memory_usage_mb = (num_elements * element_size) / (1024 * 1024)
            
            return OperationResult(
                operation_name=operation_name,
                input_shape=tensor_size,
                parameters=config,
                latency_stats=latency_stats,
                throughput_ops_per_sec=throughput_ops_per_sec,
                memory_usage_mb=memory_usage_mb,
                success=True
            )
            
        except Exception as e:
            return OperationResult(
                operation_name=operation_name,
                input_shape=tensor_size,
                parameters=config,
                latency_stats={},
                throughput_ops_per_sec=0.0,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _create_convolution_model(self, config: Dict[str, Any]) -> tf.keras.Model:
        """Create a test model for convolution benchmarking"""
        conv_type = config['conv_type']
        input_shape = config['input_shape']
        filter_size = config['filter_size']
        stride = config['stride']
        
        input_tensor = tf.keras.Input(shape=input_shape[1:])  # Remove batch dimension
        
        if conv_type == 'conv2d':
            output = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=filter_size,
                strides=stride,
                padding='same',
                activation=None
            )(input_tensor)
        elif conv_type == 'depthwise_conv2d':
            output = tf.keras.layers.DepthwiseConv2D(
                kernel_size=filter_size,
                strides=stride,
                padding='same',
                activation=None
            )(input_tensor)
        elif conv_type == 'separable_conv2d':
            output = tf.keras.layers.SeparableConv2D(
                filters=32,
                kernel_size=filter_size,
                strides=stride,
                padding='same',
                activation=None
            )(input_tensor)
        else:
            raise ValueError(f"Unknown convolution type: {conv_type}")
        
        return tf.keras.Model(inputs=input_tensor, outputs=output)
    
    def _create_elementwise_model(self, operation: str, tensor_size: Tuple[int, ...]) -> tf.keras.Model:
        """Create a test model for element-wise operation benchmarking"""
        input_tensor = tf.keras.Input(shape=tensor_size[1:])  # Remove batch dimension
        
        if operation == 'add':
            # Add with a constant
            output = tf.keras.layers.Add()([input_tensor, input_tensor])
        elif operation == 'multiply':
            # Multiply with a constant
            output = tf.keras.layers.Multiply()([input_tensor, input_tensor])
        elif operation == 'relu':
            output = tf.keras.layers.ReLU()(input_tensor)
        elif operation == 'sigmoid':
            output = tf.keras.activations.sigmoid(input_tensor)
        elif operation == 'tanh':
            output = tf.keras.activations.tanh(input_tensor)
        elif operation == 'maximum':
            output = tf.keras.layers.Maximum()([input_tensor, input_tensor * 0.5])
        else:
            raise ValueError(f"Unknown element-wise operation: {operation}")
        
        return tf.keras.Model(inputs=input_tensor, outputs=output)
    
    def _convert_model_to_tflite(self, model: tf.keras.Model, quantization: str) -> bytes:
        """Convert Keras model to TFLite with specified quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif quantization == 'uint8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        else:  # float32
            pass  # No quantization
        
        return converter.convert()
    
    def _generate_test_data(self, shape: Tuple[int, ...], quantization: str) -> np.ndarray:
        """Generate test data for benchmarking"""
        if quantization == 'int8':
            return np.random.randint(-128, 127, shape, dtype=np.int8)
        elif quantization == 'uint8':
            return np.random.randint(0, 255, shape, dtype=np.uint8)
        else:
            return np.random.normal(0, 1, shape).astype(np.float32)
    
    def _single_inference(self, interpreter: tf.lite.Interpreter, input_data: np.ndarray):
        """Run single inference on TFLite interpreter"""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        return interpreter.get_tensor(output_details[0]['index'])
    
    def _single_inference_multi_input(self, interpreter: tf.lite.Interpreter, input_data_list: List[np.ndarray]):
        """Run single inference with multiple inputs"""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        for i, input_data in enumerate(input_data_list):
            interpreter.set_tensor(input_details[i]['index'], input_data)
        
        interpreter.invoke()
        
        return interpreter.get_tensor(output_details[0]['index'])
    
    def _estimate_memory_usage(self, model: tf.keras.Model) -> float:
        """Estimate memory usage of model in MB"""
        total_params = model.count_params()
        # Assume 4 bytes per parameter (float32) or 1 byte for quantized
        bytes_per_param = 1  # Assuming quantized for Edge TPU
        memory_bytes = total_params * bytes_per_param
        return memory_bytes / (1024 * 1024)
    
    def generate_report(self, results: Dict[str, Any], output_path: str = 'micro_benchmarks_report.html'):
        """
        Generate comprehensive HTML report of micro-benchmark results
        
        Args:
            results: Dictionary of benchmark results
            output_path: Path for output HTML report
        """
        logger.info(f"Generating micro-benchmark report: {output_path}")
        
        try:
            html_content = self._generate_html_report(results)
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Edge TPU v6 Micro-Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .results-table {{ border-collapse: collapse; width: 100%; }}
                .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .results-table th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .performance-good {{ background-color: #d4edda; }}
                .performance-medium {{ background-color: #fff3cd; }}
                .performance-poor {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Edge TPU v6 Micro-Benchmark Report</h1>
                <p>Device: {self.device_info.device_type.value}</p>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>This report contains detailed micro-benchmark results for individual operations on Edge TPU v6.</p>
            </div>
            
            <div class="section">
                <h2>Benchmark Results</h2>
                {self._format_results_tables(results)}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_results_tables(self, results: Dict[str, Any]) -> str:
        """Format results as HTML tables"""
        tables_html = ""
        
        for category, category_results in results.items():
            if not category_results:
                continue
                
            tables_html += f"<h3>{category.replace('_', ' ').title()}</h3>"
            tables_html += '<table class="results-table">'
            tables_html += """
            <tr>
                <th>Operation</th>
                <th>Input Shape</th>
                <th>Mean Latency (ms)</th>
                <th>P95 Latency (ms)</th>
                <th>Throughput (ops/sec)</th>
                <th>Memory (MB)</th>
                <th>Status</th>
            </tr>
            """
            
            for result in category_results:
                if isinstance(result, OperationResult):
                    status_class = "success" if result.success else "failure"
                    status_text = "✓ Success" if result.success else f"✗ Failed: {result.error_message}"
                    
                    mean_latency = result.latency_stats.get('mean_ms', 0)
                    p95_latency = result.latency_stats.get('p95_ms', 0)
                    
                    tables_html += f"""
                    <tr>
                        <td>{result.operation_name}</td>
                        <td>{result.input_shape}</td>
                        <td>{mean_latency:.{self.config.output_precision}f}</td>
                        <td>{p95_latency:.{self.config.output_precision}f}</td>
                        <td>{result.throughput_ops_per_sec:.2f}</td>
                        <td>{result.memory_usage_mb:.2f}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
                    """
            
            tables_html += "</table><br>"
        
        return tables_html
    
    def cleanup(self):
        """Clean up resources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            logger.info("Thread pool shut down")