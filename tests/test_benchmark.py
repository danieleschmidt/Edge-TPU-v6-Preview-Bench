"""
Tests for core benchmark functionality
"""

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile

import tensorflow as tf

from edge_tpu_v6_bench.core.benchmark import (
    EdgeTPUBenchmark, BenchmarkConfig, BenchmarkResult
)
from edge_tpu_v6_bench.core.device_manager import DeviceInfo, DeviceType

class TestBenchmarkConfig:
    """Test suite for BenchmarkConfig"""
    
    def test_config_defaults(self):
        """Test BenchmarkConfig default values"""
        config = BenchmarkConfig()
        
        assert config.warmup_runs == 10
        assert config.measurement_runs == 100
        assert config.timeout_seconds == 300.0
        assert config.measure_power is False
        assert config.measure_thermal is False
        assert config.batch_sizes == [1]
        assert config.input_shapes is None
        assert config.concurrent_streams == 1
        assert config.output_precision == 4
    
    def test_config_custom_values(self):
        """Test BenchmarkConfig with custom values"""
        config = BenchmarkConfig(
            warmup_runs=20,
            measurement_runs=200,
            timeout_seconds=600.0,
            measure_power=True,
            batch_sizes=[1, 2, 4],
            target_latency_ms=10.0,
            concurrent_streams=2
        )
        
        assert config.warmup_runs == 20
        assert config.measurement_runs == 200
        assert config.timeout_seconds == 600.0
        assert config.measure_power is True
        assert config.batch_sizes == [1, 2, 4]
        assert config.target_latency_ms == 10.0
        assert config.concurrent_streams == 2

class TestBenchmarkResult:
    """Test suite for BenchmarkResult"""
    
    def test_result_creation(self):
        """Test BenchmarkResult creation"""
        device_info = DeviceInfo(
            device_id="test_device",
            device_type=DeviceType.EDGE_TPU_V6,
            path="/dev/test",
            performance_class="ultra",
            power_limit_w=6.0,
            thermal_limit_c=90.0,
            memory_mb=16,
            compiler_version="3.0.0",
            runtime_version="3.0.0"
        )
        
        config = BenchmarkConfig()
        
        result = BenchmarkResult(
            device_info=device_info,
            model_info={'size_mb': 5.0},
            config=config,
            metrics={'latency_mean_ms': 10.5},
            raw_measurements={'latency_ms': [10.1, 10.5, 10.9]},
            success=True
        )
        
        assert result.device_info == device_info
        assert result.model_info['size_mb'] == 5.0
        assert result.config == config
        assert result.metrics['latency_mean_ms'] == 10.5
        assert result.success is True
        assert result.error_message is None
        assert isinstance(result.timestamp, float)

class TestEdgeTPUBenchmark:
    """Test suite for EdgeTPUBenchmark"""
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    @patch('edge_tpu_v6_bench.core.benchmark.PowerMonitor')
    def test_benchmark_initialization(self, mock_power_monitor, mock_device_manager):
        """Test EdgeTPUBenchmark initialization"""
        # Mock device manager
        mock_manager = MagicMock()
        mock_device_info = DeviceInfo(
            device_id="test_device",
            device_type=DeviceType.EDGE_TPU_V6,
            path="/dev/test",
            performance_class="ultra",
            power_limit_w=6.0,
            thermal_limit_c=90.0,
            memory_mb=16,
            compiler_version="3.0.0",
            runtime_version="3.0.0"
        )
        mock_manager.select_device.return_value = mock_device_info
        mock_device_manager.return_value = mock_manager
        
        # Initialize benchmark
        benchmark = EdgeTPUBenchmark(
            device='edge_tpu_v6',
            power_monitoring=True,
            thermal_monitoring=False
        )
        
        assert benchmark.device_info == mock_device_info
        assert benchmark.power_monitoring is True
        assert benchmark.thermal_monitoring is False
        assert benchmark.interpreter is None
        assert benchmark.model_info == {}
        
        mock_manager.select_device.assert_called_once_with('edge_tpu_v6')
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_load_model_success(self, mock_device_manager):
        """Test successful model loading"""
        # Setup mocks
        mock_manager = MagicMock()
        mock_device_info = DeviceInfo(
            device_id="test_device",
            device_type=DeviceType.CPU_FALLBACK,
            path="/dev/cpu",
            performance_class="baseline",
            power_limit_w=15.0,
            thermal_limit_c=85.0,
            memory_mb=1024,
            compiler_version="2.14.0",
            runtime_version="2.14.0"
        )
        mock_manager.select_device.return_value = mock_device_info
        mock_device_manager.return_value = mock_manager
        
        # Mock interpreter
        mock_interpreter = MagicMock()
        mock_manager.create_interpreter.return_value = mock_interpreter
        
        # Mock interpreter details
        mock_interpreter.get_input_details.return_value = [
            {'shape': np.array([1, 224, 224, 3]), 'dtype': np.float32}
        ]
        mock_interpreter.get_output_details.return_value = [
            {'shape': np.array([1, 1000]), 'dtype': np.float32}
        ]
        
        benchmark = EdgeTPUBenchmark()
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_file:
            tmp_file.write(b'fake_tflite_data')
            tmp_file_path = tmp_file.name
        
        try:
            model_info = benchmark.load_model(tmp_file_path)
            
            assert 'path' in model_info
            assert 'size_bytes' in model_info
            assert 'size_mb' in model_info
            assert 'input_details' in model_info
            assert 'output_details' in model_info
            assert 'num_inputs' in model_info
            assert 'num_outputs' in model_info
            
            assert model_info['path'] == tmp_file_path
            assert model_info['num_inputs'] == 1
            assert model_info['num_outputs'] == 1
            assert benchmark.interpreter is not None
            
        finally:
            Path(tmp_file_path).unlink(missing_ok=True)
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_load_model_file_not_found(self, mock_device_manager):
        """Test model loading with non-existent file"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            benchmark.load_model('/nonexistent/path/model.tflite')
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_generate_dummy_data(self, mock_device_manager):
        """Test dummy data generation"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        # Mock interpreter
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [
            {'shape': np.array([1, 224, 224, 3]), 'dtype': np.float32},
            {'shape': np.array([1, 10]), 'dtype': np.int32}
        ]
        benchmark.interpreter = mock_interpreter
        
        dummy_data = benchmark._generate_dummy_data()
        
        assert isinstance(dummy_data, list)
        assert len(dummy_data) == 2
        
        # Check first input (float32)
        assert dummy_data[0].shape == (1, 224, 224, 3)
        assert dummy_data[0].dtype == np.float32
        
        # Check second input (int32)
        assert dummy_data[1].shape == (1, 10)
        assert dummy_data[1].dtype == np.int32
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_single_inference(self, mock_device_manager):
        """Test single inference execution"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        # Mock interpreter
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [
            {'index': 0, 'shape': np.array([1, 10])}
        ]
        mock_interpreter.get_output_details.return_value = [
            {'index': 0, 'shape': np.array([1, 5])}
        ]
        
        # Mock output
        mock_output = np.random.random((1, 5))
        mock_interpreter.get_tensor.return_value = mock_output
        
        benchmark.interpreter = mock_interpreter
        
        # Test single input
        input_data = np.random.random((1, 10))
        outputs = benchmark._single_inference(input_data)
        
        assert len(outputs) == 1
        assert np.array_equal(outputs[0], mock_output)
        
        mock_interpreter.set_tensor.assert_called_once_with(0, input_data)
        mock_interpreter.invoke.assert_called_once()
        mock_interpreter.get_tensor.assert_called_once_with(0)
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_prepare_batch_data(self, mock_device_manager):
        """Test batch data preparation"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        # Test single input
        data = np.random.random((224, 224, 3))
        batch_data = benchmark._prepare_batch_data(data, batch_size=4)
        
        expected_shape = (4, 224, 224, 3)
        assert batch_data.shape == expected_shape
        
        # Test batch size 1 (no change)
        batch_data_1 = benchmark._prepare_batch_data(data, batch_size=1)
        assert np.array_equal(batch_data_1, data)
        
        # Test list input
        data_list = [np.random.random((224, 224, 3)), np.random.random((10,))]
        batch_data_list = benchmark._prepare_batch_data(data_list, batch_size=2)
        
        assert isinstance(batch_data_list, list)
        assert len(batch_data_list) == 2
        assert batch_data_list[0].shape == (2, 224, 224, 3)
        assert batch_data_list[1].shape == (2, 10)
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_calculate_metrics(self, mock_device_manager):
        """Test metrics calculation"""
        mock_manager = MagicMock()
        mock_device_info = DeviceInfo(
            device_id="test_device",
            device_type=DeviceType.EDGE_TPU_V6,
            path="/dev/test",
            performance_class="ultra",
            power_limit_w=6.0,
            thermal_limit_c=90.0,
            memory_mb=16,
            compiler_version="3.0.0",
            runtime_version="3.0.0"
        )
        mock_manager.select_device.return_value = mock_device_info
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        benchmark.model_info = {'size_mb': 5.0}
        
        # Mock raw measurements
        raw_measurements = {
            'latency_ms_batch_1': [10.1, 10.5, 9.8, 10.2, 10.7],
            'latency_ms_batch_2': [20.1, 20.8, 19.5, 20.3]
        }
        
        requested_metrics = ['latency', 'throughput']
        
        metrics = benchmark._calculate_metrics(raw_measurements, requested_metrics)
        
        # Check latency metrics
        assert 'latency_mean_ms' in metrics
        assert 'latency_median_ms' in metrics
        assert 'latency_p50_ms' in metrics
        assert 'latency_p95_ms' in metrics
        assert 'latency_p99_ms' in metrics
        assert 'latency_std_ms' in metrics
        assert 'latency_min_ms' in metrics
        assert 'latency_max_ms' in metrics
        
        # Check throughput metrics
        assert 'throughput_fps' in metrics
        assert 'throughput_ips' in metrics
        
        # Check metadata
        assert 'device_type' in metrics
        assert 'device_id' in metrics
        assert 'model_size_mb' in metrics
        assert 'total_measurements' in metrics
        
        assert metrics['device_type'] == 'edge_tpu_v6'
        assert metrics['device_id'] == 'test_device'
        assert metrics['model_size_mb'] == 5.0
        assert metrics['total_measurements'] == 9  # 5 + 4 measurements
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_run_warmup(self, mock_device_manager):
        """Test warmup execution"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        # Mock single inference
        benchmark._single_inference = MagicMock()
        
        test_data = np.random.random((1, 10))
        benchmark._run_warmup(test_data, warmup_runs=5)
        
        # Should call single_inference 5 times
        assert benchmark._single_inference.call_count == 5
        
        # Check all calls used the same data
        for call in benchmark._single_inference.call_args_list:
            assert np.array_equal(call[0][0], test_data)
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    @patch('time.perf_counter')
    def test_benchmark_batch(self, mock_time, mock_device_manager):
        """Test batch benchmarking"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        # Mock timing
        time_values = [0.0, 0.01, 0.01, 0.012, 0.012, 0.015]  # 10ms, 12ms, 15ms latencies
        mock_time.side_effect = time_values
        
        # Mock single inference
        benchmark._single_inference = MagicMock()
        
        config = BenchmarkConfig(measurement_runs=3)
        test_data = np.random.random((2, 10))  # Batch size 2
        
        results = benchmark._benchmark_batch(test_data, batch_size=2, config=config)
        
        assert 'latency_ms' in results
        latencies = results['latency_ms']
        
        assert len(latencies) == 3
        assert latencies[0] == 10.0  # (0.01 - 0.0) * 1000
        assert latencies[1] == 2.0   # (0.012 - 0.01) * 1000  
        assert latencies[2] == 3.0   # (0.015 - 0.012) * 1000
        
        # Should call single_inference 3 times
        assert benchmark._single_inference.call_count == 3
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_get_device_info(self, mock_device_manager):
        """Test getting device information"""
        mock_manager = MagicMock()
        mock_device_info = DeviceInfo(
            device_id="test_device",
            device_type=DeviceType.EDGE_TPU_V6,
            path="/dev/test",
            performance_class="ultra",
            power_limit_w=6.0,
            thermal_limit_c=90.0,
            memory_mb=16,
            compiler_version="3.0.0",
            runtime_version="3.0.0"  
        )
        mock_manager.select_device.return_value = mock_device_info
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        device_info = benchmark.get_device_info()
        
        assert device_info == mock_device_info
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')  
    def test_get_model_info(self, mock_device_manager):
        """Test getting model information"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        benchmark.model_info = {
            'path': '/path/to/model.tflite',
            'size_mb': 5.0,
            'num_inputs': 1,
            'num_outputs': 1
        }
        
        model_info = benchmark.get_model_info()
        
        assert model_info['path'] == '/path/to/model.tflite'
        assert model_info['size_mb'] == 5.0
        assert model_info['num_inputs'] == 1
        assert model_info['num_outputs'] == 1
        
        # Should be a copy, not the original
        model_info['size_mb'] = 10.0
        assert benchmark.model_info['size_mb'] == 5.0
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_list_available_devices(self, mock_device_manager):
        """Test listing available devices"""
        mock_manager = MagicMock()
        mock_devices = [
            {'id': 'device1', 'type': 'edge_tpu_v6'},
            {'id': 'device2', 'type': 'cpu_fallback'}
        ]
        mock_manager.list_devices.return_value = mock_devices
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        devices = benchmark.list_available_devices()
        
        assert devices == mock_devices
        mock_manager.list_devices.assert_called_once()
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_switch_device(self, mock_device_manager):
        """Test switching to different device"""
        mock_manager = MagicMock()
        
        # Initial device
        initial_device = DeviceInfo(
            device_id="device1",
            device_type=DeviceType.EDGE_TPU_V6,
            path="/dev/test1",
            performance_class="ultra",
            power_limit_w=6.0,
            thermal_limit_c=90.0,
            memory_mb=16,
            compiler_version="3.0.0",
            runtime_version="3.0.0"
        )
        
        # New device
        new_device = DeviceInfo(
            device_id="device2",
            device_type=DeviceType.CPU_FALLBACK,
            path="/dev/cpu",
            performance_class="baseline",
            power_limit_w=15.0,
            thermal_limit_c=85.0,
            memory_mb=1024,
            compiler_version="2.14.0",
            runtime_version="2.14.0"
        )
        
        mock_manager.select_device.side_effect = [initial_device, new_device]
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        # Set up model info to test reloading
        benchmark.interpreter = MagicMock()
        benchmark.model_info = {'path': '/path/to/model.tflite'}
        
        # Mock load_model method
        benchmark.load_model = MagicMock()
        
        # Switch device
        switched_device = benchmark.switch_device('cpu_fallback')
        
        assert switched_device == new_device
        assert benchmark.device_info == new_device
        
        # Should reload model
        benchmark.load_model.assert_called_once_with('/path/to/model.tflite')
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_process_power_data(self, mock_device_manager):
        """Test power data processing"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        benchmark.model_info = {'total_measurements': 100}
        
        power_data = {
            'power_samples': [2.1, 2.3, 2.0, 2.4, 2.2],
            'duration_s': 10.0
        }
        
        power_metrics = benchmark._process_power_data(power_data)
        
        assert 'power_mean_w' in power_metrics
        assert 'power_max_w' in power_metrics
        assert 'power_min_w' in power_metrics
        assert 'power_std_w' in power_metrics
        assert 'energy_total_j' in power_metrics
        assert 'energy_per_inference_mj' in power_metrics
        
        assert power_metrics['power_mean_w'] == 2.2  # Mean of [2.1, 2.3, 2.0, 2.4, 2.2]
        assert power_metrics['power_max_w'] == 2.4
        assert power_metrics['power_min_w'] == 2.0
        assert power_metrics['energy_total_j'] == 22.0  # 2.2W * 10s
        assert power_metrics['energy_per_inference_mj'] == 220.0  # 22J * 1000 / 100 inferences
    
    @patch('edge_tpu_v6_bench.core.benchmark.DeviceManager')
    def test_process_power_data_empty(self, mock_device_manager):
        """Test power data processing with empty data"""
        mock_manager = MagicMock()
        mock_device_manager.return_value = mock_manager
        
        benchmark = EdgeTPUBenchmark()
        
        # Empty power data
        power_metrics = benchmark._process_power_data({})
        assert power_metrics == {}
        
        # None power data
        power_metrics = benchmark._process_power_data(None)
        assert power_metrics == {}