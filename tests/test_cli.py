"""
Tests for command-line interface
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from click.testing import CliRunner

from edge_tpu_v6_bench.cli import cli, t, load_config, save_results

class TestI18n:
    """Test internationalization functionality"""
    
    def test_translation_function(self):
        """Test translation function with different languages"""
        # Test English (default)
        with patch('edge_tpu_v6_bench.cli.CURRENT_LANG', 'en'):
            assert t('device_detected') == 'Device detected'
            assert t('benchmark_completed') == 'Benchmark completed'
        
        # Test Spanish
        with patch('edge_tpu_v6_bench.cli.CURRENT_LANG', 'es'):
            assert t('device_detected') == 'Dispositivo detectado'
            assert t('benchmark_completed') == 'Benchmark completado'
        
        # Test French
        with patch('edge_tpu_v6_bench.cli.CURRENT_LANG', 'fr'):
            assert t('device_detected') == 'Périphérique détecté'
            assert t('benchmark_completed') == 'Benchmark terminé'
        
        # Test unknown key (should return key itself)
        assert t('unknown_key') == 'unknown_key'
        
        # Test unknown language (should fallback to English)
        with patch('edge_tpu_v6_bench.cli.CURRENT_LANG', 'unknown'):
            assert t('device_detected') == 'Device detected'

class TestConfigManagement:
    """Test configuration loading and management"""
    
    def test_load_config_existing_file(self):
        """Test loading existing configuration file"""
        config_data = {
            'device': 'edge_tpu_v6',
            'output_dir': '/custom/output',
            'benchmark': {
                'warmup_runs': 20,
                'measurement_runs': 200
            }
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='device: edge_tpu_v6\noutput_dir: /custom/output\nbenchmark:\n  warmup_runs: 20\n  measurement_runs: 200')):
                with patch('yaml.safe_load', return_value=config_data):
                    config = load_config('test_config.yaml')
                    
                    assert config['device'] == 'edge_tpu_v6'
                    assert config['output_dir'] == '/custom/output'
                    assert config['benchmark']['warmup_runs'] == 20
    
    def test_load_config_nonexistent_file(self):
        """Test loading non-existent configuration file"""
        with patch('pathlib.Path.exists', return_value=False):
            config = load_config('nonexistent.yaml')
            assert config == {}
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML configuration"""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='invalid: yaml: content:')):
                with patch('yaml.safe_load', side_effect=Exception('Invalid YAML')):
                    config = load_config('invalid.yaml')
                    assert config == {}
    
    def test_save_results(self):
        """Test saving benchmark results"""
        results_data = {
            'timestamp': 1234567890,
            'device': 'edge_tpu_v6',
            'model_path': '/path/to/model.tflite',
            'results': {
                'latency_mean_ms': 10.5,
                'throughput_fps': 95.2
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = save_results(results_data, temp_dir)
            
            # Check that files were created
            assert Path(output_path).exists()
            assert output_path.endswith('.json')
            
            # Check JSON content
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data['device'] == 'edge_tpu_v6'
            assert saved_data['results']['latency_mean_ms'] == 10.5
            
            # Check YAML file exists
            yaml_path = output_path.replace('.json', '.yaml')
            assert Path(yaml_path).exists()

class TestCLICommands:
    """Test CLI command functionality"""
    
    def setUp(self):
        """Set up test runner"""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Edge TPU v6 Benchmark Suite' in result.output
        assert 'Comprehensive benchmarking' in result.output
    
    @patch('edge_tpu_v6_bench.cli.DeviceManager')
    def test_devices_command(self, mock_device_manager):
        """Test devices list command"""
        # Mock device manager
        mock_manager = MagicMock()
        mock_devices = [
            {
                'id': 'edge_tpu_v6_preview',
                'type': 'edge_tpu_v6',
                'performance': 'ultra',
                'power_limit_w': 6.0,
                'memory_mb': 16,
                'features': {'int4': True, 'structured_sparsity': True}
            },
            {
                'id': 'cpu_fallback',
                'type': 'cpu_fallback',
                'performance': 'baseline',
                'power_limit_w': 15.0,
                'memory_mb': 1024,
                'features': {'int4': False, 'structured_sparsity': False}
            }
        ]
        mock_manager.list_devices.return_value = mock_devices
        mock_device_manager.return_value = mock_manager
        
        runner = CliRunner()
        result = runner.invoke(cli, ['devices'])
        
        assert result.exit_code == 0
        assert '📱 Available devices:' in result.output
        assert 'edge_tpu_v6_preview' in result.output
        assert 'cpu_fallback' in result.output
        assert 'INT4' in result.output
        assert 'Sparsity' in result.output
    
    @patch('edge_tpu_v6_bench.cli.DeviceManager')
    def test_devices_command_no_devices(self, mock_device_manager):
        """Test devices command with no devices"""
        mock_manager = MagicMock()
        mock_manager.list_devices.return_value = []
        mock_device_manager.return_value = mock_manager
        
        runner = CliRunner()
        result = runner.invoke(cli, ['devices'])
        
        assert result.exit_code == 0
        assert '⚠️  No Edge TPU devices detected' in result.output
    
    @patch('edge_tpu_v6_bench.cli.EdgeTPUBenchmark')
    @patch('pathlib.Path.exists')
    def test_benchmark_command_success(self, mock_exists, mock_benchmark_class):
        """Test successful benchmark command"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock benchmark
        mock_benchmark = MagicMock()
        mock_benchmark_class.return_value = mock_benchmark
        
        # Mock device info
        from edge_tpu_v6_bench.core.device_manager import DeviceInfo, DeviceType
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
        mock_benchmark.device_info = mock_device_info
        
        # Mock benchmark result
        from edge_tpu_v6_bench.core.benchmark import BenchmarkResult, BenchmarkConfig
        mock_result = BenchmarkResult(
            device_info=mock_device_info,
            model_info={'size_mb': 5.0},
            config=BenchmarkConfig(),
            metrics={
                'latency_mean_ms': 10.5,
                'latency_p95_ms': 12.0,
                'latency_p99_ms': 15.0,
                'throughput_fps': 95.2
            },
            raw_measurements={},
            success=True
        )
        mock_benchmark.benchmark.return_value = mock_result
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                '--output', temp_dir,
                'benchmark',
                '--model', '/path/to/model.tflite',
                '--runs', '50'
            ])
            
            assert result.exit_code == 0
            assert '🔧 Benchmark started' in result.output or 'benchmark_started' in result.output
            assert '✅' in result.output
            assert '10.50 ms' in result.output
            assert '95.2 FPS' in result.output
    
    @patch('pathlib.Path.exists')
    def test_benchmark_command_file_not_found(self, mock_exists):
        """Test benchmark command with non-existent model file"""
        mock_exists.return_value = False
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'benchmark',
            '--model', '/nonexistent/model.tflite'
        ])
        
        assert result.exit_code == 1
        assert '❌ Model file not found' in result.output
    
    @patch('edge_tpu_v6_bench.cli.MicroBenchmarkSuite')
    def test_micro_command(self, mock_micro_class):
        """Test micro-benchmark command"""
        # Mock micro benchmark suite
        mock_micro = MagicMock()
        mock_micro_class.return_value = mock_micro
        
        # Mock device info
        from edge_tpu_v6_bench.core.device_manager import DeviceInfo, DeviceType
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
        mock_micro.device_info = mock_device_info
        
        # Mock benchmark results
        mock_conv_results = {'conv2d': []}
        mock_matmul_results = []
        mock_elementwise_results = []
        
        mock_micro.benchmark_convolutions.return_value = mock_conv_results
        mock_micro.benchmark_matmul.return_value = mock_matmul_results
        mock_micro.benchmark_elementwise.return_value = mock_elementwise_results
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                '--output', temp_dir,
                'micro',
                '--operations', 'conv2d',
                '--quantization', 'int8'
            ])
            
            assert result.exit_code == 0
            assert '🔬 Starting micro-benchmarks' in result.output
            assert '✅ Micro-benchmarks completed' in result.output
            
            # Check that benchmark methods were called
            mock_micro.benchmark_convolutions.assert_called_once()
            mock_micro.generate_report.assert_called_once()
            mock_micro.cleanup.assert_called_once()
    
    @patch('edge_tpu_v6_bench.cli.StandardBenchmark')
    def test_standard_command(self, mock_standard_class):
        """Test standard benchmark command"""
        # Mock standard benchmark
        mock_standard = MagicMock()
        mock_standard_class.return_value = mock_standard
        
        # Mock device info
        from edge_tpu_v6_bench.core.device_manager import DeviceInfo, DeviceType
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
        mock_standard.device_info = mock_device_info
        
        # Mock benchmark results
        from edge_tpu_v6_bench.benchmarks.standard import ModelBenchmarkResult
        mock_results = [
            ModelBenchmarkResult(
                model_name='mobilenet_v3_small',
                model_type='vision',
                framework='tflite',
                device_type='edge_tpu_v6',
                quantization_applied='int8',
                model_size_mb=5.0,
                accuracy_metrics={'accuracy_top1': 75.0},
                performance_metrics={'latency_mean_ms': 10.5, 'throughput_fps': 95.2},
                power_metrics={'power_mean_w': 3.2},
                success=True
            )
        ]
        mock_standard.benchmark_vision_models.return_value = mock_results
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                '--output', temp_dir,
                'standard',
                '--models', 'mobilenet_v3_small',
                '--category', 'vision'
            ])
            
            assert result.exit_code == 0
            assert '📈 Starting standard benchmarks' in result.output
            assert '✅ Standard benchmarks completed' in result.output
            assert '1/1' in result.output  # Success rate
            
            mock_standard.benchmark_vision_models.assert_called_once()
            mock_standard.cleanup.assert_called_once()
    
    @patch('edge_tpu_v6_bench.cli.AutoQuantizer')
    @patch('pathlib.Path.exists')
    def test_quantize_command(self, mock_exists, mock_quantizer_class):
        """Test quantization command"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock quantizer
        mock_quantizer = MagicMock()
        mock_quantizer_class.return_value = mock_quantizer
        
        # Mock quantization result
        from edge_tpu_v6_bench.quantization.auto_quantizer import QuantizationResult
        mock_result = QuantizationResult(
            success=True,
            strategy_used='int8_post_training',
            model_path='/path/to/quantized.tflite',
            original_size_mb=10.0,
            quantized_size_mb=2.5,
            compression_ratio=4.0,
            accuracy_drop=0.01,
            estimated_speedup=2.8
        )
        mock_quantizer.quantize.return_value = mock_result
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                '--output', temp_dir,
                'quantize',
                '--input', '/path/to/model',
                '--strategy', 'int8'
            ])
            
            assert result.exit_code == 0
            assert '⚡ Quantizing model' in result.output
            assert '✅ Quantization successful!' in result.output
            assert 'int8_post_training' in result.output
            assert '4.0x' in result.output
            assert '2.8x' in result.output
    
    def test_analyze_command_no_results(self):
        """Test analyze command with no results"""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                'analyze',
                '--results-dir', temp_dir
            ])
            
            assert result.exit_code == 1
            assert '❌ No benchmark results found' in result.output
    
    def test_analyze_command_with_results(self):
        """Test analyze command with results"""
        # Create mock result files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample result file
            result_data = {
                'timestamp': 1234567890,
                'model_path': '/path/to/model1.tflite',
                'device': 'edge_tpu_v6',
                'results': {
                    'latency_mean_ms': 10.5,
                    'throughput_fps': 95.2
                }
            }
            
            result_file = Path(temp_dir) / 'benchmark_results_20240101_120000.json'
            with open(result_file, 'w') as f:
                json.dump(result_data, f)
            
            runner = CliRunner()
            result = runner.invoke(cli, [
                'analyze',
                '--results-dir', temp_dir,
                '--format', 'table'
            ])
            
            assert result.exit_code == 0
            assert '📊 Analyzing 1 result files' in result.output
            assert '📈 Results Analysis' in result.output
            assert 'model1' in result.output
            assert 'edge_tpu_v6' in result.output
    
    def test_cli_language_setting(self):
        """Test CLI language setting"""
        runner = CliRunner()
        
        # Test Spanish language
        result = runner.invoke(cli, ['--lang', 'es', '--help'])
        assert result.exit_code == 0
        # The help text itself is in English, but the initialization message should show Spanish
        
        # Test invalid language (should not crash)
        result = runner.invoke(cli, ['--lang', 'invalid', '--help'])
        assert result.exit_code == 0
    
    def test_cli_verbose_logging(self):
        """Test verbose logging option"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', '--help'])
        
        assert result.exit_code == 0
        # With verbose flag, logging level should be set to DEBUG
    
    def test_cli_config_file(self):
        """Test custom config file option"""
        config_data = {
            'device': 'edge_tpu_v5e',
            'benchmark': {'runs': 50}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            config_file.write('device: edge_tpu_v5e\nbenchmark:\n  runs: 50')
            config_file.flush()
            
            try:
                runner = CliRunner()
                result = runner.invoke(cli, ['--config', config_file.name, '--help'])
                
                assert result.exit_code == 0
                
            finally:
                Path(config_file.name).unlink(missing_ok=True)
    
    @patch('edge_tpu_v6_bench.cli.StandardBenchmark')
    def test_standard_command_device_comparison(self, mock_standard_class):
        """Test standard benchmark with device comparison"""
        # Mock standard benchmark
        mock_standard = MagicMock()
        mock_standard_class.return_value = mock_standard
        
        # Mock comparison results
        mock_comparison = {
            'devices': ['edge_tpu_v6', 'edge_tpu_v5e', 'cpu_fallback'],
            'rankings': {
                'edge_tpu_v6': {'rank': 1, 'score': 8.5},
                'edge_tpu_v5e': {'rank': 2, 'score': 12.0},
                'cpu_fallback': {'rank': 3, 'score': 25.0}
            }
        }
        mock_standard.compare_devices.return_value = mock_comparison
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                '--output', temp_dir,
                'standard',
                '--compare-devices'
            ])
            
            assert result.exit_code == 0
            assert '🔄 Comparing' in result.output
            assert '📊 Device Comparison Results:' in result.output
            assert 'edge_tpu_v6' in result.output
            assert 'edge_tpu_v5e' in result.output
            assert 'cpu_fallback' in result.output
            
            mock_standard.compare_devices.assert_called_once()
            mock_standard.plot_comparison.assert_called_once()
            mock_standard.cleanup.assert_called_once()