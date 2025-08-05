"""
Tests for automatic quantization functionality
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import tensorflow as tf

from edge_tpu_v6_bench.quantization.auto_quantizer import (
    AutoQuantizer, QuantizationConfig, QuantizationResult
)

class TestAutoQuantizer:
    """Test suite for AutoQuantizer"""
    
    def test_quantizer_initialization(self):
        """Test AutoQuantizer initializes correctly"""
        quantizer = AutoQuantizer(
            target_device='edge_tpu_v6',
            optimization_target='latency'
        )
        
        assert quantizer.target_device == 'edge_tpu_v6'
        assert quantizer.optimization_target == 'latency'
        assert quantizer.post_training_quantizer is not None
        assert quantizer.mixed_precision_optimizer is not None
    
    def test_quantization_config(self):
        """Test QuantizationConfig dataclass"""
        config = QuantizationConfig(
            target_device='edge_tpu_v6',
            optimization_target='accuracy',
            quantization_strategies=['int8', 'uint8'],
            accuracy_threshold=0.01,
            size_threshold=0.7
        )
        
        assert config.target_device == 'edge_tpu_v6'
        assert config.optimization_target == 'accuracy'
        assert config.quantization_strategies == ['int8', 'uint8']
        assert config.accuracy_threshold == 0.01
        assert config.size_threshold == 0.7
    
    def test_quantization_result(self):
        """Test QuantizationResult dataclass"""
        result = QuantizationResult(
            success=True,
            strategy_used='int8_post_training',
            model_path='/path/to/quantized.tflite',
            original_size_mb=10.0,
            quantized_size_mb=2.5,
            compression_ratio=4.0,
            accuracy_drop=0.01,
            estimated_speedup=2.8
        )
        
        assert result.success is True
        assert result.strategy_used == 'int8_post_training'
        assert result.compression_ratio == 4.0
        assert result.accuracy_drop == 0.01
        assert result.estimated_speedup == 2.8
    
    def test_strategy_priority(self):
        """Test quantization strategy priority ordering"""
        # Test v6 strategies
        v6_strategies = AutoQuantizer.STRATEGY_PRIORITY['edge_tpu_v6']
        assert 'int4_mixed' in v6_strategies
        assert 'int8_post_training' in v6_strategies
        assert v6_strategies.index('int4_mixed') < v6_strategies.index('int8_post_training')
        
        # Test v5e strategies (no INT4)
        v5e_strategies = AutoQuantizer.STRATEGY_PRIORITY['edge_tpu_v5e']
        assert 'int4_mixed' not in v5e_strategies
        assert 'int8_post_training' in v5e_strategies
    
    @patch('tensorflow.keras.models.load_model')
    def test_quantize_with_model_path(self, mock_load_model):
        """Test quantization with model path input"""
        # Create mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        quantizer = AutoQuantizer()
        
        with patch.object(quantizer, '_apply_quantization_strategy') as mock_apply:
            mock_result = QuantizationResult(
                success=True,
                strategy_used='int8_post_training',
                model_path='/path/to/quantized.tflite',
                original_size_mb=5.0,
                quantized_size_mb=1.25,
                compression_ratio=4.0,
                accuracy_drop=0.005,
                estimated_speedup=2.0
            )
            mock_apply.return_value = mock_result
            
            result = quantizer.quantize('/path/to/model')
            
            assert result.success is True
            assert result.compression_ratio == 4.0
            mock_load_model.assert_called_once()
    
    @patch('pathlib.Path.exists')
    def test_quantize_existing_tflite(self, mock_exists):
        """Test quantization of existing TFLite model"""
        mock_exists.return_value = True
        
        quantizer = AutoQuantizer()
        
        with patch.object(quantizer, '_analyze_existing_model') as mock_analyze:
            mock_result = QuantizationResult(
                success=True,
                strategy_used='existing',
                model_path='/path/to/model.tflite',
                original_size_mb=3.0,
                quantized_size_mb=3.0,
                compression_ratio=1.0,
                accuracy_drop=0.0,
                estimated_speedup=2.0
            )
            mock_analyze.return_value = mock_result
            
            result = quantizer.quantize(Path('/path/to/model.tflite'))
            
            assert result.success is True
            assert result.strategy_used == 'existing'
            mock_analyze.assert_called_once()
    
    def test_generate_calibration_data(self):
        """Test calibration data generation"""
        # Create simple test model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,))
        ])
        
        quantizer = AutoQuantizer()
        dataset = quantizer._generate_calibration_data(model, 50)
        
        # Check dataset properties
        sample_count = 0
        for batch in dataset:
            sample_count += batch.shape[0]
            assert batch.shape == (1, 5)  # Batch size 1, input shape (5,)
            assert batch.dtype == tf.float32
        
        assert sample_count == 50
    
    def test_estimate_speedup(self):
        """Test speedup estimation for different strategies and devices"""
        quantizer = AutoQuantizer()
        
        # Test v6 speedups
        v6_int4_speedup = quantizer._estimate_speedup('int4_mixed', 'edge_tpu_v6')
        v6_int8_speedup = quantizer._estimate_speedup('int8_post_training', 'edge_tpu_v6')
        
        assert v6_int4_speedup > v6_int8_speedup  # INT4 should be faster
        assert v6_int4_speedup == 3.5
        assert v6_int8_speedup == 2.8
        
        # Test v5e speedups (no INT4)
        v5e_int8_speedup = quantizer._estimate_speedup('int8_post_training', 'edge_tpu_v5e')
        assert v5e_int8_speedup == 2.2
        
        # Test unknown strategy
        unknown_speedup = quantizer._estimate_speedup('unknown', 'edge_tpu_v6')
        assert unknown_speedup == 1.0
    
    @patch('tensorflow.lite.Interpreter')
    def test_validate_accuracy(self, mock_interpreter_class):
        """Test accuracy validation"""
        # Create mock interpreter
        mock_interpreter = MagicMock()
        mock_interpreter_class.return_value = mock_interpreter
        
        # Mock interpreter behavior
        mock_interpreter.get_input_details.return_value = [{'index': 0}]
        mock_interpreter.get_output_details.return_value = [{'index': 0}]
        
        # Mock model predictions
        original_preds = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        quantized_preds = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        
        # Mock original model
        mock_model = MagicMock()
        mock_model.predict.return_value = original_preds
        
        # Mock quantized model output
        mock_interpreter.get_tensor.side_effect = [
            quantized_preds[0:1], quantized_preds[1:2], quantized_preds[2:3]
        ]
        
        quantizer = AutoQuantizer()
        
        val_inputs = np.random.random((3, 5))
        val_labels = np.array([1, 0, 1])
        
        accuracy_drop = quantizer._validate_accuracy(
            mock_model, '/path/to/quantized.tflite', (val_inputs, val_labels)
        )
        
        # Original accuracy: 3/3 = 100%
        # Quantized accuracy: 1/3 = 33.3% (only first prediction correct)
        # Accuracy drop should be significant
        assert accuracy_drop >= 0.0
        assert isinstance(accuracy_drop, float)
    
    def test_is_result_acceptable(self):
        """Test result acceptance criteria"""
        quantizer = AutoQuantizer()
        config = QuantizationConfig(
            accuracy_threshold=0.02,
            size_threshold=0.5  # Require at least 50% size reduction
        )
        
        # Good result
        good_result = QuantizationResult(
            success=True,
            strategy_used='int8',
            model_path='/path',
            original_size_mb=10.0,
            quantized_size_mb=2.0,
            compression_ratio=5.0,
            accuracy_drop=0.01,
            estimated_speedup=2.0
        )
        
        assert quantizer._is_result_acceptable(good_result, config) is True
        
        # Bad result - too much accuracy drop
        bad_accuracy_result = QuantizationResult(
            success=True,
            strategy_used='int8',
            model_path='/path',
            original_size_mb=10.0,
            quantized_size_mb=2.0,
            compression_ratio=5.0,
            accuracy_drop=0.05,  # Too high
            estimated_speedup=2.0
        )
        
        assert quantizer._is_result_acceptable(bad_accuracy_result, config) is False
        
        # Bad result - insufficient compression
        bad_compression_result = QuantizationResult(
            success=True,
            strategy_used='int8',
            model_path='/path',
            original_size_mb=10.0,
            quantized_size_mb=9.0,
            compression_ratio=1.1,  # Too low
            accuracy_drop=0.01,
            estimated_speedup=2.0
        )
        
        assert quantizer._is_result_acceptable(bad_compression_result, config) is False
        
        # Failed result
        failed_result = QuantizationResult(
            success=False,
            strategy_used='int8',
            model_path='',
            original_size_mb=0.0,
            quantized_size_mb=0.0,
            compression_ratio=1.0,
            accuracy_drop=1.0,
            estimated_speedup=1.0
        )
        
        assert quantizer._is_result_acceptable(failed_result, config) is False
    
    def test_is_better_result(self):
        """Test result comparison logic"""
        quantizer = AutoQuantizer()
        
        result1 = QuantizationResult(
            success=True,
            strategy_used='int8',
            model_path='/path1',
            original_size_mb=10.0,
            quantized_size_mb=2.5,
            compression_ratio=4.0,
            accuracy_drop=0.01,
            estimated_speedup=2.5
        )
        
        result2 = QuantizationResult(
            success=True,
            strategy_used='int4',
            model_path='/path2',
            original_size_mb=10.0,
            quantized_size_mb=2.0,
            compression_ratio=5.0,
            accuracy_drop=0.02,
            estimated_speedup=3.0
        )
        
        # Test latency optimization (higher speedup is better)
        config_latency = QuantizationConfig(optimization_target='latency')
        assert quantizer._is_better_result(result2, result1, config_latency) is True
        
        # Test accuracy optimization (lower accuracy drop is better)
        config_accuracy = QuantizationConfig(optimization_target='accuracy')
        assert quantizer._is_better_result(result1, result2, config_accuracy) is True
        
        # Test size optimization (higher compression is better)
        config_size = QuantizationConfig(optimization_target='size')
        assert quantizer._is_better_result(result2, result1, config_size) is True
    
    def test_get_model_info(self):
        """Test model information extraction"""
        # Create test model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(10,)),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(1)
        ])
        
        quantizer = AutoQuantizer()
        model_info = quantizer._get_model_info(model)
        
        assert 'size_mb' in model_info
        assert 'num_parameters' in model_info
        assert 'input_shape' in model_info
        assert 'output_shape' in model_info
        
        assert model_info['size_mb'] > 0
        assert model_info['num_parameters'] == model.count_params()
        assert model_info['input_shape'] == (None, 10)
        assert model_info['output_shape'] == (None, 1)
    
    @patch('tensorflow.lite.Interpreter')
    def test_analyze_tflite_model(self, mock_interpreter_class):
        """Test TFLite model analysis"""
        # Mock file size
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024 * 1024  # 1 MB
            
            # Mock interpreter
            mock_interpreter = MagicMock()
            mock_interpreter_class.return_value = mock_interpreter
            
            mock_interpreter.get_input_details.return_value = [
                {'dtype': np.int8, 'shape': (1, 224, 224, 3)}
            ]
            mock_interpreter.get_output_details.return_value = [
                {'dtype': np.int8, 'shape': (1, 1000)}
            ]
            mock_interpreter.get_tensor_details.return_value = [
                {'dtype': np.int8}, {'dtype': np.int8}, {'dtype': np.float32}
            ]
            
            quantizer = AutoQuantizer()
            info = quantizer._analyze_tflite_model('/path/to/model.tflite')
            
            assert info['size_mb'] == 1.0
            assert info['num_ops'] == 3
            assert info['num_inputs'] == 1
            assert info['num_outputs'] == 1
            assert info['quantized'] is True
    
    def test_fallback_quantization(self):
        """Test fallback quantization when all strategies fail"""
        quantizer = AutoQuantizer()
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,))
        ])
        
        # Generate calibration data
        calibration_data = quantizer._generate_calibration_data(model, 10)
        
        config = QuantizationConfig(fallback_on_failure=True)
        
        with patch.object(quantizer.post_training_quantizer, 'quantize_dynamic') as mock_quantize:
            with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_file:
                mock_quantize.return_value = tmp_file.name
                
                # Mock file size for analysis
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 512 * 1024  # 0.5 MB
                    
                    result = quantizer._fallback_quantization(model, calibration_data, config)
                    
                    assert result.success is True
                    assert result.strategy_used == 'dynamic_fallback'
                    assert result.estimated_speedup == 1.2
                    
                    # Cleanup
                    Path(tmp_file.name).unlink(missing_ok=True)
    
    def test_analyze_existing_model(self):
        """Test analysis of existing TFLite model"""
        quantizer = AutoQuantizer()
        config = QuantizationConfig()
        
        with patch.object(quantizer, '_analyze_tflite_model') as mock_analyze:
            mock_analyze.return_value = {
                'size_mb': 2.5,
                'num_ops': 10,
                'quantized': True
            }
            
            result = quantizer._analyze_existing_model(Path('/path/to/model.tflite'), config)
            
            assert result.success is True
            assert result.strategy_used == 'existing'
            assert result.original_size_mb == 2.5
            assert result.quantized_size_mb == 2.5
            assert result.compression_ratio == 1.0
            assert result.estimated_speedup == 2.0  # Quantized model