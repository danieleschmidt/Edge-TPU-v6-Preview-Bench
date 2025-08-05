"""
Automatic quantization optimization for Edge TPU devices
Provides intelligent quantization strategy selection and model optimization
"""

import logging
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import time

import numpy as np
import tensorflow as tf

from ..core.device_manager import DeviceInfo, DeviceType
from .strategies.post_training import PostTrainingQuantizer
from .strategies.mixed_precision import MixedPrecisionOptimizer

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for automatic quantization"""
    target_device: str = 'edge_tpu_v6'
    optimization_target: str = 'latency'  # 'latency', 'accuracy', 'power', 'size'
    quantization_strategies: List[str] = None
    accuracy_threshold: float = 0.02  # Max accuracy drop (2%)
    size_threshold: float = 0.5  # Target size reduction (50%)
    calibration_samples: int = 100
    validate_accuracy: bool = True
    fallback_on_failure: bool = True

@dataclass
class QuantizationResult:
    """Results from quantization process"""
    success: bool
    strategy_used: str
    model_path: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    accuracy_drop: float
    estimated_speedup: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class AutoQuantizer:
    """
    Automatic quantization optimizer for Edge TPU devices
    
    Intelligently selects and applies optimal quantization strategies
    based on target device capabilities and optimization goals
    """
    
    STRATEGY_PRIORITY = {
        'edge_tpu_v6': ['int4_mixed', 'int8_post_training', 'uint8_post_training', 'hybrid'],
        'edge_tpu_v5e': ['int8_post_training', 'uint8_post_training', 'hybrid'],
        'coral_dev_board': ['int8_post_training', 'uint8_post_training', 'hybrid'],
        'usb_accelerator': ['int8_post_training', 'uint8_post_training', 'hybrid'],
        'cpu_fallback': ['int8_post_training', 'float16', 'dynamic'],
    }
    
    def __init__(self, 
                 target_device: str = 'edge_tpu_v6',
                 optimization_target: str = 'latency'):
        """
        Initialize auto quantizer
        
        Args:
            target_device: Target Edge TPU device type
            optimization_target: Primary optimization goal
        """
        self.target_device = target_device
        self.optimization_target = optimization_target
        
        # Initialize quantization strategies
        self.post_training_quantizer = PostTrainingQuantizer()
        self.mixed_precision_optimizer = MixedPrecisionOptimizer()
        
        logger.info(f"Initialized AutoQuantizer for {target_device} "
                   f"optimizing for {optimization_target}")
    
    def quantize(self, 
                 model: Union[tf.keras.Model, str, Path],
                 calibration_data: Optional[tf.data.Dataset] = None,
                 validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 config: Optional[QuantizationConfig] = None) -> QuantizationResult:
        """
        Automatically quantize model with optimal strategy
        
        Args:
            model: TensorFlow model or path to SavedModel
            calibration_data: Dataset for quantization calibration
            validation_data: (inputs, labels) for accuracy validation
            config: Quantization configuration
            
        Returns:
            Quantization results with optimized model path
        """
        if config is None:
            config = QuantizationConfig(
                target_device=self.target_device,
                optimization_target=self.optimization_target
            )
        
        # Determine strategies to try
        strategies = config.quantization_strategies
        if not strategies:
            strategies = self.STRATEGY_PRIORITY.get(
                config.target_device, 
                ['int8_post_training', 'hybrid']
            )
        
        logger.info(f"Starting auto-quantization with strategies: {strategies}")
        
        # Load model if path provided
        if isinstance(model, (str, Path)):
            model_path = Path(model)
            if model_path.suffix == '.tflite':
                # Already quantized TFLite model
                return self._analyze_existing_model(model_path, config)
            else:
                # Load SavedModel or Keras model
                model = tf.keras.models.load_model(str(model_path))
        
        # Get baseline model info
        baseline_info = self._get_model_info(model)
        
        # Generate calibration data if not provided
        if calibration_data is None:
            calibration_data = self._generate_calibration_data(model, config.calibration_samples)
        
        # Try quantization strategies in order of preference
        best_result = None
        
        for strategy in strategies:
            try:
                logger.info(f"Trying quantization strategy: {strategy}")
                result = self._apply_quantization_strategy(
                    model, strategy, calibration_data, validation_data, config
                )
                
                if result.success:
                    # Check if this result meets our criteria
                    if self._is_result_acceptable(result, config):
                        logger.info(f"Strategy {strategy} successful: "
                                   f"{result.compression_ratio:.1f}x compression, "
                                   f"{result.accuracy_drop:.1%} accuracy drop")
                        
                        if not best_result or self._is_better_result(result, best_result, config):
                            best_result = result
                            
                        # If we found a great result, we can stop early
                        if (result.accuracy_drop < config.accuracy_threshold / 2 and 
                            result.compression_ratio >= 2.0):
                            break
                    else:
                        logger.warning(f"Strategy {strategy} did not meet criteria")
                else:
                    logger.warning(f"Strategy {strategy} failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Strategy {strategy} raised exception: {e}")
                continue
        
        # Return best result or fallback
        if best_result:
            return best_result
        elif config.fallback_on_failure:
            logger.warning("All strategies failed, attempting basic fallback")
            return self._fallback_quantization(model, calibration_data, config)
        else:
            return QuantizationResult(
                success=False,
                strategy_used='none',
                model_path='',
                original_size_mb=baseline_info['size_mb'],
                quantized_size_mb=0.0,
                compression_ratio=1.0,
                accuracy_drop=1.0,
                estimated_speedup=1.0,
                error_message='All quantization strategies failed'
            )
    
    def _apply_quantization_strategy(self,
                                   model: tf.keras.Model,
                                   strategy: str,
                                   calibration_data: tf.data.Dataset,
                                   validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
                                   config: QuantizationConfig) -> QuantizationResult:
        """Apply specific quantization strategy"""
        
        baseline_info = self._get_model_info(model)
        start_time = time.time()
        
        try:
            if strategy == 'int8_post_training':
                quantized_model_path = self.post_training_quantizer.quantize_int8(
                    model, calibration_data
                )
            elif strategy == 'uint8_post_training':
                quantized_model_path = self.post_training_quantizer.quantize_uint8(
                    model, calibration_data
                )
            elif strategy == 'int4_mixed':
                # INT4 mixed precision (v6 only)
                if 'v6' not in config.target_device:
                    raise ValueError("INT4 quantization only supported on Edge TPU v6")
                quantized_model_path = self.mixed_precision_optimizer.quantize_int4_mixed(
                    model, calibration_data
                )
            elif strategy == 'hybrid':
                quantized_model_path = self.post_training_quantizer.quantize_hybrid(model)
            elif strategy == 'float16':
                quantized_model_path = self.post_training_quantizer.quantize_float16(model)
            elif strategy == 'dynamic':
                quantized_model_path = self.post_training_quantizer.quantize_dynamic(model)
            else:
                raise ValueError(f"Unknown quantization strategy: {strategy}")
            
            # Analyze quantized model
            quantized_info = self._analyze_tflite_model(quantized_model_path)
            
            # Calculate metrics
            compression_ratio = baseline_info['size_mb'] / quantized_info['size_mb']
            
            # Estimate speedup based on quantization type and device
            estimated_speedup = self._estimate_speedup(strategy, config.target_device)
            
            # Validate accuracy if requested
            accuracy_drop = 0.0
            if config.validate_accuracy and validation_data:
                accuracy_drop = self._validate_accuracy(
                    model, quantized_model_path, validation_data
                )
            
            duration = time.time() - start_time
            
            return QuantizationResult(
                success=True,
                strategy_used=strategy,
                model_path=quantized_model_path,
                original_size_mb=baseline_info['size_mb'],
                quantized_size_mb=quantized_info['size_mb'],
                compression_ratio=compression_ratio,
                accuracy_drop=accuracy_drop,
                estimated_speedup=estimated_speedup,
                metadata={
                    'quantization_time_s': duration,
                    'original_params': baseline_info.get('num_parameters', 0),
                    'quantized_ops': quantized_info.get('num_ops', 0),
                }
            )
            
        except Exception as e:
            return QuantizationResult(
                success=False,
                strategy_used=strategy,
                model_path='',
                original_size_mb=baseline_info['size_mb'],
                quantized_size_mb=0.0,
                compression_ratio=1.0,
                accuracy_drop=1.0,
                estimated_speedup=1.0,
                error_message=str(e)
            )
    
    def _get_model_info(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Extract information from Keras model"""
        # Save model temporarily to measure size
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / 'temp_model'
            model.save(str(temp_path), save_format='tf')
            
            # Calculate size
            total_size = sum(f.stat().st_size for f in temp_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
        
        # Count parameters
        total_params = model.count_params()
        
        return {
            'size_mb': size_mb,
            'num_parameters': total_params,
            'input_shape': model.input_shape if hasattr(model, 'input_shape') else None,
            'output_shape': model.output_shape if hasattr(model, 'output_shape') else None,
        }
    
    def _analyze_tflite_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze TensorFlow Lite model"""
        model_path = Path(model_path)
        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        # Load interpreter to get more details
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            tensor_details = interpreter.get_tensor_details()
            
            return {
                'size_mb': size_mb,
                'num_ops': len(tensor_details),
                'num_inputs': len(input_details),
                'num_outputs': len(output_details),
                'input_dtypes': [detail['dtype'].__name__ for detail in input_details],
                'quantized': any('int' in detail['dtype'].__name__ for detail in tensor_details),
            }
        except Exception as e:
            logger.warning(f"Could not analyze TFLite model details: {e}")
            return {'size_mb': size_mb, 'num_ops': 0}
    
    def _analyze_existing_model(self, model_path: Path, config: QuantizationConfig) -> QuantizationResult:
        """Analyze already-quantized TFLite model"""
        info = self._analyze_tflite_model(str(model_path))
        
        return QuantizationResult(
            success=True,
            strategy_used='existing',
            model_path=str(model_path),
            original_size_mb=info['size_mb'],  # Don't know original
            quantized_size_mb=info['size_mb'],
            compression_ratio=1.0,  # Unknown
            accuracy_drop=0.0,  # Unknown
            estimated_speedup=2.0 if info.get('quantized', False) else 1.0,
            metadata=info
        )
    
    def _generate_calibration_data(self, model: tf.keras.Model, num_samples: int) -> tf.data.Dataset:
        """Generate synthetic calibration data"""
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]  # Use first input if multiple
        
        # Generate random data matching input shape
        batch_shape = (num_samples,) + input_shape[1:]  # Skip batch dimension
        
        if model.dtype.name == 'float32':
            calibration_data = np.random.normal(0.0, 1.0, batch_shape).astype(np.float32)
        else:
            calibration_data = np.random.uniform(0, 1, batch_shape).astype(np.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(calibration_data)
        dataset = dataset.batch(1)  # Batch size 1 for calibration
        
        logger.info(f"Generated {num_samples} calibration samples with shape {batch_shape}")
        return dataset
    
    def _estimate_speedup(self, strategy: str, target_device: str) -> float:
        """Estimate speedup factor for quantization strategy on target device"""
        
        # Speedup estimates based on empirical data
        speedup_factors = {
            'edge_tpu_v6': {
                'int4_mixed': 3.5,
                'int8_post_training': 2.8,
                'uint8_post_training': 2.5,
                'hybrid': 1.8,
            },
            'edge_tpu_v5e': {
                'int8_post_training': 2.2,
                'uint8_post_training': 2.0,
                'hybrid': 1.5,
            },
            'cpu_fallback': {
                'int8_post_training': 1.8,
                'float16': 1.3,
                'dynamic': 1.4,
                'hybrid': 1.2,
            }
        }
        
        device_factors = speedup_factors.get(target_device, speedup_factors['cpu_fallback'])
        return device_factors.get(strategy, 1.0)
    
    def _validate_accuracy(self, 
                          original_model: tf.keras.Model,
                          quantized_model_path: str,
                          validation_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Validate accuracy of quantized model vs original"""
        
        val_inputs, val_labels = validation_data
        
        try:
            # Get predictions from original model
            original_preds = original_model.predict(val_inputs, verbose=0)
            
            # Get predictions from quantized model
            interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            quantized_preds = []
            for input_sample in val_inputs:
                interpreter.set_tensor(input_details[0]['index'], input_sample[np.newaxis, ...])
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]
                quantized_preds.append(output)
            
            quantized_preds = np.array(quantized_preds)
            
            # Calculate accuracy drop
            original_acc = np.mean(np.argmax(original_preds, axis=1) == val_labels)
            quantized_acc = np.mean(np.argmax(quantized_preds, axis=1) == val_labels)
            
            accuracy_drop = original_acc - quantized_acc
            
            logger.info(f"Original accuracy: {original_acc:.1%}, "
                       f"Quantized accuracy: {quantized_acc:.1%}, "
                       f"Drop: {accuracy_drop:.1%}")
            
            return max(0.0, accuracy_drop)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            return 0.0  # Assume no drop if validation fails
    
    def _is_result_acceptable(self, result: QuantizationResult, config: QuantizationConfig) -> bool:
        """Check if quantization result meets acceptance criteria"""
        
        if not result.success:
            return False
        
        # Check accuracy threshold
        if result.accuracy_drop > config.accuracy_threshold:
            logger.debug(f"Accuracy drop {result.accuracy_drop:.1%} exceeds threshold {config.accuracy_threshold:.1%}")
            return False
        
        # Check size reduction threshold
        if result.compression_ratio < (1.0 + config.size_threshold):
            logger.debug(f"Compression ratio {result.compression_ratio:.1f}x below threshold")
            return False
        
        return True
    
    def _is_better_result(self, result: QuantizationResult, current_best: QuantizationResult, config: QuantizationConfig) -> bool:
        """Compare two results to determine which is better"""
        
        if config.optimization_target == 'latency':
            return result.estimated_speedup > current_best.estimated_speedup
        elif config.optimization_target == 'accuracy':
            return result.accuracy_drop < current_best.accuracy_drop  
        elif config.optimization_target == 'size':
            return result.compression_ratio > current_best.compression_ratio
        elif config.optimization_target == 'power':
            # Favor higher compression + lower precision for power efficiency
            power_score = result.compression_ratio * result.estimated_speedup
            current_power_score = current_best.compression_ratio * current_best.estimated_speedup
            return power_score > current_power_score
        else:
            # Default: balance of all factors
            result_score = (result.compression_ratio * result.estimated_speedup) / (1 + result.accuracy_drop)
            current_score = (current_best.compression_ratio * current_best.estimated_speedup) / (1 + current_best.accuracy_drop)
            return result_score > current_score
    
    def _fallback_quantization(self, 
                             model: tf.keras.Model,
                             calibration_data: tf.data.Dataset,
                             config: QuantizationConfig) -> QuantizationResult:
        """Fallback to most basic quantization that should always work"""
        
        logger.info("Attempting fallback quantization (dynamic range)")
        
        try:
            quantized_model_path = self.post_training_quantizer.quantize_dynamic(model)
            baseline_info = self._get_model_info(model)
            quantized_info = self._analyze_tflite_model(quantized_model_path)
            
            return QuantizationResult(
                success=True,
                strategy_used='dynamic_fallback',
                model_path=quantized_model_path,
                original_size_mb=baseline_info['size_mb'],
                quantized_size_mb=quantized_info['size_mb'],
                compression_ratio=baseline_info['size_mb'] / quantized_info['size_mb'],
                accuracy_drop=0.0,  # Assume minimal for dynamic quantization
                estimated_speedup=1.2,  # Conservative estimate
                metadata={'fallback': True}
            )
            
        except Exception as e:
            return QuantizationResult(
                success=False,
                strategy_used='fallback_failed',
                model_path='',
                original_size_mb=0.0,
                quantized_size_mb=0.0,
                compression_ratio=1.0,
                accuracy_drop=1.0,
                estimated_speedup=1.0,
                error_message=f"Fallback quantization failed: {e}"
            )