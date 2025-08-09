"""
Quantization-Aware Training (QAT) for Edge TPU optimization
Implements QAT strategies with robust error handling and validation
"""

import logging
import tempfile
from typing import Dict, List, Optional, Union, Callable, Any
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class QATOptimizer:
    """
    Quantization-Aware Training optimizer for Edge TPU devices
    
    Provides QAT functionality:
    - Model preparation for quantization-aware training
    - Custom quantization configurations
    - Training callbacks and monitoring
    - Post-QAT model conversion
    """
    
    def __init__(self, 
                 target_device: str = 'edge_tpu_v6',
                 quantization_scheme: str = 'asymmetric_uint8'):
        """
        Initialize QAT optimizer
        
        Args:
            target_device: Target Edge TPU device type
            quantization_scheme: Quantization scheme to apply
        """
        self.target_device = target_device
        self.quantization_scheme = quantization_scheme
        
        # Import TensorFlow Model Optimization if available
        try:
            import tensorflow_model_optimization as tfmot
            self.tfmot = tfmot
            self.tfmot_available = True
            logger.info("TensorFlow Model Optimization available for QAT")
        except ImportError:
            self.tfmot = None
            self.tfmot_available = False
            logger.warning("TensorFlow Model Optimization not available - QAT functionality limited")
    
    def prepare_qat(self,
                   model: tf.keras.Model,
                   custom_quantize_config: Optional[Dict[str, Dict[str, Any]]] = None) -> tf.keras.Model:
        """
        Prepare model for quantization-aware training
        
        Args:
            model: Original Keras model
            custom_quantize_config: Custom quantization configuration per layer type
            
        Returns:
            QAT-prepared model
            
        Raises:
            RuntimeError: If QAT preparation fails
        """
        if not self.tfmot_available:
            raise RuntimeError("TensorFlow Model Optimization required for QAT")
        
        logger.info("Preparing model for quantization-aware training")
        
        try:
            # Create quantization config
            if custom_quantize_config is None:
                quantize_config = self._get_default_quantize_config()
            else:
                quantize_config = self._create_custom_quantize_config(custom_quantize_config)
            
            # Apply quantization-aware training to model
            qat_model = self.tfmot.quantization.keras.quantize_model(
                model,
                quantized_layer_name_prefix='quant_'
            )
            
            logger.info(f"QAT model prepared with {len(qat_model.layers)} layers")
            
            # Validate QAT model structure
            self._validate_qat_model(qat_model)
            
            return qat_model
            
        except Exception as e:
            error_msg = f"QAT preparation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _get_default_quantize_config(self):
        """Get default quantization configuration"""
        if not self.tfmot_available:
            return None
            
        # Default configuration for Edge TPU
        if self.quantization_scheme == 'asymmetric_uint8':
            return self.tfmot.quantization.keras.QuantizeConfig(
                activation_bits=8,
                weight_bits=8,
                activation_range=(-128, 127),
                weight_range=(-128, 127)
            )
        else:
            return None  # Use default TensorFlow Model Optimization config
    
    def _create_custom_quantize_config(self, custom_config: Dict[str, Dict[str, Any]]):
        """Create custom quantization configuration"""
        if not self.tfmot_available:
            return None
            
        # This would create layer-specific quantization configs
        # Implementation depends on specific requirements
        logger.info(f"Creating custom quantization config for {len(custom_config)} layer types")
        
        # For now, return default config
        # In practice, this would map layer types to specific quantization parameters
        return self._get_default_quantize_config()
    
    def get_callbacks(self) -> List[Any]:
        """
        Get recommended callbacks for QAT training
        
        Returns:
            List of Keras callbacks for QAT monitoring
        """
        try:
            # Import tensorflow.keras.callbacks when actually needed
            import tensorflow.keras.callbacks as callbacks_module
        except ImportError:
            logger.warning("TensorFlow not available - returning empty callbacks list")
            return []
            
        callbacks = []
        
        # Learning rate scheduling for QAT
        def qat_lr_schedule(epoch, lr):
            """Learning rate schedule optimized for QAT"""
            if epoch < 5:
                return lr  # Keep initial LR for first few epochs
            elif epoch < 15:
                return lr * 0.5  # Reduce LR in middle phase
            else:
                return lr * 0.1  # Fine-tuning phase
        
        callbacks.append(callbacks_module.LearningRateScheduler(qat_lr_schedule))
        
        # Early stopping to prevent overfitting
        callbacks.append(callbacks_module.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ))
        
        # Model checkpointing
        callbacks.append(callbacks_module.ModelCheckpoint(
            filepath='qat_checkpoint_{epoch:02d}_{val_accuracy:.2f}.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ))
        
        # QAT-specific logging
        callbacks.append(self._create_qat_logging_callback())
        
        return callbacks
    
    def _create_qat_logging_callback(self) -> Any:
        """Create callback for QAT-specific logging"""
        
        try:
            # Import tensorflow.keras.callbacks when actually needed
            import tensorflow.keras.callbacks as callbacks_module
            base_class = callbacks_module.Callback
        except ImportError:
            logger.warning("TensorFlow not available - returning mock callback")
            # Return a mock callback that does nothing
            class MockCallback:
                def on_epoch_end(self, epoch, logs=None):
                    pass
            return MockCallback()
        
        class QATLoggingCallback(base_class):
            def __init__(self):
                super().__init__()
                self.qat_metrics = []
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                
                # Log QAT-specific metrics
                qat_info = {
                    'epoch': epoch,
                    'loss': logs.get('loss', 0),
                    'val_loss': logs.get('val_loss', 0),
                    'accuracy': logs.get('accuracy', 0),
                    'val_accuracy': logs.get('val_accuracy', 0),
                }
                
                self.qat_metrics.append(qat_info)
                
                logger.info(f"QAT Epoch {epoch}: "
                           f"loss={qat_info['loss']:.4f}, "
                           f"val_loss={qat_info['val_loss']:.4f}, "
                           f"acc={qat_info['accuracy']:.4f}, "
                           f"val_acc={qat_info['val_accuracy']:.4f}")
        
        return QATLoggingCallback()
    
    def convert_to_edge_tpu(self,
                           qat_model: tf.keras.Model,
                           output_path: Optional[str] = None) -> str:
        """
        Convert QAT model to Edge TPU-compatible TFLite format
        
        Args:
            qat_model: Trained QAT model
            output_path: Output path for converted model
            
        Returns:
            Path to converted TFLite model
        """
        logger.info("Converting QAT model to Edge TPU format")
        
        try:
            # Configure converter for Edge TPU
            converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
            
            # Optimization settings for Edge TPU
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Edge TPU requires full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            if self.quantization_scheme == 'asymmetric_uint8':
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            else:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            # Enable experimental features
            converter.experimental_new_converter = True
            
            # Convert model
            start_time = time.time()
            quantized_tflite = converter.convert()
            conversion_time = time.time() - start_time
            
            # Save converted model
            if output_path is None:
                output_path = self._generate_temp_path('qat_edge_tpu_model.tflite')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(quantized_tflite)
            
            logger.info(f"QAT model converted to Edge TPU format in {conversion_time:.2f}s: {output_path}")
            
            # Validate converted model
            self._validate_converted_model(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            error_msg = f"QAT to Edge TPU conversion failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _validate_qat_model(self, qat_model: tf.keras.Model):
        """
        Validate QAT model structure and configuration
        
        Args:
            qat_model: QAT-prepared model to validate
            
        Raises:
            RuntimeError: If validation fails
        """
        try:
            # Check that model has quantization layers
            has_quant_layers = any('quant' in layer.name.lower() for layer in qat_model.layers)
            
            if not has_quant_layers:
                logger.warning("QAT model may not have quantization layers")
            
            # Try to compile the model
            qat_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Test with dummy data
            input_shape = qat_model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            dummy_input = np.random.random((1,) + input_shape[1:]).astype(np.float32)
            
            # Test forward pass
            output = qat_model.predict(dummy_input, verbose=0)
            
            if output is None or output.shape[0] != 1:
                raise RuntimeError("QAT model forward pass failed")
            
            logger.debug("QAT model validation successful")
            
        except Exception as e:
            error_msg = f"QAT model validation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _validate_converted_model(self, model_path: str):
        """
        Validate converted TFLite model
        
        Args:
            model_path: Path to converted TFLite model
            
        Raises:
            RuntimeError: If validation fails
        """
        try:
            # Load and test TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test inference
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']
            
            dummy_input = np.random.random(input_shape).astype(input_dtype)
            
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            
            if output is None:
                raise RuntimeError("Converted model inference failed")
            
            # Check quantization
            is_quantized = any(detail['dtype'] != np.float32 for detail in input_details + output_details)
            
            if not is_quantized:
                logger.warning("Converted model may not be properly quantized")
            
            logger.debug("Converted model validation successful")
            
        except Exception as e:
            error_msg = f"Converted model validation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _generate_temp_path(self, filename: str) -> str:
        """Generate temporary file path"""
        temp_dir = tempfile.mkdtemp()
        return str(Path(temp_dir) / filename)
    
    def estimate_qat_improvement(self,
                               original_model: tf.keras.Model,
                               validation_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Estimate potential improvement from QAT
        
        Args:
            original_model: Original model before QAT
            validation_data: Validation dataset
            
        Returns:
            Dictionary with estimated improvements
        """
        try:
            # This is a simplified estimation
            # In practice, would require actual QAT training to measure
            
            # Count quantizable parameters
            total_params = original_model.count_params()
            quantizable_params = self._count_quantizable_params(original_model)
            
            quantizable_ratio = quantizable_params / total_params if total_params > 0 else 0
            
            # Estimate compression and speedup
            estimated_compression = 1.0 + (3.0 * quantizable_ratio)  # Up to 4x compression
            estimated_speedup = 1.0 + (1.5 * quantizable_ratio)  # Up to 2.5x speedup
            
            # Estimate accuracy retention (QAT typically preserves accuracy better)
            estimated_accuracy_retention = 0.98 + (0.015 * quantizable_ratio)  # Up to 99.5%
            
            return {
                'estimated_compression_ratio': estimated_compression,
                'estimated_speedup': estimated_speedup,
                'estimated_accuracy_retention': estimated_accuracy_retention,
                'quantizable_param_ratio': quantizable_ratio,
            }
            
        except Exception as e:
            logger.error(f"QAT improvement estimation failed: {e}")
            return {}
    
    def _count_quantizable_params(self, model: tf.keras.Model) -> int:
        """Count parameters in quantizable layers"""
        quantizable_params = 0
        
        for layer in model.layers:
            layer_type = type(layer).__name__
            
            # Check if layer is quantizable
            if any(qtype in layer_type for qtype in ['Conv', 'Dense', 'Linear']):
                if hasattr(layer, 'count_params'):
                    quantizable_params += layer.count_params()
        
        return quantizable_params