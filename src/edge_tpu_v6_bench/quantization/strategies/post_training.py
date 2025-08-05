"""
Post-training quantization strategies for Edge TPU devices
Implements various post-training quantization methods with robust error handling
"""

import logging
import tempfile
from typing import Optional, Union, Iterator
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class PostTrainingQuantizer:
    """
    Post-training quantization implementation with comprehensive error handling
    
    Provides multiple quantization strategies:
    - INT8 full integer quantization
    - UINT8 quantization for Edge TPU compatibility
    - Float16 quantization for reduced memory
    - Dynamic range quantization
    - Hybrid quantization
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize post-training quantizer
        
        Args:
            temp_dir: Directory for temporary files (None for system temp)
        """
        self.temp_dir = Path(temp_dir) if temp_dir else None
        
    def quantize_int8(self, 
                      model: Union[tf.keras.Model, str],
                      calibration_data: tf.data.Dataset,
                      output_path: Optional[str] = None) -> str:
        """
        Apply INT8 full integer quantization
        
        Args:
            model: Keras model or path to SavedModel
            calibration_data: Dataset for quantization calibration
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized TFLite model
            
        Raises:
            RuntimeError: If quantization fails
        """
        logger.info("Starting INT8 post-training quantization")
        
        try:
            # Load model if string path provided
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Create representative dataset function
            def representative_dataset() -> Iterator[List[tf.Tensor]]:
                for sample in calibration_data.take(100):  # Limit calibration samples
                    yield [sample]
            
            # Configure converter for INT8 quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            
            # Force full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Enable experimental features for better quantization
            converter.experimental_new_converter = True
            
            # Convert model
            start_time = time.time()
            quantized_tflite = converter.convert()
            conversion_time = time.time() - start_time
            
            # Save quantized model
            if output_path is None:
                output_path = self._generate_temp_path('int8_quantized.tflite')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(quantized_tflite)
            
            logger.info(f"INT8 quantization completed in {conversion_time:.2f}s, "
                       f"output: {output_path}")
            
            # Validate the quantized model
            self._validate_quantized_model(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            error_msg = f"INT8 quantization failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def quantize_uint8(self,
                       model: Union[tf.keras.Model, str], 
                       calibration_data: tf.data.Dataset,
                       output_path: Optional[str] = None) -> str:
        """
        Apply UINT8 quantization optimized for Edge TPU
        
        Args:
            model: Keras model or path to SavedModel
            calibration_data: Dataset for quantization calibration
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized TFLite model
        """
        logger.info("Starting UINT8 post-training quantization for Edge TPU")
        
        try:
            # Load model if string path provided
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Create representative dataset function
            def representative_dataset() -> Iterator[List[tf.Tensor]]:
                for sample in calibration_data.take(100):
                    # Ensure input is in [0, 1] range for UINT8
                    if hasattr(sample, 'numpy'):
                        sample_np = sample.numpy()
                    else:
                        sample_np = np.array(sample)
                    
                    # Normalize to [0, 1] if not already
                    if sample_np.max() > 1.0 or sample_np.min() < 0.0:
                        sample_np = (sample_np - sample_np.min()) / (sample_np.max() - sample_np.min())
                        sample = tf.constant(sample_np, dtype=tf.float32)
                    
                    yield [sample]
            
            # Configure converter for UINT8 quantization  
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            
            # Configure for Edge TPU compatibility
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # Enable experimental optimizations
            converter.experimental_new_converter = True
            
            # Convert model
            start_time = time.time()
            quantized_tflite = converter.convert()
            conversion_time = time.time() - start_time
            
            # Save quantized model
            if output_path is None:
                output_path = self._generate_temp_path('uint8_quantized.tflite')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(quantized_tflite)
            
            logger.info(f"UINT8 quantization completed in {conversion_time:.2f}s, "
                       f"output: {output_path}")
            
            # Validate the quantized model
            self._validate_quantized_model(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            error_msg = f"UINT8 quantization failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def quantize_float16(self,
                        model: Union[tf.keras.Model, str],
                        output_path: Optional[str] = None) -> str:
        """
        Apply Float16 quantization for reduced memory usage
        
        Args:
            model: Keras model or path to SavedModel
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized TFLite model
        """
        logger.info("Starting Float16 quantization")
        
        try:
            # Load model if string path provided
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Configure converter for Float16
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            # Convert model
            start_time = time.time()
            quantized_tflite = converter.convert()
            conversion_time = time.time() - start_time
            
            # Save quantized model
            if output_path is None:
                output_path = self._generate_temp_path('float16_quantized.tflite')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(quantized_tflite)
            
            logger.info(f"Float16 quantization completed in {conversion_time:.2f}s, "
                       f"output: {output_path}")
            
            # Validate the quantized model
            self._validate_quantized_model(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Float16 quantization failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def quantize_dynamic(self,
                        model: Union[tf.keras.Model, str],
                        output_path: Optional[str] = None) -> str:
        """
        Apply dynamic range quantization (weights only)
        
        Args:
            model: Keras model or path to SavedModel
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized TFLite model
        """
        logger.info("Starting dynamic range quantization")
        
        try:
            # Load model if string path provided
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Configure converter for dynamic quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert model
            start_time = time.time()
            quantized_tflite = converter.convert()
            conversion_time = time.time() - start_time
            
            # Save quantized model
            if output_path is None:
                output_path = self._generate_temp_path('dynamic_quantized.tflite')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(quantized_tflite)
            
            logger.info(f"Dynamic quantization completed in {conversion_time:.2f}s, "
                       f"output: {output_path}")
            
            # Validate the quantized model
            self._validate_quantized_model(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Dynamic quantization failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def quantize_hybrid(self,
                       model: Union[tf.keras.Model, str],
                       output_path: Optional[str] = None) -> str:
        """
        Apply hybrid quantization (int8 weights, float32 activations)
        
        Args:
            model: Keras model or path to SavedModel 
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized TFLite model
        """
        logger.info("Starting hybrid quantization")
        
        try:
            # Load model if string path provided
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Configure converter for hybrid quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            
            # Convert model
            start_time = time.time()
            quantized_tflite = converter.convert()
            conversion_time = time.time() - start_time
            
            # Save quantized model
            if output_path is None:
                output_path = self._generate_temp_path('hybrid_quantized.tflite')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(quantized_tflite)
            
            logger.info(f"Hybrid quantization completed in {conversion_time:.2f}s, "
                       f"output: {output_path}")
            
            # Validate the quantized model
            self._validate_quantized_model(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Hybrid quantization failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _generate_temp_path(self, filename: str) -> str:
        """Generate temporary file path"""
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            return str(self.temp_dir / filename)
        else:
            temp_dir = tempfile.mkdtemp()
            return str(Path(temp_dir) / filename)
    
    def _validate_quantized_model(self, model_path: str):
        """
        Validate that quantized model can be loaded and used
        
        Args:
            model_path: Path to quantized TFLite model
            
        Raises:
            RuntimeError: If model validation fails
        """
        try:
            # Try to load the model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            if not input_details or not output_details:
                raise RuntimeError("Model has no inputs or outputs")
            
            # Try a dummy inference
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']
            
            # Generate dummy input
            dummy_input = np.random.random(input_shape).astype(input_dtype)
            
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            
            # Get output to ensure inference works
            output = interpreter.get_tensor(output_details[0]['index'])
            
            if output is None:
                raise RuntimeError("Model inference returned None")
            
            logger.debug(f"Model validation successful: {model_path}")
            
        except Exception as e:
            error_msg = f"Quantized model validation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e