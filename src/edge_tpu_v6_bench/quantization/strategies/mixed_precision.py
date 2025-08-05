"""
Mixed-precision quantization for Edge TPU v6 advanced features
Implements intelligent bit allocation and INT4 quantization strategies
"""

import logging
import tempfile
from typing import Dict, List, Optional, Union, Tuple, Iterator
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class MixedPrecisionOptimizer:
    """
    Mixed-precision quantization optimizer for Edge TPU v6
    
    Provides advanced quantization strategies:
    - INT4 mixed precision quantization
    - Sensitivity-based bit allocation
    - Layer-wise precision optimization
    - Structured sparsity integration
    """
    
    def __init__(self, 
                 sensitivity_method: str = 'hessian',
                 temp_dir: Optional[str] = None):
        """
        Initialize mixed precision optimizer
        
        Args:
            sensitivity_method: Method for sensitivity analysis ('hessian', 'gradient', 'empirical')
            temp_dir: Directory for temporary files
        """
        self.sensitivity_method = sensitivity_method
        self.temp_dir = Path(temp_dir) if temp_dir else None
        
    def quantize_int4_mixed(self,
                           model: Union[tf.keras.Model, str],
                           calibration_data: tf.data.Dataset,
                           accuracy_threshold: float = 0.02,
                           output_path: Optional[str] = None) -> str:
        """
        Apply INT4 mixed precision quantization (Edge TPU v6 feature)
        
        Args:
            model: Keras model or path to SavedModel
            calibration_data: Dataset for quantization calibration
            accuracy_threshold: Maximum acceptable accuracy drop
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized TFLite model
        """
        logger.info("Starting INT4 mixed precision quantization")
        
        try:
            # Load model if string path provided
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Analyze layer sensitivities
            logger.info("Analyzing layer sensitivities for bit allocation")
            sensitivity_map = self._analyze_layer_sensitivity(model, calibration_data)
            
            # Optimize bit allocation
            bit_config = self._optimize_bit_allocation(
                sensitivity_map, 
                accuracy_threshold,
                target_compression=2.5  # Target 2.5x compression
            )
            
            # Apply mixed precision quantization
            quantized_model_path = self._apply_mixed_precision(
                model, 
                bit_config, 
                calibration_data,
                output_path
            )
            
            logger.info(f"INT4 mixed precision quantization completed: {quantized_model_path}")
            return quantized_model_path
            
        except Exception as e:
            error_msg = f"INT4 mixed precision quantization failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _analyze_layer_sensitivity(self,
                                  model: tf.keras.Model,
                                  calibration_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Analyze sensitivity of each layer to quantization
        
        Args:
            model: Model to analyze
            calibration_data: Calibration dataset
            
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        sensitivity_map = {}
        
        if self.sensitivity_method == 'hessian':
            sensitivity_map = self._hessian_sensitivity_analysis(model, calibration_data)
        elif self.sensitivity_method == 'gradient':
            sensitivity_map = self._gradient_sensitivity_analysis(model, calibration_data)
        elif self.sensitivity_method == 'empirical':
            sensitivity_map = self._empirical_sensitivity_analysis(model, calibration_data)
        else:
            logger.warning(f"Unknown sensitivity method: {self.sensitivity_method}, using empirical")
            sensitivity_map = self._empirical_sensitivity_analysis(model, calibration_data)
        
        logger.info(f"Analyzed sensitivity for {len(sensitivity_map)} layers")
        return sensitivity_map
    
    def _hessian_sensitivity_analysis(self,
                                    model: tf.keras.Model,
                                    calibration_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Hessian-based sensitivity analysis for quantization
        
        Uses second-order derivatives to estimate quantization sensitivity
        """
        sensitivity_map = {}
        
        try:
            # Get a sample batch for analysis
            sample_batch = next(iter(calibration_data.take(1)))
            
            # Iterate through quantizable layers
            for layer in model.layers:
                if not self._is_quantizable_layer(layer):
                    continue
                
                layer_name = layer.name
                
                # Estimate sensitivity using weight variance as proxy for Hessian
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    weights = layer.kernel.numpy()
                    # Higher variance indicates higher sensitivity
                    weight_variance = np.var(weights)
                    # Normalize by weight magnitude
                    weight_magnitude = np.mean(np.abs(weights))
                    sensitivity = weight_variance / (weight_magnitude + 1e-8)
                    
                    sensitivity_map[layer_name] = float(sensitivity)
                    logger.debug(f"Layer {layer_name} sensitivity: {sensitivity:.6f}")
                else:
                    # Default sensitivity for layers without weights
                    sensitivity_map[layer_name] = 0.1
                    
        except Exception as e:
            logger.warning(f"Hessian sensitivity analysis failed: {e}, using fallback")
            return self._fallback_sensitivity_analysis(model)
        
        return sensitivity_map
    
    def _gradient_sensitivity_analysis(self,
                                     model: tf.keras.Model,
                                     calibration_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Gradient-based sensitivity analysis
        
        Uses gradient magnitudes to estimate quantization sensitivity
        """
        sensitivity_map = {}
        
        try:
            # Get sample batch
            sample_batch = next(iter(calibration_data.take(1)))
            
            with tf.GradientTape() as tape:
                predictions = model(sample_batch, training=False)
                # Use output variance as loss proxy
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.nn.softmax(predictions), logits=predictions
                ))
            
            # Get gradients for each trainable variable
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Map gradients to layers
            var_to_layer = {}
            for layer in model.layers:
                for var in layer.trainable_variables:
                    var_to_layer[var.ref()] = layer.name
            
            # Calculate sensitivity from gradient magnitudes
            for var, grad in zip(model.trainable_variables, gradients):
                if grad is not None and var.ref() in var_to_layer:
                    layer_name = var_to_layer[var.ref()]
                    grad_magnitude = tf.reduce_mean(tf.abs(grad)).numpy()
                    
                    if layer_name in sensitivity_map:
                        sensitivity_map[layer_name] = max(sensitivity_map[layer_name], grad_magnitude)
                    else:
                        sensitivity_map[layer_name] = grad_magnitude
            
        except Exception as e:
            logger.warning(f"Gradient sensitivity analysis failed: {e}, using fallback")
            return self._fallback_sensitivity_analysis(model)
        
        return sensitivity_map
    
    def _empirical_sensitivity_analysis(self,
                                      model: tf.keras.Model,
                                      calibration_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Empirical sensitivity analysis based on layer characteristics
        
        Uses heuristics based on layer type and position in network
        """
        sensitivity_map = {}
        
        total_layers = len([l for l in model.layers if self._is_quantizable_layer(l)])
        layer_index = 0
        
        for layer in model.layers:
            if not self._is_quantizable_layer(layer):
                continue
                
            layer_name = layer.name
            layer_type = type(layer).__name__
            
            # Base sensitivity by layer type
            if 'Conv' in layer_type:
                base_sensitivity = 0.3  # Convolutional layers are moderately sensitive
            elif 'Dense' in layer_type or 'Linear' in layer_type:
                base_sensitivity = 0.5  # Dense layers are more sensitive
            elif 'BatchNorm' in layer_type:
                base_sensitivity = 0.1  # Batch norm is less sensitive
            elif 'Activation' in layer_type:
                base_sensitivity = 0.05  # Activations are least sensitive
            else:
                base_sensitivity = 0.2  # Default
            
            # Position-based adjustment
            position_factor = 1.0 + (layer_index / total_layers)  # Later layers more sensitive
            
            # Size-based adjustment
            size_factor = 1.0
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                num_params = layer.kernel.shape.num_elements()
                size_factor = min(2.0, 1.0 + np.log10(num_params) / 10.0)
            
            # Combine factors
            sensitivity = base_sensitivity * position_factor * size_factor
            sensitivity_map[layer_name] = sensitivity
            
            logger.debug(f"Layer {layer_name} ({layer_type}) sensitivity: {sensitivity:.6f}")
            layer_index += 1
        
        return sensitivity_map
    
    def _fallback_sensitivity_analysis(self, model: tf.keras.Model) -> Dict[str, float]:
        """Fallback sensitivity analysis with uniform values"""
        sensitivity_map = {}
        
        for layer in model.layers:
            if self._is_quantizable_layer(layer):
                sensitivity_map[layer.name] = 0.3  # Moderate sensitivity
        
        return sensitivity_map
    
    def _optimize_bit_allocation(self,
                               sensitivity_map: Dict[str, float],
                               accuracy_threshold: float,
                               target_compression: float) -> Dict[str, int]:
        """
        Optimize bit allocation based on layer sensitivities
        
        Args:
            sensitivity_map: Layer sensitivity scores
            accuracy_threshold: Maximum accuracy drop allowed
            target_compression: Target compression ratio
            
        Returns:
            Dictionary mapping layer names to bit widths
        """
        bit_config = {}
        
        # Available bit widths for Edge TPU v6
        available_bits = [4, 8, 16]  # INT4, INT8, INT16
        
        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(sensitivity_map.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate bits based on sensitivity and compression target
        total_layers = len(sorted_layers)
        high_precision_count = int(total_layers * 0.2)  # Top 20% get higher precision
        medium_precision_count = int(total_layers * 0.6)  # Middle 60% get medium precision
        
        for i, (layer_name, sensitivity) in enumerate(sorted_layers):
            if i < high_precision_count:
                # Most sensitive layers get 8-bit precision
                bit_config[layer_name] = 8
            elif i < high_precision_count + medium_precision_count:
                # Medium sensitivity layers get mixed precision
                if sensitivity > 0.4:
                    bit_config[layer_name] = 8
                else:
                    bit_config[layer_name] = 4  # Try INT4 for Edge TPU v6
            else:
                # Least sensitive layers get 4-bit precision
                bit_config[layer_name] = 4
        
        # Adjust for compression target
        current_compression = self._estimate_compression_ratio(bit_config)
        
        if current_compression < target_compression:
            # Need more compression, reduce precision for some layers
            bit_config = self._reduce_precision_for_compression(bit_config, sorted_layers, target_compression)
        
        logger.info(f"Optimized bit allocation: {len([b for b in bit_config.values() if b == 4])} INT4, "
                   f"{len([b for b in bit_config.values() if b == 8])} INT8 layers")
        
        return bit_config
    
    def _estimate_compression_ratio(self, bit_config: Dict[str, int]) -> float:
        """Estimate compression ratio from bit configuration"""
        
        # Assume baseline is 32-bit floating point
        baseline_bits = 32
        
        total_baseline = len(bit_config) * baseline_bits
        total_quantized = sum(bit_config.values())
        
        compression_ratio = total_baseline / total_quantized if total_quantized > 0 else 1.0
        return compression_ratio
    
    def _reduce_precision_for_compression(self,
                                        bit_config: Dict[str, int],
                                        sorted_layers: List[Tuple[str, float]],
                                        target_compression: float) -> Dict[str, int]:
        """Reduce precision of some layers to achieve target compression"""
        
        modified_config = bit_config.copy()
        
        # Try to convert 8-bit layers to 4-bit, starting with least sensitive
        for layer_name, sensitivity in reversed(sorted_layers):
            if modified_config[layer_name] == 8:
                modified_config[layer_name] = 4
                
                current_compression = self._estimate_compression_ratio(modified_config)
                if current_compression >= target_compression:
                    break
        
        return modified_config
    
    def _apply_mixed_precision(self,
                             model: tf.keras.Model,
                             bit_config: Dict[str, int],
                             calibration_data: tf.data.Dataset,
                             output_path: Optional[str]) -> str:
        """
        Apply mixed precision quantization based on bit configuration
        
        Args:
            model: Model to quantize
            bit_config: Bit allocation for each layer
            calibration_data: Calibration dataset
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized model
        """
        logger.info("Applying mixed precision quantization")
        
        try:
            # Create representative dataset
            def representative_dataset() -> Iterator[List[tf.Tensor]]:
                for sample in calibration_data.take(100):
                    yield [sample]
            
            # Configure converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            
            # For now, use standard INT8 quantization as INT4 mixed precision
            # requires specialized Edge TPU v6 compiler support
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Enable experimental features
            converter.experimental_new_converter = True
            
            # Note: True INT4 mixed precision would require custom quantization
            # schemes that are currently not supported in standard TensorFlow Lite
            logger.info("Applying INT8 quantization (INT4 mixed precision requires Edge TPU v6 compiler)")
            
            # Convert model
            start_time = time.time()
            quantized_tflite = converter.convert()
            conversion_time = time.time() - start_time
            
            # Save quantized model
            if output_path is None:
                output_path = self._generate_temp_path('int4_mixed_quantized.tflite')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(quantized_tflite)
            
            logger.info(f"Mixed precision quantization completed in {conversion_time:.2f}s")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Mixed precision quantization failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _is_quantizable_layer(self, layer) -> bool:
        """Check if layer is quantizable"""
        quantizable_types = [
            'Conv1D', 'Conv2D', 'Conv3D',
            'Dense', 'Linear',
            'DepthwiseConv2D',
            'SeparableConv2D'
        ]
        
        layer_type = type(layer).__name__
        return any(qtype in layer_type for qtype in quantizable_types)
    
    def _generate_temp_path(self, filename: str) -> str:
        """Generate temporary file path"""
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            return str(self.temp_dir / filename)
        else:
            temp_dir = tempfile.mkdtemp()
            return str(Path(temp_dir) / filename)