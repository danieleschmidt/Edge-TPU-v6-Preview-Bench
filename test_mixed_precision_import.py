#!/usr/bin/env python3
"""
Specific test for MixedPrecisionOptimizer import to identify scipy dependency
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Mock dependencies step by step
class MockNumpy:
    def __getattr__(self, name):
        if name == 'percentile':
            # This might be the scipy dependency - numpy.percentile is sometimes linked to scipy
            return lambda x, p: sorted(x)[int(len(x) * p / 100)] if x else 0
        elif name == 'ndarray':
            return list
        elif name == 'mean':
            return lambda x: sum(x) / len(x) if x else 0
        elif name == 'var':
            return lambda x: 0.5  # Mock variance
        elif name == 'abs':
            return abs
        elif name == 'array':
            return list
        elif name == 'argmax':
            return lambda x, **kwargs: [0] * len(x) if hasattr(x, '__len__') else 0
        elif name == 'argsort':
            return lambda x, **kwargs: list(range(len(x))) if hasattr(x, '__len__') else []
        elif name == 'sum':
            return sum
        elif name == 'newaxis':
            return None
        else:
            return lambda *args, **kwargs: None

class MockTensorFlow:
    class keras:
        class Model:
            def __init__(self, *args, **kwargs):
                self.layers = []
            def count_params(self):
                return 1000000
            def summary(self):
                return "Mock model summary"
        
        class models:
            @staticmethod
            def load_model(path):
                return MockTensorFlow.keras.Model()
        
        class layers:
            @staticmethod
            def Input(*args, **kwargs):
                return None
    
    class data:
        class Dataset:
            @staticmethod
            def from_tensor_slices(data):
                return MockTensorFlow.data.Dataset()
            
            def batch(self, size):
                return self
            
            def take(self, num):
                return self
                
            def __iter__(self):
                return iter([[1, 2, 3]])  # Mock sample
    
    class lite:
        class TFLiteConverter:
            @staticmethod
            def from_keras_model(model):
                converter = MockTensorFlow.lite.TFLiteConverter()
                return converter
            
            def __init__(self):
                self.optimizations = []
                self.representative_dataset = None
                self.target_spec = MockTargetSpec()
                self.inference_input_type = None
                self.inference_output_type = None
                self.experimental_new_converter = True
            
            def convert(self):
                return b'fake_tflite_model_data'
        
        class Optimize:
            DEFAULT = 'default'
        
        class OpsSet:
            TFLITE_BUILTINS_INT8 = 'tflite_builtins_int8'
        
        class Interpreter:
            def __init__(self, model_path=None, model_content=None):
                self.model_path = model_path
                self.model_content = model_content
            
            def allocate_tensors(self):
                pass
            
            def get_input_details(self):
                return [{'index': 0, 'shape': [1, 224, 224, 3]}]
            
            def get_output_details(self):
                return [{'index': 0, 'shape': [1, 1000]}]
            
            def set_tensor(self, index, data):
                pass
            
            def invoke(self):
                pass
            
            def get_tensor(self, index):
                return [[0.1] * 1000]  # Mock predictions
    
    int8 = 'int8'
    
    @staticmethod
    def GradientTape():
        return MockGradientTape()
    
    @staticmethod 
    def reduce_mean(x, **kwargs):
        return 0.5
    
    class nn:
        @staticmethod
        def softmax_cross_entropy_with_logits(**kwargs):
            return 0.1
        
        @staticmethod
        def softmax(x):
            return x

class MockGradientTape:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def gradient(self, loss, variables):
        return [None] * len(variables)

class MockTargetSpec:
    def __init__(self):
        self.supported_ops = []

# Install mocks
sys.modules['numpy'] = MockNumpy()
sys.modules['tensorflow'] = MockTensorFlow()

print("Testing MixedPrecisionOptimizer import specifically...")

try:
    from edge_tpu_v6_bench.quantization.strategies.mixed_precision import MixedPrecisionOptimizer
    print("✓ MixedPrecisionOptimizer imported successfully!")
    
    # Test instantiation
    optimizer = MixedPrecisionOptimizer()
    print("✓ MixedPrecisionOptimizer instantiated successfully!")
    
    # Test method exists
    if hasattr(optimizer, 'quantize_int4_mixed'):
        print("✓ quantize_int4_mixed method exists")
    else:
        print("✗ quantize_int4_mixed method missing")
        
except ImportError as e:
    print(f"✗ ImportError: {e}")
    # Try to identify the specific missing module
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()