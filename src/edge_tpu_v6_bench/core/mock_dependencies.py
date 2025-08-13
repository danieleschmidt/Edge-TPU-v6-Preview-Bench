"""
Mock dependencies for development environment without external packages
Provides basic implementations to make the system functional
"""

import math
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Mock numpy
class MockNumPy:
    float32 = "float32"
    uint8 = "uint8"
    int8 = "int8"
    
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def zeros(shape, dtype=None):
        if isinstance(shape, tuple):
            size = 1
            for dim in shape:
                size *= dim
            return [0] * size
        return [0] * shape
    
    class random:
        @staticmethod
        def normal(mean, std, shape):
            import random as rnd
            if isinstance(shape, tuple):
                size = 1
                for dim in shape:
                    size *= dim
            else:
                size = shape
            return [rnd.gauss(mean, std) for _ in range(size)]
        
        @staticmethod
        def randint(low, high, shape, dtype=None):
            import random as rnd
            if isinstance(shape, tuple):
                size = 1
                for dim in shape:
                    size *= dim
            else:
                size = shape
            return [rnd.randint(low, high-1) for _ in range(size)]
    
    @staticmethod
    def repeat(arr, repeats, axis=None):
        return [arr] * repeats
    
    @staticmethod
    def percentile(data, p):
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)
    
    @staticmethod
    def mean(data):
        return sum(data) / len(data) if data else 0
    
    @staticmethod
    def argmax(data, axis=None):
        if isinstance(data[0], list):
            return [data[i].index(max(data[i])) for i in range(len(data))]
        return data.index(max(data))

# Mock TensorFlow
class MockTensorFlow:
    @staticmethod
    def lite():
        return MockTFLite()
    
    @staticmethod
    def keras():
        return MockKeras()
    
    @staticmethod
    def data():
        return MockTFData()

class MockTFLite:
    @staticmethod
    def Interpreter(model_path=None):
        return MockInterpreter()

class MockInterpreter:
    def __init__(self):
        self.inputs = [{'index': 0, 'shape': [1, 224, 224, 3], 'dtype': MockNumPy.float32}]
        self.outputs = [{'index': 0, 'shape': [1, 1000], 'dtype': MockNumPy.float32}]
        self.tensors = []
    
    def allocate_tensors(self):
        pass
    
    def get_input_details(self):
        return self.inputs
    
    def get_output_details(self):
        return self.outputs
    
    def get_tensor_details(self):
        return self.tensors
    
    def set_tensor(self, index, data):
        pass
    
    def invoke(self):
        time.sleep(0.001)  # Simulate inference time
    
    def get_tensor(self, index):
        return MockNumPy.zeros((1, 1000))

class MockKeras:
    @staticmethod
    def Model():
        return MockKerasModel()
    
    @staticmethod
    def applications():
        return MockKerasApplications()
    
    @staticmethod
    def models():
        return MockKerasModels()

class MockKerasModel:
    def __init__(self):
        self.input_shape = (None, 224, 224, 3)
        self.output_shape = (None, 1000)
        self.dtype = MockDType()
    
    def count_params(self):
        return 1000000
    
    def predict(self, data, verbose=0):
        batch_size = len(data) if isinstance(data, list) else 1
        return [[0.1] * 1000 for _ in range(batch_size)]
    
    def save(self, path, save_format='tf'):
        import os
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/saved_model.pb", 'w') as f:
            f.write("mock model data")

class MockDType:
    name = "float32"

class MockKerasApplications:
    @staticmethod
    def MobileNetV3Small():
        return MockKerasModel()
    
    @staticmethod
    def EfficientNetB0():
        return MockKerasModel()

class MockKerasModels:
    @staticmethod
    def load_model(path):
        return MockKerasModel()

class MockTFData:
    @staticmethod
    def Dataset():
        return MockDataset()

class MockDataset:
    def __init__(self, data=None):
        self.data = data or []
    
    @staticmethod
    def from_tensor_slices(data):
        return MockDataset(data)
    
    def batch(self, batch_size):
        return MockDataset(self.data)

# Initialize mock modules
np = MockNumPy()
tf = MockTensorFlow()

# Mock other dependencies
class MockPsutil:
    @staticmethod
    def cpu_percent():
        return 50.0
    
    @staticmethod
    def virtual_memory():
        class MemInfo:
            total = 8000000000
            available = 4000000000
            percent = 50.0
        return MemInfo()

psutil = MockPsutil()

# Export for use
__all__ = ['np', 'tf', 'psutil']