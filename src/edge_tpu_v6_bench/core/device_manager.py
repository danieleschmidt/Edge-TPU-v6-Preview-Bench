"""
Device detection and management for Edge TPU devices
Handles automatic detection, enumeration, and configuration of available Edge TPU hardware
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

try:
    import pycoral
    from pycoral.utils import edgetpu
    from pycoral.utils.dataset import read_label_file
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False
    logging.warning("PyCoral not available - Edge TPU functionality limited")

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Supported Edge TPU device types"""
    EDGE_TPU_V6 = "edge_tpu_v6"
    EDGE_TPU_V5E = "edge_tpu_v5e" 
    CORAL_DEV_BOARD = "coral_dev_board"
    USB_ACCELERATOR = "usb_accelerator"
    CPU_FALLBACK = "cpu_fallback"
    UNKNOWN = "unknown"

@dataclass
class DeviceInfo:
    """Information about a detected Edge TPU device"""
    device_id: str
    device_type: DeviceType
    path: str
    performance_class: str
    power_limit_w: float
    thermal_limit_c: float
    memory_mb: int
    compiler_version: str
    runtime_version: str
    supports_int8: bool = True
    supports_int4: bool = False
    supports_structured_sparsity: bool = False
    supports_dynamic_shapes: bool = False

class DeviceManager:
    """
    Manages Edge TPU device detection, enumeration, and configuration
    
    Provides automatic device detection with fallback hierarchy:
    Edge TPU v6 → v5e → Coral Dev Board → USB Accelerator → CPU
    """
    
    def __init__(self):
        self.devices: List[DeviceInfo] = []
        self.active_device: Optional[DeviceInfo] = None
        self._detection_cache_time = 0
        self._cache_ttl = 30  # Cache device list for 30 seconds
        
    def detect_devices(self, force_refresh: bool = False) -> List[DeviceInfo]:
        """
        Detect all available Edge TPU devices
        
        Args:
            force_refresh: Force re-detection even if cache is valid
            
        Returns:
            List of detected device information
        """
        current_time = time.time()
        
        # Use cached results if recent and not forcing refresh
        if (not force_refresh and 
            self.devices and 
            current_time - self._detection_cache_time < self._cache_ttl):
            return self.devices
            
        logger.info("Detecting Edge TPU devices...")
        detected_devices = []
        
        if CORAL_AVAILABLE:
            detected_devices.extend(self._detect_coral_devices())
        
        # Add simulated v6 device for preview/testing
        detected_devices.extend(self._simulate_v6_devices())
        
        # Add CPU fallback
        detected_devices.append(self._create_cpu_fallback())
        
        self.devices = detected_devices
        self._detection_cache_time = current_time
        
        logger.info(f"Detected {len(self.devices)} devices: "
                   f"{[d.device_type.value for d in self.devices]}")
        
        return self.devices
    
    def _detect_coral_devices(self) -> List[DeviceInfo]:
        """Detect actual Coral/Edge TPU devices using PyCoral"""
        devices = []
        
        try:
            # Get list of Edge TPU devices
            edge_tpu_devices = edgetpu.list_edge_tpus()
            
            for i, device_path in enumerate(edge_tpu_devices):
                # Determine device type based on path/properties
                device_type = self._classify_device(device_path)
                
                device_info = DeviceInfo(
                    device_id=f"edgetpu_{i}",
                    device_type=device_type,
                    path=device_path,
                    performance_class="high" if "usb" not in device_path.lower() else "standard",
                    power_limit_w=4.0 if device_type == DeviceType.EDGE_TPU_V5E else 2.0,
                    thermal_limit_c=85.0,
                    memory_mb=8 if device_type == DeviceType.CORAL_DEV_BOARD else 4,
                    compiler_version="2.14.0",
                    runtime_version=pycoral.__version__ if hasattr(pycoral, '__version__') else "2.0.0",
                    supports_int4=False,  # v5e doesn't support INT4
                    supports_structured_sparsity=False,  # v5e doesn't support structured sparsity
                    supports_dynamic_shapes=False,
                )
                
                devices.append(device_info)
                logger.info(f"Detected {device_type.value} at {device_path}")
                
        except Exception as e:
            logger.warning(f"Error detecting Coral devices: {e}")
            
        return devices
    
    def _simulate_v6_devices(self) -> List[DeviceInfo]:
        """Simulate Edge TPU v6 devices for preview/development"""
        devices = []
        
        # Simulate v6 device with enhanced capabilities
        v6_device = DeviceInfo(
            device_id="edgetpu_v6_preview",
            device_type=DeviceType.EDGE_TPU_V6,
            path="/dev/apex_v6_preview",
            performance_class="ultra",
            power_limit_w=6.0,  # Higher power budget
            thermal_limit_c=90.0,  # Better thermal design
            memory_mb=16,  # More memory
            compiler_version="3.0.0-preview",
            runtime_version="3.0.0-preview",
            supports_int8=True,
            supports_int4=True,  # New v6 feature
            supports_structured_sparsity=True,  # New v6 feature
            supports_dynamic_shapes=True,  # New v6 feature
        )
        
        devices.append(v6_device)
        logger.info("Added simulated Edge TPU v6 device for preview testing")
        
        return devices
    
    def _create_cpu_fallback(self) -> DeviceInfo:
        """Create CPU fallback device info"""
        return DeviceInfo(
            device_id="cpu_fallback",
            device_type=DeviceType.CPU_FALLBACK,
            path="/dev/cpu",
            performance_class="baseline",
            power_limit_w=15.0,  # CPU power consumption
            thermal_limit_c=85.0,
            memory_mb=1024,  # System memory available
            compiler_version=tf.__version__,
            runtime_version=tf.__version__,
            supports_int8=True,
            supports_int4=False,
            supports_structured_sparsity=False,
            supports_dynamic_shapes=True,
        )
    
    def _classify_device(self, device_path: str) -> DeviceType:
        """Classify device type based on path and properties"""
        path_lower = device_path.lower()
        
        if "usb" in path_lower:
            return DeviceType.USB_ACCELERATOR
        elif "pci" in path_lower or "dev_board" in path_lower:
            return DeviceType.CORAL_DEV_BOARD
        else:
            # Default to v5e for existing Edge TPU devices
            return DeviceType.EDGE_TPU_V5E
    
    def select_device(self, device_spec: Union[str, DeviceType, int, None] = 'auto') -> DeviceInfo:
        """
        Select and configure a device for benchmarking
        
        Args:
            device_spec: Device specification:
                - 'auto': Automatically select best available device
                - DeviceType: Select specific device type
                - int: Select device by index
                - str: Select device by ID or type name
                
        Returns:
            Selected device information
            
        Raises:
            RuntimeError: If no suitable device found
        """
        if not self.devices:
            self.detect_devices()
            
        if not self.devices:
            raise RuntimeError("No Edge TPU devices detected")
        
        selected_device = None
        
        if device_spec == 'auto':
            # Auto-select best available device (v6 > v5e > dev_board > usb > cpu)
            priority_order = [
                DeviceType.EDGE_TPU_V6,
                DeviceType.EDGE_TPU_V5E,
                DeviceType.CORAL_DEV_BOARD,
                DeviceType.USB_ACCELERATOR,
                DeviceType.CPU_FALLBACK,
            ]
            
            for device_type in priority_order:
                for device in self.devices:
                    if device.device_type == device_type:
                        selected_device = device
                        break
                if selected_device:
                    break
                    
        elif isinstance(device_spec, DeviceType):
            # Select by device type
            for device in self.devices:
                if device.device_type == device_spec:
                    selected_device = device
                    break
                    
        elif isinstance(device_spec, int):
            # Select by index
            if 0 <= device_spec < len(self.devices):
                selected_device = self.devices[device_spec]
                
        elif isinstance(device_spec, str):
            # Select by ID or type name
            for device in self.devices:
                if (device.device_id == device_spec or 
                    device.device_type.value == device_spec):
                    selected_device = device
                    break
        
        if not selected_device:
            available = [d.device_type.value for d in self.devices]
            raise RuntimeError(f"Device '{device_spec}' not found. Available: {available}")
        
        self.active_device = selected_device
        logger.info(f"Selected device: {selected_device.device_type.value} "
                   f"({selected_device.device_id})")
        
        return selected_device
    
    def get_device_capabilities(self, device: Optional[DeviceInfo] = None) -> Dict[str, Any]:
        """
        Get detailed capabilities of a device
        
        Args:
            device: Device to query, or active device if None
            
        Returns:
            Dictionary of device capabilities
        """
        if device is None:
            device = self.active_device
            
        if device is None:
            raise RuntimeError("No device selected")
        
        return {
            'device_type': device.device_type.value,
            'performance_class': device.performance_class,
            'quantization_support': {
                'int8': device.supports_int8,
                'int4': device.supports_int4,
                'uint8': device.supports_int8,  # Usually same as int8
                'float16': device.device_type == DeviceType.CPU_FALLBACK,
            },
            'features': {
                'structured_sparsity': device.supports_structured_sparsity,
                'dynamic_shapes': device.supports_dynamic_shapes,
                'batch_inference': True,
                'concurrent_execution': device.device_type != DeviceType.CPU_FALLBACK,
            },
            'limits': {
                'power_w': device.power_limit_w,
                'thermal_c': device.thermal_limit_c,
                'memory_mb': device.memory_mb,
            },
            'versions': {
                'compiler': device.compiler_version,
                'runtime': device.runtime_version,
            }
        }
    
    def create_interpreter(self, model_path: str, device: Optional[DeviceInfo] = None) -> tf.lite.Interpreter:
        """
        Create TensorFlow Lite interpreter for the specified device
        
        Args:
            model_path: Path to TFLite model file
            device: Target device, or active device if None
            
        Returns:
            Configured TensorFlow Lite interpreter
        """
        if device is None:
            device = self.active_device
            
        if device is None:
            raise RuntimeError("No device selected")
        
        if device.device_type == DeviceType.CPU_FALLBACK:
            # CPU fallback
            interpreter = tf.lite.Interpreter(model_path=model_path)
        else:
            # Edge TPU device
            if not CORAL_AVAILABLE:
                logger.warning("PyCoral not available, falling back to CPU")
                interpreter = tf.lite.Interpreter(model_path=model_path)
            else:
                try:
                    interpreter = tf.lite.Interpreter(
                        model_path=model_path,
                        experimental_delegates=[
                            tf.lite.experimental.load_delegate('libedgetpu.so.1')
                        ]
                    )
                except Exception as e:
                    logger.warning(f"Failed to create Edge TPU interpreter: {e}")
                    logger.info("Falling back to CPU interpreter")
                    interpreter = tf.lite.Interpreter(model_path=model_path)
        
        interpreter.allocate_tensors()
        return interpreter
    
    def get_active_device(self) -> Optional[DeviceInfo]:
        """Get the currently active device"""
        return self.active_device
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        Get a formatted list of all detected devices
        
        Returns:
            List of device information dictionaries
        """
        if not self.devices:
            self.detect_devices()
            
        return [
            {
                'id': device.device_id,
                'type': device.device_type.value,
                'path': device.path,
                'performance': device.performance_class,
                'power_limit_w': device.power_limit_w,
                'memory_mb': device.memory_mb,
                'features': {
                    'int4': device.supports_int4,
                    'structured_sparsity': device.supports_structured_sparsity,
                    'dynamic_shapes': device.supports_dynamic_shapes,
                }
            }
            for device in self.devices
        ]