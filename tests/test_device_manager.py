"""
Tests for device manager functionality
"""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch

from edge_tpu_v6_bench.core.device_manager import (
    DeviceManager, DeviceInfo, DeviceType
)

class TestDeviceManager:
    """Test suite for DeviceManager"""
    
    def test_device_manager_initialization(self):
        """Test DeviceManager initializes correctly"""
        manager = DeviceManager()
        assert manager.devices == []
        assert manager.active_device is None
        assert manager._cache_ttl == 30
    
    def test_device_type_enum(self):
        """Test DeviceType enum values"""
        assert DeviceType.EDGE_TPU_V6.value == "edge_tpu_v6"
        assert DeviceType.EDGE_TPU_V5E.value == "edge_tpu_v5e"
        assert DeviceType.CORAL_DEV_BOARD.value == "coral_dev_board"
        assert DeviceType.USB_ACCELERATOR.value == "usb_accelerator"
        assert DeviceType.CPU_FALLBACK.value == "cpu_fallback"
    
    def test_device_info_creation(self):
        """Test DeviceInfo dataclass creation"""
        device_info = DeviceInfo(
            device_id="test_device",
            device_type=DeviceType.EDGE_TPU_V6,
            path="/dev/test",
            performance_class="ultra",
            power_limit_w=6.0,
            thermal_limit_c=90.0,
            memory_mb=16,
            compiler_version="3.0.0",
            runtime_version="3.0.0",
            supports_int4=True,
            supports_structured_sparsity=True
        )
        
        assert device_info.device_id == "test_device"
        assert device_info.device_type == DeviceType.EDGE_TPU_V6
        assert device_info.supports_int4 is True
        assert device_info.supports_structured_sparsity is True
    
    @patch('edge_tpu_v6_bench.core.device_manager.CORAL_AVAILABLE', False)
    def test_detect_devices_no_coral(self):
        """Test device detection when PyCoral is not available"""
        manager = DeviceManager()
        devices = manager.detect_devices()
        
        # Should have at least v6 preview and CPU fallback
        assert len(devices) >= 2
        
        device_types = [d.device_type for d in devices]
        assert DeviceType.EDGE_TPU_V6 in device_types
        assert DeviceType.CPU_FALLBACK in device_types
    
    @patch('edge_tpu_v6_bench.core.device_manager.CORAL_AVAILABLE', True)
    @patch('edge_tpu_v6_bench.core.device_manager.edgetpu')
    def test_detect_devices_with_coral(self, mock_edgetpu):
        """Test device detection with PyCoral available"""
        # Mock PyCoral device detection
        mock_edgetpu.list_edge_tpus.return_value = [
            '/dev/apex_0',
            'usb:0'
        ]
        
        manager = DeviceManager()
        devices = manager.detect_devices()
        
        # Should detect mock devices plus v6 preview and CPU fallback
        assert len(devices) >= 4
        
        device_types = [d.device_type for d in devices]
        assert DeviceType.EDGE_TPU_V6 in device_types
        assert DeviceType.CPU_FALLBACK in device_types
    
    def test_select_device_auto(self):
        """Test automatic device selection"""
        manager = DeviceManager()
        devices = manager.detect_devices()
        
        # Auto selection should prefer v6 > v5e > dev_board > usb > cpu
        selected = manager.select_device('auto')
        
        assert selected is not None
        assert manager.active_device == selected
        
        # Should prefer v6 if available
        v6_devices = [d for d in devices if d.device_type == DeviceType.EDGE_TPU_V6]
        if v6_devices:
            assert selected.device_type == DeviceType.EDGE_TPU_V6
    
    def test_select_device_by_type(self):
        """Test device selection by type"""
        manager = DeviceManager()
        manager.detect_devices()
        
        # Select CPU fallback specifically
        selected = manager.select_device(DeviceType.CPU_FALLBACK)
        
        assert selected.device_type == DeviceType.CPU_FALLBACK
        assert manager.active_device == selected
    
    def test_select_device_invalid(self):
        """Test selection of invalid device"""
        manager = DeviceManager()
        manager.detect_devices()
        
        with pytest.raises(RuntimeError, match="Device .* not found"):
            manager.select_device('invalid_device')
    
    def test_get_device_capabilities(self):
        """Test getting device capabilities"""
        manager = DeviceManager()
        manager.detect_devices()
        
        # Select a device and get capabilities
        device = manager.select_device('auto')
        capabilities = manager.get_device_capabilities()
        
        assert 'device_type' in capabilities
        assert 'quantization_support' in capabilities
        assert 'features' in capabilities
        assert 'limits' in capabilities
        assert 'versions' in capabilities
        
        # Check quantization support structure
        quant_support = capabilities['quantization_support']
        assert 'int8' in quant_support
        assert 'int4' in quant_support
        assert isinstance(quant_support['int8'], bool)
    
    @patch('tensorflow.lite.Interpreter')
    def test_create_interpreter_cpu(self, mock_interpreter):
        """Test TFLite interpreter creation for CPU"""
        manager = DeviceManager()
        manager.select_device(DeviceType.CPU_FALLBACK)
        
        # Mock interpreter
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        
        interpreter = manager.create_interpreter('/path/to/model.tflite')
        
        assert interpreter == mock_interp
        mock_interpreter.assert_called_once_with(model_path='/path/to/model.tflite')
        mock_interp.allocate_tensors.assert_called_once()
    
    def test_list_devices(self):
        """Test device listing functionality"""
        manager = DeviceManager()
        manager.detect_devices()
        
        device_list = manager.list_devices()
        
        assert isinstance(device_list, list)
        assert len(device_list) > 0
        
        for device in device_list:
            assert 'id' in device
            assert 'type' in device
            assert 'path' in device
            assert 'performance' in device
            assert 'features' in device
            
            # Check features structure
            features = device['features']
            assert 'int4' in features
            assert 'structured_sparsity' in features
            assert 'dynamic_shapes' in features
    
    def test_cache_behavior(self):
        """Test device detection caching"""
        manager = DeviceManager()
        
        # First call should populate cache
        devices1 = manager.detect_devices()
        cache_time1 = manager._detection_cache_time
        
        # Second call should use cache
        devices2 = manager.detect_devices()
        cache_time2 = manager._detection_cache_time
        
        assert devices1 == devices2
        assert cache_time1 == cache_time2
        
        # Force refresh should update cache
        devices3 = manager.detect_devices(force_refresh=True)
        cache_time3 = manager._detection_cache_time
        
        assert cache_time3 > cache_time2
    
    def test_classify_device(self):
        """Test device path classification"""
        manager = DeviceManager()
        
        # Test USB device classification
        usb_type = manager._classify_device('/dev/ttyUSB0')
        assert usb_type == DeviceType.USB_ACCELERATOR
        
        # Test PCI device classification
        pci_type = manager._classify_device('/dev/apex_0')
        assert pci_type == DeviceType.CORAL_DEV_BOARD
        
        # Test default classification
        default_type = manager._classify_device('/dev/unknown')
        assert default_type == DeviceType.EDGE_TPU_V5E
    
    def test_simulate_v6_devices(self):
        """Test v6 device simulation"""
        manager = DeviceManager()
        
        v6_devices = manager._simulate_v6_devices()
        
        assert len(v6_devices) == 1
        
        v6_device = v6_devices[0]
        assert v6_device.device_type == DeviceType.EDGE_TPU_V6
        assert v6_device.supports_int4 is True
        assert v6_device.supports_structured_sparsity is True
        assert v6_device.supports_dynamic_shapes is True
        assert v6_device.power_limit_w == 6.0
        assert v6_device.memory_mb == 16
    
    def test_cpu_fallback_creation(self):
        """Test CPU fallback device creation"""
        manager = DeviceManager()
        
        cpu_device = manager._create_cpu_fallback()
        
        assert cpu_device.device_type == DeviceType.CPU_FALLBACK
        assert cpu_device.device_id == "cpu_fallback"
        assert cpu_device.supports_dynamic_shapes is True
        assert cpu_device.supports_int4 is False
        assert cpu_device.power_limit_w == 15.0
    
    def test_get_active_device(self):
        """Test getting active device"""
        manager = DeviceManager()
        
        # Initially no active device
        assert manager.get_active_device() is None
        
        # After selection, should return active device
        manager.detect_devices()
        selected = manager.select_device('auto')
        
        active = manager.get_active_device()
        assert active == selected
    
    def test_no_devices_error(self):
        """Test error when no devices available"""
        manager = DeviceManager()
        # Don't call detect_devices()
        
        with pytest.raises(RuntimeError, match="No Edge TPU devices detected"):
            manager.select_device('auto')