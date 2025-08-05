"""
Power monitoring and measurement utilities for Edge TPU devices
Provides real-time power consumption tracking and energy efficiency analysis
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from queue import Queue, Empty
import statistics

# Optional power measurement dependencies
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PowerSample:
    """Single power measurement sample"""
    timestamp: float
    voltage_v: float
    current_a: float
    power_w: float
    temperature_c: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PowerMeasurementInterface:
    """Abstract base class for power measurement interfaces"""
    
    def connect(self) -> bool:
        """Connect to power measurement device"""
        raise NotImplementedError
    
    def disconnect(self):
        """Disconnect from power measurement device"""
        raise NotImplementedError
    
    def read_sample(self) -> Optional[PowerSample]:
        """Read a single power measurement sample"""
        raise NotImplementedError
    
    def is_connected(self) -> bool:
        """Check if measurement device is connected"""
        raise NotImplementedError

class INA260Interface(PowerMeasurementInterface):
    """Interface for INA260 I2C power monitor"""
    
    def __init__(self, i2c_bus: int = 1, i2c_address: int = 0x40):
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.bus = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to INA260 via I2C"""
        if not SMBUS_AVAILABLE:
            logger.error("smbus2 not available for I2C communication")
            return False
            
        try:
            self.bus = smbus2.SMBus(self.i2c_bus)
            # Test connection by reading device ID
            device_id = self.bus.read_word_data(self.i2c_address, 0xFE)
            if device_id == 0x2270:  # INA260 device ID
                self.connected = True
                logger.info(f"Connected to INA260 at bus {self.i2c_bus}, address 0x{self.i2c_address:02x}")
                return True
            else:
                logger.error(f"Unexpected device ID: 0x{device_id:04x}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to INA260: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from INA260"""
        if self.bus:
            self.bus.close()
            self.bus = None
        self.connected = False
    
    def read_sample(self) -> Optional[PowerSample]:
        """Read power measurement from INA260"""
        if not self.connected or not self.bus:
            return None
            
        try:
            # Read voltage (register 0x02)
            voltage_raw = self.bus.read_word_data(self.i2c_address, 0x02)
            voltage_v = (voltage_raw * 1.25) / 1000.0  # LSB = 1.25mV
            
            # Read current (register 0x01)  
            current_raw = self.bus.read_word_data(self.i2c_address, 0x01)
            if current_raw > 32767:  # Handle signed 16-bit
                current_raw -= 65536
            current_a = (current_raw * 1.25) / 1000.0  # LSB = 1.25mA
            
            # Read power (register 0x03)
            power_raw = self.bus.read_word_data(self.i2c_address, 0x03)
            power_w = (power_raw * 10.0) / 1000.0  # LSB = 10mW
            
            return PowerSample(
                timestamp=time.time(),
                voltage_v=voltage_v,
                current_a=current_a,
                power_w=power_w
            )
            
        except Exception as e:
            logger.error(f"Failed to read from INA260: {e}")
            return None
    
    def is_connected(self) -> bool:
        return self.connected

class SerialPowerInterface(PowerMeasurementInterface):
    """Interface for serial-based power measurement devices"""
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to serial power measurement device"""
        if not SERIAL_AVAILABLE:
            logger.error("pyserial not available for serial communication")
            return False
            
        try:
            self.serial_conn = serial.Serial(
                self.port, 
                self.baudrate, 
                timeout=1.0
            )
            self.connected = True
            logger.info(f"Connected to serial power monitor at {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to serial device: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial device"""
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        self.connected = False
    
    def read_sample(self) -> Optional[PowerSample]:
        """Read power measurement from serial device"""
        if not self.connected or not self.serial_conn:
            return None
            
        try:
            # Read line from serial device
            line = self.serial_conn.readline().decode('utf-8').strip()
            
            # Parse comma-separated values: voltage,current,power
            parts = line.split(',')
            if len(parts) >= 3:
                voltage_v = float(parts[0])
                current_a = float(parts[1])
                power_w = float(parts[2])
                
                return PowerSample(
                    timestamp=time.time(),
                    voltage_v=voltage_v,
                    current_a=current_a,
                    power_w=power_w
                )
        except Exception as e:
            logger.error(f"Failed to read from serial device: {e}")
            
        return None
    
    def is_connected(self) -> bool:
        return self.connected

class SimulatedPowerInterface(PowerMeasurementInterface):
    """Simulated power interface for testing and development"""
    
    def __init__(self, base_power_w: float = 2.0, variation_w: float = 0.5):
        self.base_power_w = base_power_w
        self.variation_w = variation_w
        self.connected = False
        
    def connect(self) -> bool:
        """Simulate connection"""
        self.connected = True
        logger.info("Connected to simulated power monitor")
        return True
    
    def disconnect(self):
        """Simulate disconnection"""
        self.connected = False
    
    def read_sample(self) -> Optional[PowerSample]:
        """Generate simulated power measurement"""
        if not self.connected:
            return None
            
        # Generate realistic power values with some variation
        noise = np.random.normal(0, self.variation_w * 0.1)
        power_w = max(0.1, self.base_power_w + noise)
        
        # Simulate voltage and current based on power
        voltage_v = 5.0 + np.random.normal(0, 0.1)  # ~5V supply
        current_a = power_w / voltage_v
        
        return PowerSample(
            timestamp=time.time(),
            voltage_v=voltage_v,
            current_a=current_a,
            power_w=power_w
        )
    
    def is_connected(self) -> bool:
        return self.connected

class PowerMonitor:
    """
    Real-time power monitoring for Edge TPU devices
    
    Supports multiple measurement interfaces and provides
    continuous power consumption tracking during benchmarks
    """
    
    def __init__(self, 
                 interface: Optional[PowerMeasurementInterface] = None,
                 sample_rate_hz: float = 10.0):
        """
        Initialize power monitor
        
        Args:
            interface: Power measurement interface (auto-detect if None)
            sample_rate_hz: Sampling rate for power measurements
        """
        self.interface = interface or self._auto_detect_interface()
        self.sample_rate_hz = sample_rate_hz
        self.sample_interval_s = 1.0 / sample_rate_hz
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.sample_queue: Queue = Queue()
        self.samples: List[PowerSample] = []
        
    def _auto_detect_interface(self) -> PowerMeasurementInterface:
        """Auto-detect available power measurement interface"""
        
        # Try INA260 I2C interface first
        if SMBUS_AVAILABLE:
            ina260 = INA260Interface()
            if ina260.connect():
                logger.info("Auto-detected INA260 power monitor")
                return ina260
            ina260.disconnect()
        
        # Try common serial ports for power monitors
        if SERIAL_AVAILABLE:
            common_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', 'COM3', 'COM4']
            for port in common_ports:
                try:
                    serial_interface = SerialPowerInterface(port)
                    if serial_interface.connect():
                        logger.info(f"Auto-detected serial power monitor at {port}")
                        return serial_interface
                    serial_interface.disconnect()
                except:
                    continue
        
        # Fall back to simulated interface
        logger.info("No hardware power monitor detected, using simulation")
        return SimulatedPowerInterface()
    
    def start_monitoring(self) -> bool:
        """
        Start continuous power monitoring
        
        Returns:
            True if monitoring started successfully
        """
        if self.monitoring:
            logger.warning("Power monitoring already active")
            return True
            
        if not self.interface.connect():
            logger.error("Failed to connect to power measurement interface")
            return False
        
        self.monitoring = True
        self.samples.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Started power monitoring at {self.sample_rate_hz} Hz")
        return True
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop power monitoring and return collected data
        
        Returns:
            Dictionary containing power measurement data and statistics
        """
        if not self.monitoring:
            logger.warning("Power monitoring not active")
            return {}
        
        self.monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        # Collect any remaining samples from queue
        self._collect_queued_samples()
        
        # Disconnect from interface
        self.interface.disconnect()
        
        # Process collected samples
        power_data = self._process_samples()
        
        logger.info(f"Stopped power monitoring, collected {len(self.samples)} samples")
        return power_data
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread"""
        logger.debug("Power monitoring loop started")
        
        while self.monitoring:
            start_time = time.time()
            
            # Read sample from interface
            sample = self.interface.read_sample()
            if sample:
                self.sample_queue.put(sample)
            
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.sample_interval_s - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.debug("Power monitoring loop stopped")
    
    def _collect_queued_samples(self):
        """Collect all samples from the queue"""
        while True:
            try:
                sample = self.sample_queue.get_nowait()
                self.samples.append(sample)
            except Empty:
                break
    
    def _process_samples(self) -> Dict[str, Any]:
        """Process collected samples into summary statistics"""
        if not self.samples:
            return {}
        
        # Extract measurement arrays
        timestamps = [s.timestamp for s in self.samples]
        voltages = [s.voltage_v for s in self.samples]
        currents = [s.current_a for s in self.samples]
        powers = [s.power_w for s in self.samples]
        
        # Calculate duration
        duration_s = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        
        # Calculate statistics
        power_data = {
            'num_samples': len(self.samples),
            'duration_s': duration_s,
            'sample_rate_hz': len(self.samples) / duration_s if duration_s > 0 else 0.0,
            
            # Raw data
            'timestamps': timestamps,
            'voltage_samples': voltages,
            'current_samples': currents, 
            'power_samples': powers,
            
            # Voltage statistics
            'voltage_mean_v': statistics.mean(voltages),
            'voltage_min_v': min(voltages),
            'voltage_max_v': max(voltages),
            'voltage_std_v': statistics.stdev(voltages) if len(voltages) > 1 else 0.0,
            
            # Current statistics
            'current_mean_a': statistics.mean(currents),
            'current_min_a': min(currents),
            'current_max_a': max(currents),
            'current_std_a': statistics.stdev(currents) if len(currents) > 1 else 0.0,
            
            # Power statistics
            'power_mean_w': statistics.mean(powers),
            'power_min_w': min(powers),
            'power_max_w': max(powers),
            'power_std_w': statistics.stdev(powers) if len(powers) > 1 else 0.0,
            'power_p95_w': np.percentile(powers, 95),
            'power_p99_w': np.percentile(powers, 99),
            
            # Energy calculation
            'energy_total_j': statistics.mean(powers) * duration_s,
        }
        
        return power_data
    
    def get_current_power(self) -> Optional[float]:
        """Get current instantaneous power reading"""
        if not self.interface.is_connected():
            return None
            
        sample = self.interface.read_sample()
        return sample.power_w if sample else None
    
    def is_monitoring(self) -> bool:
        """Check if power monitoring is currently active"""
        return self.monitoring
    
    def get_interface_info(self) -> Dict[str, Any]:
        """Get information about the power measurement interface"""
        return {
            'interface_type': type(self.interface).__name__,
            'connected': self.interface.is_connected(),
            'sample_rate_hz': self.sample_rate_hz,
        }