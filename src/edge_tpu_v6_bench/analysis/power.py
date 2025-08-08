"""
Power Analysis Module for Edge TPU v6
Comprehensive power measurement and efficiency analysis
"""

import time
import threading
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class WorkloadType(Enum):
    CONTINUOUS = "continuous"
    BURST = "burst"  
    PERIODIC = "periodic"
    IDLE = "idle"

@dataclass
class PowerSample:
    """Single power measurement sample"""
    timestamp: float
    power_w: float
    voltage_v: float
    current_a: float
    temperature_c: Optional[float] = None

@dataclass
class PowerTrace:
    """Collection of power measurements over time"""
    samples: List[PowerSample] = field(default_factory=list)
    workload_type: WorkloadType = WorkloadType.CONTINUOUS
    duration_s: float = 0.0
    sample_rate_hz: float = 1000.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PowerAnalysis:
    """Power analysis results"""
    avg_power_w: float
    peak_power_w: float
    min_power_w: float
    idle_power_w: float
    active_power_w: float
    power_std_w: float
    energy_total_j: float
    energy_per_inference_mj: float
    efficiency_ips_per_watt: float
    thermal_throttling_events: int
    power_efficiency_score: float

class PowerAnalyzer:
    """
    Advanced power analysis for Edge TPU devices
    
    Features:
    - Real-time power measurement
    - Workload-specific power profiling  
    - Energy efficiency analysis
    - Thermal throttling detection
    - Multi-device power comparison
    """
    
    def __init__(self, 
                 device: str = 'edge_tpu_v6',
                 measurement_interface: str = 'ina260',
                 sample_rate_hz: float = 1000.0):
        """
        Initialize power analyzer
        
        Args:
            device: Target device for power measurement
            measurement_interface: Power measurement hardware interface
            sample_rate_hz: Power sampling frequency
        """
        self.device = device
        self.measurement_interface = measurement_interface
        self.sample_rate_hz = sample_rate_hz
        
        # Power monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.power_samples: List[PowerSample] = []
        
        # Baseline measurements
        self.idle_power_w = 0.0
        self.baseline_measured = False
        
        logger.info(f"PowerAnalyzer initialized: {device} via {measurement_interface} @ {sample_rate_hz}Hz")
    
    def measure_baseline(self, duration_s: float = 10.0) -> float:
        """
        Measure idle power consumption baseline
        
        Args:
            duration_s: Measurement duration in seconds
            
        Returns:
            Idle power consumption in watts
        """
        logger.info(f"Measuring baseline power for {duration_s}s...")
        
        baseline_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration_s:
            sample = self._take_power_sample()
            baseline_samples.append(sample.power_w)
            time.sleep(1.0 / self.sample_rate_hz)
        
        self.idle_power_w = statistics.mean(baseline_samples)
        self.baseline_measured = True
        
        logger.info(f"Baseline power: {self.idle_power_w:.3f}W")
        return self.idle_power_w
    
    def measure(self, 
                model = None,
                duration_seconds: float = 60.0,
                workload: WorkloadType = WorkloadType.CONTINUOUS) -> PowerTrace:
        """
        Measure power consumption during model execution
        
        Args:
            model: Model to execute during measurement
            duration_seconds: Measurement duration
            workload: Type of workload pattern
            
        Returns:
            Power measurement trace
        """
        if not self.baseline_measured:
            self.measure_baseline()
        
        logger.info(f"Starting power measurement: {duration_seconds}s {workload.value}")
        
        # Start power monitoring
        self.power_samples = []
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._power_monitoring_loop)
        self.monitoring_thread.start()
        
        # Execute workload
        start_time = time.time()
        inference_count = 0
        
        try:
            if model:
                inference_count = self._execute_workload(model, duration_seconds, workload)
            else:
                # Just measure without workload
                time.sleep(duration_seconds)
                
        finally:
            # Stop monitoring
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join()
        
        actual_duration = time.time() - start_time
        
        power_trace = PowerTrace(
            samples=self.power_samples.copy(),
            workload_type=workload,
            duration_s=actual_duration,
            sample_rate_hz=self.sample_rate_hz,
            metadata={
                'device': self.device,
                'inference_count': inference_count,
                'model': str(model) if model else None
            }
        )
        
        logger.info(f"Power measurement complete: {len(self.power_samples)} samples, "
                   f"{inference_count} inferences")
        
        return power_trace
    
    def analyze(self, power_trace: PowerTrace) -> PowerAnalysis:
        """
        Analyze power trace and calculate metrics
        
        Args:
            power_trace: Power measurement data
            
        Returns:
            Comprehensive power analysis
        """
        if not power_trace.samples:
            raise ValueError("No power samples to analyze")
        
        power_values = [sample.power_w for sample in power_trace.samples]
        inference_count = power_trace.metadata.get('inference_count', 0)
        
        # Basic power statistics
        avg_power = statistics.mean(power_values)
        peak_power = max(power_values)
        min_power = min(power_values)
        power_std = statistics.stdev(power_values) if len(power_values) > 1 else 0.0
        
        # Active vs idle power
        active_power = avg_power - self.idle_power_w
        
        # Energy calculations
        energy_total_j = avg_power * power_trace.duration_s
        energy_per_inference_mj = 0.0
        if inference_count > 0:
            energy_per_inference_mj = (energy_total_j * 1000) / inference_count
        
        # Efficiency metrics
        efficiency_ips_per_watt = 0.0
        if avg_power > 0 and inference_count > 0:
            efficiency_ips_per_watt = (inference_count / power_trace.duration_s) / avg_power
        
        # Detect thermal throttling events (mock implementation)
        throttling_events = self._detect_thermal_throttling(power_trace)
        
        # Calculate power efficiency score
        efficiency_score = self._calculate_efficiency_score(
            avg_power, peak_power, efficiency_ips_per_watt, throttling_events
        )
        
        analysis = PowerAnalysis(
            avg_power_w=avg_power,
            peak_power_w=peak_power,
            min_power_w=min_power,
            idle_power_w=self.idle_power_w,
            active_power_w=active_power,
            power_std_w=power_std,
            energy_total_j=energy_total_j,
            energy_per_inference_mj=energy_per_inference_mj,
            efficiency_ips_per_watt=efficiency_ips_per_watt,
            thermal_throttling_events=throttling_events,
            power_efficiency_score=efficiency_score
        )
        
        logger.info(f"Power analysis complete: {avg_power:.3f}W avg, "
                   f"{efficiency_ips_per_watt:.1f} IPS/W")
        
        return analysis
    
    def _take_power_sample(self) -> PowerSample:
        """Take a single power measurement sample (mock implementation)"""
        # Mock power measurement - in real implementation would interface with hardware
        timestamp = time.time()
        
        # Simulate realistic power values for Edge TPU v6
        base_power = 2.5 + np.random.normal(0, 0.1)  # ~2.5W base with noise
        if hasattr(self, '_workload_active') and self._workload_active:
            # Add inference power consumption
            inference_power = np.random.uniform(1.0, 3.0)  # 1-3W additional during inference
            base_power += inference_power
        
        # Simulate voltage and current
        voltage = 3.3 + np.random.normal(0, 0.05)  # 3.3V nominal with noise
        current = base_power / voltage
        
        # Simulate temperature
        temperature = 45.0 + np.random.normal(0, 2.0)  # ~45C with variation
        
        return PowerSample(
            timestamp=timestamp,
            power_w=max(0.1, base_power),  # Ensure positive power
            voltage_v=voltage,
            current_a=current,
            temperature_c=temperature
        )
    
    def _power_monitoring_loop(self):
        """Background power monitoring thread"""
        while self.is_monitoring:
            sample = self._take_power_sample()
            self.power_samples.append(sample)
            time.sleep(1.0 / self.sample_rate_hz)
    
    def _execute_workload(self, model, duration_s: float, workload_type: WorkloadType) -> int:
        """Execute workload pattern and return inference count"""
        self._workload_active = True
        inference_count = 0
        start_time = time.time()
        
        try:
            if workload_type == WorkloadType.CONTINUOUS:
                # Continuous inference
                while time.time() - start_time < duration_s:
                    self._mock_inference(model)
                    inference_count += 1
                    
            elif workload_type == WorkloadType.BURST:
                # Burst pattern: 1s active, 2s idle
                while time.time() - start_time < duration_s:
                    # Active burst
                    burst_start = time.time()
                    while time.time() - burst_start < 1.0:
                        self._mock_inference(model)
                        inference_count += 1
                    
                    # Idle period
                    self._workload_active = False
                    time.sleep(2.0)
                    self._workload_active = True
                    
            elif workload_type == WorkloadType.PERIODIC:
                # Periodic inference every 100ms
                while time.time() - start_time < duration_s:
                    self._mock_inference(model)
                    inference_count += 1
                    time.sleep(0.1)
                    
        finally:
            self._workload_active = False
        
        return inference_count
    
    def _mock_inference(self, model):
        """Mock model inference"""
        # Simulate inference time
        time.sleep(np.random.uniform(0.005, 0.015))  # 5-15ms inference time
    
    def _detect_thermal_throttling(self, power_trace: PowerTrace) -> int:
        """Detect thermal throttling events from power trace"""
        throttling_events = 0
        
        # Look for sudden power drops that might indicate throttling
        power_values = [sample.power_w for sample in power_trace.samples]
        
        for i in range(1, len(power_values)):
            power_drop = power_values[i-1] - power_values[i]
            if power_drop > 1.0:  # Sudden 1W drop might be throttling
                throttling_events += 1
        
        return throttling_events
    
    def _calculate_efficiency_score(self, 
                                   avg_power: float,
                                   peak_power: float, 
                                   efficiency: float,
                                   throttling: int) -> float:
        """Calculate overall power efficiency score (0-10)"""
        # Base score from efficiency (inferences per watt)
        efficiency_score = min(efficiency / 10.0, 5.0)  # Cap at 5 points
        
        # Penalty for high peak power
        peak_penalty = max(0, (peak_power - 5.0) * 0.5)
        
        # Penalty for thermal throttling
        throttling_penalty = throttling * 0.1
        
        # Power stability bonus (low std dev)
        # stability_bonus would be calculated if we had power_std available
        
        final_score = max(0, efficiency_score - peak_penalty - throttling_penalty)
        return min(final_score, 10.0)
    
    def plot_power_trace(self, power_trace: PowerTrace):
        """Plot power consumption over time"""
        try:
            import matplotlib.pyplot as plt
            
            timestamps = [sample.timestamp for sample in power_trace.samples]
            power_values = [sample.power_w for sample in power_trace.samples]
            
            # Convert timestamps to relative time
            start_time = timestamps[0]
            relative_times = [(t - start_time) for t in timestamps]
            
            plt.figure(figsize=(12, 6))
            plt.plot(relative_times, power_values, 'b-', linewidth=0.8)
            plt.axhline(y=self.idle_power_w, color='r', linestyle='--', 
                       label=f'Idle Power ({self.idle_power_w:.2f}W)')
            
            plt.title(f'Power Consumption - {power_trace.workload_type.value.title()} Workload')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Power (W)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    def plot_efficiency_comparison(self, devices: List[str]):
        """Plot power efficiency comparison across devices"""
        try:
            import matplotlib.pyplot as plt
            
            # Mock efficiency data for demonstration
            mock_efficiency = {
                'edge_tpu_v6': 15.2,
                'edge_tpu_v5e': 12.8,
                'gpu_nano': 8.4,
                'cpu_arm': 3.1
            }
            
            available_devices = [d for d in devices if d in mock_efficiency]
            efficiencies = [mock_efficiency[d] for d in available_devices]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(available_devices, efficiencies, 
                          color=['green', 'blue', 'orange', 'red'])
            plt.title('Power Efficiency Comparison')
            plt.ylabel('Inferences per Second per Watt')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, eff in zip(bars, efficiencies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{eff:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    def get_power_report(self, analysis: PowerAnalysis) -> Dict[str, Any]:
        """Generate comprehensive power analysis report"""
        return {
            'device': self.device,
            'power_metrics': {
                'average_power_w': analysis.avg_power_w,
                'peak_power_w': analysis.peak_power_w,
                'idle_power_w': analysis.idle_power_w,
                'active_power_w': analysis.active_power_w,
                'power_stability': analysis.power_std_w
            },
            'energy_metrics': {
                'total_energy_j': analysis.energy_total_j,
                'energy_per_inference_mj': analysis.energy_per_inference_mj
            },
            'efficiency_metrics': {
                'inferences_per_watt': analysis.efficiency_ips_per_watt,
                'power_efficiency_score': analysis.power_efficiency_score
            },
            'thermal_metrics': {
                'throttling_events': analysis.thermal_throttling_events
            }
        }