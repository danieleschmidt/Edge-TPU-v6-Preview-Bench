"""
Robust benchmark implementation for Generation 2 - Make It Robust
Enhanced error handling, validation, security, and comprehensive logging
"""

import time
import statistics
import logging
import hashlib
import json
import os
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading
import queue
import traceback

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(thread)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('edge_tpu_robust.log')
    ]
)
logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Security-related errors"""
    pass

class ValidationError(Exception):
    """Input validation errors"""
    pass

class BenchmarkError(Exception):
    """Benchmark execution errors"""
    pass

class DeviceError(Exception):
    """Device-related errors"""
    pass

class SecurityLevel(Enum):
    """Security level settings"""
    DISABLED = "disabled"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"

@dataclass
class SecurityConfig:
    """Security configuration"""
    level: SecurityLevel = SecurityLevel.STRICT
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: ['.tflite', '.txt', '.pb'])
    blocked_paths: List[str] = field(default_factory=lambda: ['/etc', '/proc', '/sys'])
    enable_integrity_checks: bool = True
    enable_sandboxing: bool = True

@dataclass 
class RobustBenchmarkConfig:
    """Robust configuration with validation"""
    warmup_runs: int = 10
    measurement_runs: int = 100
    timeout_seconds: float = 300.0
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    enable_monitoring: bool = True
    enable_checkpoints: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.warmup_runs < 0:
            raise ValidationError("Warmup runs must be non-negative")
        if self.measurement_runs <= 0:
            raise ValidationError("Measurement runs must be positive")
        if self.timeout_seconds <= 0:
            raise ValidationError("Timeout must be positive")
        if not all(bs > 0 for bs in self.batch_sizes):
            raise ValidationError("All batch sizes must be positive")
        if self.retry_attempts < 0:
            raise ValidationError("Retry attempts must be non-negative")

@dataclass
class RobustBenchmarkResult:
    """Comprehensive benchmark results with metadata"""
    success: bool
    model_path: str
    device_type: str
    security_level: str
    
    # Core metrics
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    
    # Throughput metrics
    throughput_fps: float
    throughput_mbps: float
    
    # Reliability metrics
    total_measurements: int
    successful_measurements: int
    failed_measurements: int
    success_rate: float
    
    # System metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    
    # Error information
    error_message: Optional[str] = None
    error_trace: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Timing information
    start_timestamp: float = field(default_factory=time.time)
    end_timestamp: Optional[float] = None
    total_duration_s: Optional[float] = None
    
    # Model information
    model_size_mb: float = 0.0
    model_hash: Optional[str] = None
    
    # Validation results
    integrity_verified: bool = False
    security_scan_passed: bool = False

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise BenchmarkError("Circuit breaker is OPEN - too many failures")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.reset()
                return result
            except Exception as e:
                self.record_failure()
                raise e
    
    def record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def reset(self):
        """Reset circuit breaker"""
        self.failure_count = 0
        self.state = 'CLOSED'
        logger.info("Circuit breaker reset to CLOSED")

class SecurityManager:
    """Comprehensive security management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.allowed_hashes = set()
        logger.info(f"Security manager initialized with level: {config.level.value}")
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and sanitize file path"""
        path = Path(file_path).resolve()
        
        # Check if file exists
        if not path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise SecurityError(f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB")
        
        # Check file extension
        if path.suffix.lower() not in self.config.allowed_extensions:
            raise SecurityError(f"File extension not allowed: {path.suffix}")
        
        # Check for blocked paths
        for blocked in self.config.blocked_paths:
            if str(path).startswith(blocked):
                raise SecurityError(f"Access to blocked path: {blocked}")
        
        logger.debug(f"File path validation passed: {path}")
        return path
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            raise SecurityError(f"Failed to compute file hash: {e}")
    
    def verify_file_integrity(self, file_path: Path, expected_hash: Optional[str] = None) -> bool:
        """Verify file integrity"""
        if not self.config.enable_integrity_checks:
            return True
        
        try:
            actual_hash = self.compute_file_hash(file_path)
            
            if expected_hash:
                if actual_hash != expected_hash:
                    raise SecurityError(f"File integrity check failed: {file_path}")
                logger.info(f"File integrity verified: {file_path}")
                return True
            else:
                # Store hash for future verification
                self.allowed_hashes.add(actual_hash)
                logger.debug(f"File hash stored: {actual_hash[:16]}...")
                return True
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def scan_for_threats(self, file_path: Path) -> List[str]:
        """Basic threat scanning"""
        threats = []
        
        try:
            # Check file content for suspicious patterns
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
                
                # Look for suspicious binary patterns
                suspicious_patterns = [
                    b'\x4d\x5a',  # PE header
                    b'\x7f\x45\x4c\x46',  # ELF header
                    b'#!/bin/sh',  # Shell script
                    b'#!/bin/bash',  # Bash script
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in content:
                        threats.append(f"Suspicious pattern detected: {pattern.hex()}")
        except Exception as e:
            threats.append(f"Failed to scan file: {e}")
        
        if threats:
            logger.warning(f"Threats detected in {file_path}: {threats}")
        
        return threats

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.start_time = None
        self.measurements = []
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.start_time = time.time()
        self.measurements = []
        logger.debug("System monitoring started")
    
    def take_measurement(self) -> Dict[str, float]:
        """Take a system measurement"""
        try:
            # Basic CPU and memory monitoring without psutil
            import os
            
            # Get load average (Unix-like systems only)
            try:
                load_avg = os.getloadavg()[0]
                cpu_usage = min(load_avg * 20, 100.0)  # Rough estimate
            except (AttributeError, OSError):
                cpu_usage = 0.0
            
            # Get memory info from /proc/meminfo (Linux only)
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    
                total_kb = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
                avail_kb = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1])
                
                memory_usage_mb = (total_kb - avail_kb) / 1024
            except (FileNotFoundError, IndexError, ValueError):
                memory_usage_mb = 0.0
            
            measurement = {
                'timestamp': time.time(),
                'cpu_usage_percent': cpu_usage,
                'memory_usage_mb': memory_usage_mb
            }
            
            self.measurements.append(measurement)
            return measurement
            
        except Exception as e:
            logger.error(f"Failed to take system measurement: {e}")
            return {'timestamp': time.time(), 'cpu_usage_percent': 0.0, 'memory_usage_mb': 0.0}
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average system metrics"""
        if not self.measurements:
            return {'cpu_usage_percent': 0.0, 'memory_usage_mb': 0.0}
        
        avg_cpu = sum(m['cpu_usage_percent'] for m in self.measurements) / len(self.measurements)
        avg_memory = sum(m['memory_usage_mb'] for m in self.measurements) / len(self.measurements)
        
        return {
            'cpu_usage_percent': avg_cpu,
            'memory_usage_mb': avg_memory
        }

class RobustEdgeTPUBenchmark:
    """
    Robust Edge TPU benchmark with comprehensive error handling,
    security, validation, and monitoring capabilities
    """
    
    def __init__(self, 
                 device: str = 'edge_tpu_v6',
                 security_config: Optional[SecurityConfig] = None):
        """Initialize robust benchmark"""
        self.device_type = device
        self.model_path: Optional[str] = None
        self.security_config = security_config or SecurityConfig()
        self.security_manager = SecurityManager(self.security_config)
        self.circuit_breaker = CircuitBreaker()
        self.system_monitor = SystemMonitor()
        
        # Initialize state
        self.is_initialized = True
        self.benchmark_id = hashlib.md5(f"{device}_{time.time()}".encode()).hexdigest()[:8]
        
        logger.info(f"RobustEdgeTPUBenchmark initialized: {self.benchmark_id}")
        logger.info(f"Device: {device}, Security: {self.security_config.level.value}")
    
    def load_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate model with comprehensive security checks"""
        try:
            # Security validation
            validated_path = self.security_manager.validate_file_path(model_path)
            
            # Compute file hash for integrity
            file_hash = self.security_manager.compute_file_hash(validated_path)
            
            # Verify integrity
            integrity_ok = self.security_manager.verify_file_integrity(validated_path)
            
            # Threat scanning
            threats = self.security_manager.scan_for_threats(validated_path)
            if threats and self.security_config.level != SecurityLevel.DISABLED:
                raise SecurityError(f"Security threats detected: {threats}")
            
            self.model_path = str(validated_path)
            
            # Get model info
            model_size_bytes = validated_path.stat().st_size
            model_info = {
                'path': self.model_path,
                'size_bytes': model_size_bytes,
                'size_mb': model_size_bytes / (1024 * 1024),
                'format': validated_path.suffix,
                'hash': file_hash,
                'integrity_verified': integrity_ok,
                'security_scan_passed': len(threats) == 0
            }
            
            logger.info(f"Model loaded successfully: {model_info['size_mb']:.1f} MB")
            logger.info(f"Model hash: {file_hash[:16]}...")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise BenchmarkError(f"Model loading failed: {e}")
    
    def benchmark(self, 
                  model_path: Optional[Union[str, Path]] = None,
                  config: Optional[RobustBenchmarkConfig] = None) -> RobustBenchmarkResult:
        """
        Run comprehensive robust benchmark with error handling,
        monitoring, and detailed reporting
        """
        start_time = time.time()
        
        # Initialize configuration
        if config is None:
            config = RobustBenchmarkConfig()
        
        # Initialize result
        result = RobustBenchmarkResult(
            success=False,
            model_path='',
            device_type=self.device_type,
            security_level=self.security_config.level.value,
            latency_mean_ms=0.0,
            latency_median_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            latency_std_ms=0.0,
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            throughput_fps=0.0,
            throughput_mbps=0.0,
            total_measurements=0,
            successful_measurements=0,
            failed_measurements=0,
            success_rate=0.0,
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            start_timestamp=start_time
        )
        
        try:
            # Load model if provided
            if model_path:
                model_info = self.load_model(model_path)
                result.model_path = model_info['path']
                result.model_size_mb = model_info['size_mb']
                result.model_hash = model_info['hash']
                result.integrity_verified = model_info['integrity_verified']
                result.security_scan_passed = model_info['security_scan_passed']
            
            if not self.model_path:
                raise BenchmarkError("No model loaded for benchmarking")
            
            # Start system monitoring
            if config.enable_monitoring:
                self.system_monitor.start_monitoring()
            
            logger.info(f"Starting robust benchmark: {config.measurement_runs} runs, "
                       f"{config.retry_attempts} max retries")
            
            # Run benchmark with circuit breaker protection
            latency_measurements = []
            failed_count = 0
            
            # Warmup phase
            logger.info(f"Warmup phase: {config.warmup_runs} runs")
            for i in range(config.warmup_runs):
                try:
                    self.circuit_breaker.call(self._simulate_inference_robust)
                except Exception as e:
                    logger.warning(f"Warmup run {i} failed: {e}")
                    result.warnings.append(f"Warmup run {i} failed: {e}")
            
            # Measurement phase
            logger.info(f"Measurement phase: {config.measurement_runs} runs")
            for run in range(config.measurement_runs):
                retry_count = 0
                measurement_successful = False
                
                while retry_count <= config.retry_attempts and not measurement_successful:
                    try:
                        if config.enable_monitoring:
                            self.system_monitor.take_measurement()
                        
                        start_inference = time.perf_counter()
                        self.circuit_breaker.call(self._simulate_inference_robust)
                        end_inference = time.perf_counter()
                        
                        latency_ms = (end_inference - start_inference) * 1000.0
                        latency_measurements.append(latency_ms)
                        measurement_successful = True
                        
                        # Checkpoint every 10 runs
                        if config.enable_checkpoints and (run + 1) % 10 == 0:
                            logger.debug(f"Checkpoint: {run + 1}/{config.measurement_runs} runs completed")
                        
                    except Exception as e:
                        retry_count += 1
                        failed_count += 1
                        
                        if retry_count <= config.retry_attempts:
                            logger.warning(f"Run {run} attempt {retry_count} failed: {e}, retrying...")
                            time.sleep(0.1 * retry_count)  # Exponential backoff
                        else:
                            logger.error(f"Run {run} failed after {config.retry_attempts} attempts: {e}")
                            result.warnings.append(f"Run {run} failed: {e}")
                
                # Check timeout
                if time.time() - start_time > config.timeout_seconds:
                    logger.warning(f"Benchmark timeout reached at run {run}")
                    result.warnings.append("Benchmark timeout reached")
                    break
            
            # Calculate comprehensive metrics
            if latency_measurements:
                result.latency_mean_ms = statistics.mean(latency_measurements)
                result.latency_median_ms = statistics.median(latency_measurements)
                result.latency_p95_ms = self._percentile(latency_measurements, 95)
                result.latency_p99_ms = self._percentile(latency_measurements, 99)
                result.latency_std_ms = statistics.stdev(latency_measurements) if len(latency_measurements) > 1 else 0.0
                result.latency_min_ms = min(latency_measurements)
                result.latency_max_ms = max(latency_measurements)
                
                result.throughput_fps = 1000.0 / result.latency_mean_ms if result.latency_mean_ms > 0 else 0.0
                result.throughput_mbps = result.throughput_fps * result.model_size_mb
                
                result.total_measurements = config.measurement_runs
                result.successful_measurements = len(latency_measurements)
                result.failed_measurements = failed_count
                result.success_rate = result.successful_measurements / result.total_measurements
                
                # Get system metrics
                if config.enable_monitoring:
                    sys_metrics = self.system_monitor.get_average_metrics()
                    result.cpu_usage_percent = sys_metrics['cpu_usage_percent']
                    result.memory_usage_mb = sys_metrics['memory_usage_mb']
                
                result.success = True
                
                logger.info(f"Benchmark completed successfully!")
                logger.info(f"Latency: {result.latency_mean_ms:.2f}ms ± {result.latency_std_ms:.2f}ms")
                logger.info(f"Throughput: {result.throughput_fps:.1f} FPS")
                logger.info(f"Success rate: {result.success_rate:.1%}")
                
            else:
                raise BenchmarkError("No successful measurements collected")
            
        except Exception as e:
            result.error_message = str(e)
            result.error_trace = traceback.format_exc()
            logger.error(f"Benchmark failed: {e}")
            logger.debug(f"Error trace: {result.error_trace}")
        
        finally:
            result.end_timestamp = time.time()
            result.total_duration_s = result.end_timestamp - result.start_timestamp
        
        return result
    
    def _simulate_inference_robust(self):
        """Simulate inference with potential failure scenarios"""
        # Simulate variable inference time based on device type
        base_latency = {
            'edge_tpu_v6': 0.002,
            'edge_tpu_v5e': 0.003,
            'cpu_fallback': 0.020,
        }.get(self.device_type, 0.010)
        
        # Add realistic variation and occasional failures
        import random
        
        # 2% chance of simulated failure for testing robustness
        if random.random() < 0.02:
            failure_types = [
                "Device timeout",
                "Memory allocation failed", 
                "Model compilation error",
                "Hardware busy"
            ]
            raise DeviceError(random.choice(failure_types))
        
        # Normal operation with variation
        variation = random.uniform(0.7, 1.5)
        sleep_time = base_latency * variation
        
        time.sleep(sleep_time)
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1
        
        if c >= len(sorted_data):
            return sorted_data[-1]
        if f < 0:
            return sorted_data[0]
        
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return {
            'device_type': self.device_type,
            'benchmark_id': self.benchmark_id,
            'security_level': self.security_config.level.value,
            'circuit_breaker_state': self.circuit_breaker.state,
            'status': 'available' if self.is_initialized else 'error'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'issues': []
        }
        
        try:
            # Test basic functionality
            self._simulate_inference_robust()
            health['basic_inference'] = 'ok'
        except Exception as e:
            health['basic_inference'] = f'failed: {e}'
            health['issues'].append('Basic inference test failed')
            health['status'] = 'unhealthy'
        
        # Check circuit breaker state
        if self.circuit_breaker.state != 'CLOSED':
            health['circuit_breaker'] = self.circuit_breaker.state
            health['issues'].append(f'Circuit breaker is {self.circuit_breaker.state}')
            health['status'] = 'degraded'
        
        return health

def save_robust_results(result: RobustBenchmarkResult, output_dir: str) -> str:
    """Save comprehensive benchmark results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    json_path = output_path / f'robust_benchmark_{timestamp}.json'
    
    # Convert result to dict with all fields
    result_dict = {
        'benchmark_info': {
            'timestamp': result.start_timestamp,
            'end_timestamp': result.end_timestamp,
            'total_duration_s': result.total_duration_s,
            'success': result.success,
            'device_type': result.device_type,
            'security_level': result.security_level
        },
        'model_info': {
            'path': result.model_path,
            'size_mb': result.model_size_mb,
            'hash': result.model_hash,
            'integrity_verified': result.integrity_verified,
            'security_scan_passed': result.security_scan_passed
        },
        'performance_metrics': {
            'latency_mean_ms': result.latency_mean_ms,
            'latency_median_ms': result.latency_median_ms,
            'latency_p95_ms': result.latency_p95_ms,
            'latency_p99_ms': result.latency_p99_ms,
            'latency_std_ms': result.latency_std_ms,
            'latency_min_ms': result.latency_min_ms,
            'latency_max_ms': result.latency_max_ms,
            'throughput_fps': result.throughput_fps,
            'throughput_mbps': result.throughput_mbps
        },
        'reliability_metrics': {
            'total_measurements': result.total_measurements,
            'successful_measurements': result.successful_measurements,
            'failed_measurements': result.failed_measurements,
            'success_rate': result.success_rate
        },
        'system_metrics': {
            'cpu_usage_percent': result.cpu_usage_percent,
            'memory_usage_mb': result.memory_usage_mb
        },
        'error_info': {
            'error_message': result.error_message,
            'error_trace': result.error_trace,
            'warnings': result.warnings
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"Robust benchmark results saved to: {json_path}")
    return str(json_path)