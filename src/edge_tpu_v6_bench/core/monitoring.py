"""
Real-time monitoring and health checks for Edge TPU v6 benchmarking
Comprehensive system monitoring, alerting, and diagnostics
"""

import time
import threading
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import queue
import statistics
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

class MetricType(Enum):
    """Types of metrics to monitor"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    DEVICE_TEMPERATURE = "device_temperature"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    POWER_CONSUMPTION = "power_consumption"

@dataclass
class MetricSample:
    """Single metric measurement"""
    metric_type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable[[], bool]
    warning_threshold: float = 80.0
    critical_threshold: float = 95.0
    check_interval: float = 30.0
    enabled: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0
    max_failures: int = 3

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    timestamp: float
    severity: HealthStatus
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    resolved: bool = False

class SystemMonitor:
    """
    Comprehensive system monitoring and health checking
    
    Features:
    - Real-time resource monitoring
    - Configurable health checks
    - Alerting and notification system  
    - Performance trending and analysis
    - Automatic recovery recommendations
    """
    
    def __init__(self, 
                 sample_interval: float = 1.0,
                 history_size: int = 1000,
                 enable_alerting: bool = True):
        """
        Initialize system monitor
        
        Args:
            sample_interval: Sampling interval in seconds
            history_size: Maximum samples to keep in memory
            enable_alerting: Enable alert generation
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.enable_alerting = enable_alerting
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Metric storage
        self.metrics_history: Dict[MetricType, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.current_metrics: Dict[MetricType, MetricSample] = {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status = HealthStatus.HEALTHY
        
        # Alerting
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Performance analysis
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Thread synchronization
        self.metrics_lock = threading.RLock()
        self.alerts_lock = threading.RLock()
        
        self._setup_default_health_checks()
        logger.info(f"SystemMonitor initialized: {sample_interval}s interval, {history_size} samples history")
    
    def _setup_default_health_checks(self):
        """Setup default system health checks"""
        
        # CPU usage check
        self.add_health_check(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            warning_threshold=70.0,
            critical_threshold=90.0,
            check_interval=10.0
        ))
        
        # Memory usage check
        self.add_health_check(HealthCheck(
            name="memory_usage", 
            check_function=self._check_memory_usage,
            warning_threshold=80.0,
            critical_threshold=95.0,
            check_interval=10.0
        ))
        
        # System load check
        self.add_health_check(HealthCheck(
            name="system_load",
            check_function=self._check_system_load,
            warning_threshold=80.0,
            critical_threshold=95.0,
            check_interval=15.0
        ))
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already active")
            return
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, using mock monitoring")
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SystemMonitor"
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.debug("Monitoring loop started")
        
        while self.monitoring:
            start_time = time.time()
            
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                # Update overall health status
                self._update_health_status()
                
                # Analyze trends
                self._analyze_trends()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.sample_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.debug("Monitoring loop stopped")
    
    def _collect_metrics(self):
        """Collect current system metrics"""
        timestamp = time.time()
        
        try:
            if PSUTIL_AVAILABLE:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_sample = MetricSample(
                    MetricType.CPU_USAGE, cpu_percent, timestamp, unit="%"
                )
                
                # Memory metrics  
                memory = psutil.virtual_memory()
                memory_sample = MetricSample(
                    MetricType.MEMORY_USAGE, memory.percent, timestamp, unit="%"
                )
                
                # Network I/O metrics
                network = psutil.net_io_counters()
                if network:
                    network_total = network.bytes_sent + network.bytes_recv
                    network_sample = MetricSample(
                        MetricType.NETWORK_IO, network_total, timestamp, unit="bytes"
                    )
                else:
                    network_sample = MetricSample(
                        MetricType.NETWORK_IO, 0.0, timestamp, unit="bytes"
                    )
                
                metrics = [cpu_sample, memory_sample, network_sample]
            else:
                # Mock metrics for testing
                import random
                cpu_sample = MetricSample(
                    MetricType.CPU_USAGE, random.uniform(10, 80), timestamp, unit="%"
                )
                memory_sample = MetricSample(
                    MetricType.MEMORY_USAGE, random.uniform(20, 70), timestamp, unit="%"
                )
                metrics = [cpu_sample, memory_sample]
            
            # Store metrics
            with self.metrics_lock:
                for metric in metrics:
                    self.current_metrics[metric.metric_type] = metric
                    self.metrics_history[metric.metric_type].append(metric)
                
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    def _run_health_checks(self):
        """Run all enabled health checks"""
        current_time = time.time()
        
        for check in self.health_checks.values():
            if not check.enabled:
                continue
            
            # Check if it's time to run this check
            if current_time - check.last_check < check.check_interval:
                continue
            
            try:
                check.last_check = current_time
                is_healthy = check.check_function()
                
                if is_healthy:
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1
                    
                    # Generate alert if threshold exceeded
                    if check.consecutive_failures >= check.max_failures:
                        self._generate_health_alert(check)
                        
            except Exception as e:
                logger.error(f"Health check '{check.name}' failed: {e}")
                check.consecutive_failures += 1
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage health"""
        if MetricType.CPU_USAGE not in self.current_metrics:
            return True  # No data available, assume healthy
        
        cpu_usage = self.current_metrics[MetricType.CPU_USAGE].value
        check = self.health_checks["cpu_usage"]
        
        if cpu_usage > check.critical_threshold:
            return False
        elif cpu_usage > check.warning_threshold:
            self._generate_warning_alert("cpu_usage", cpu_usage, check.warning_threshold)
        
        return True
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage health"""
        if MetricType.MEMORY_USAGE not in self.current_metrics:
            return True
        
        memory_usage = self.current_metrics[MetricType.MEMORY_USAGE].value
        check = self.health_checks["memory_usage"]
        
        if memory_usage > check.critical_threshold:
            return False
        elif memory_usage > check.warning_threshold:
            self._generate_warning_alert("memory_usage", memory_usage, check.warning_threshold)
        
        return True
    
    def _check_system_load(self) -> bool:
        """Check system load health"""
        if not PSUTIL_AVAILABLE:
            return True
        
        try:
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100
            
            check = self.health_checks["system_load"]
            
            if load_percent > check.critical_threshold:
                return False
            elif load_percent > check.warning_threshold:
                self._generate_warning_alert("system_load", load_percent, check.warning_threshold)
            
            return True
            
        except Exception:
            return True  # If we can't get load average, assume healthy
    
    def _generate_health_alert(self, check: HealthCheck):
        """Generate alert for failed health check"""
        if not self.enable_alerting:
            return
        
        metric_type = getattr(MetricType, check.name.upper(), MetricType.CPU_USAGE)
        current_value = self.current_metrics.get(metric_type)
        
        alert = Alert(
            alert_id=f"health_{check.name}_{int(time.time())}",
            timestamp=time.time(),
            severity=HealthStatus.CRITICAL,
            metric_type=metric_type,
            message=f"Health check '{check.name}' failed {check.consecutive_failures} times",
            value=current_value.value if current_value else 0.0,
            threshold=check.critical_threshold
        )
        
        self._add_alert(alert)
    
    def _generate_warning_alert(self, metric_name: str, value: float, threshold: float):
        """Generate warning alert"""
        if not self.enable_alerting:
            return
        
        metric_type = getattr(MetricType, metric_name.upper(), MetricType.CPU_USAGE)
        
        alert = Alert(
            alert_id=f"warning_{metric_name}_{int(time.time())}",
            timestamp=time.time(),
            severity=HealthStatus.WARNING,
            metric_type=metric_type,
            message=f"{metric_name} is above warning threshold",
            value=value,
            threshold=threshold
        )
        
        self._add_alert(alert)
    
    def _add_alert(self, alert: Alert):
        """Add alert and notify handlers"""
        with self.alerts_lock:
            self.alerts.append(alert)
            
            # Keep alerts list bounded
            if len(self.alerts) > 1000:
                self.alerts.pop(0)
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        # Log alert
        logger.log(
            logging.ERROR if alert.severity == HealthStatus.CRITICAL else logging.WARNING,
            f"ALERT [{alert.severity.value}] {alert.message}: {alert.value:.1f} > {alert.threshold:.1f}"
        )
    
    def _update_health_status(self):
        """Update overall system health status"""
        # Count recent critical alerts
        current_time = time.time()
        recent_critical = sum(
            1 for alert in self.alerts[-10:]  # Last 10 alerts
            if not alert.resolved and 
               alert.severity == HealthStatus.CRITICAL and 
               current_time - alert.timestamp < 300  # Last 5 minutes
        )
        
        recent_warnings = sum(
            1 for alert in self.alerts[-20:]  # Last 20 alerts
            if not alert.resolved and 
               alert.severity == HealthStatus.WARNING and 
               current_time - alert.timestamp < 300
        )
        
        # Determine overall health
        if recent_critical > 0:
            self.health_status = HealthStatus.CRITICAL
        elif recent_warnings > 2:
            self.health_status = HealthStatus.WARNING
        else:
            self.health_status = HealthStatus.HEALTHY
    
    def _analyze_trends(self):
        """Analyze performance trends"""
        try:
            # Analyze CPU trend
            if len(self.metrics_history[MetricType.CPU_USAGE]) >= 10:
                recent_cpu = [m.value for m in list(self.metrics_history[MetricType.CPU_USAGE])[-10:]]
                cpu_trend = statistics.mean(recent_cpu)
                self.performance_trends["cpu_trend"].append(cpu_trend)
            
            # Analyze memory trend
            if len(self.metrics_history[MetricType.MEMORY_USAGE]) >= 10:
                recent_memory = [m.value for m in list(self.metrics_history[MetricType.MEMORY_USAGE])[-10:]]
                memory_trend = statistics.mean(recent_memory)
                self.performance_trends["memory_trend"].append(memory_trend)
            
            # Keep trends bounded
            for trend_list in self.performance_trends.values():
                if len(trend_list) > 100:
                    trend_list.pop(0)
                    
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
    
    def add_health_check(self, health_check: HealthCheck):
        """Add custom health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        with self.metrics_lock:
            return {
                metric_type.value: sample.value
                for metric_type, sample in self.current_metrics.items()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        current_metrics = self.get_current_metrics()
        
        # Recent alerts summary
        recent_alerts = [
            alert for alert in self.alerts[-20:]
            if not alert.resolved and time.time() - alert.timestamp < 300
        ]
        
        critical_count = sum(1 for a in recent_alerts if a.severity == HealthStatus.CRITICAL)
        warning_count = sum(1 for a in recent_alerts if a.severity == HealthStatus.WARNING)
        
        return {
            "overall_status": self.health_status.value,
            "current_metrics": current_metrics,
            "health_checks": {
                name: {
                    "enabled": check.enabled,
                    "consecutive_failures": check.consecutive_failures,
                    "last_check": check.last_check,
                    "warning_threshold": check.warning_threshold,
                    "critical_threshold": check.critical_threshold
                }
                for name, check in self.health_checks.items()
            },
            "alert_summary": {
                "critical_alerts": critical_count,
                "warning_alerts": warning_count,
                "total_alerts": len(self.alerts),
                "recent_alerts": len(recent_alerts)
            }
        }

# Global system monitor instance
global_system_monitor = SystemMonitor()