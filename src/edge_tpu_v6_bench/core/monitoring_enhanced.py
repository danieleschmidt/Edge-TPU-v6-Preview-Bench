"""
Enhanced Monitoring and Observability System
Production-grade monitoring with metrics, logging, and alerting
"""

import time
import logging
import threading
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import statistics
from collections import defaultdict, deque
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert definition and state"""
    name: str
    condition: str
    threshold: float
    severity: str
    description: str
    active: bool = False
    triggered_at: Optional[float] = None
    resolved_at: Optional[float] = None

class MetricsCollector:
    """
    Advanced metrics collection system with Prometheus-style metrics
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        
    def counter_inc(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            metric_key = self._build_metric_key(name, labels or {})
            self.counters[metric_key] += value
            self._record_metric(name, self.counters[metric_key], labels or {})
    
    def gauge_set(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        with self.lock:
            metric_key = self._build_metric_key(name, labels or {})
            self.gauges[metric_key] = value
            self._record_metric(name, value, labels or {})
    
    def histogram_observe(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a value in a histogram"""
        with self.lock:
            metric_key = self._build_metric_key(name, labels or {})
            self.histograms[metric_key].append(value)
            # Keep only last 1000 observations per histogram
            if len(self.histograms[metric_key]) > 1000:
                self.histograms[metric_key] = self.histograms[metric_key][-1000:]
            self._record_metric(name, value, labels or {})
    
    def _build_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Build a unique key for metric with labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _record_metric(self, name: str, value: float, labels: Dict[str, str]):
        """Record metric point with timestamp"""
        point = MetricPoint(timestamp=time.time(), value=value, labels=labels)
        self.metrics[name].append(point)
        self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        for metric_name, points in self.metrics.items():
            while points and points[0].timestamp < cutoff_time:
                points.popleft()
    
    def get_metric_summary(self, name: str, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get statistical summary of a metric over time range"""
        cutoff_time = time.time() - (time_range_minutes * 60)
        
        if name not in self.metrics:
            return {"error": f"Metric {name} not found"}
        
        recent_points = [p for p in self.metrics[name] if p.timestamp >= cutoff_time]
        
        if not recent_points:
            return {"error": f"No data for {name} in last {time_range_minutes} minutes"}
        
        values = [p.value for p in recent_points]
        
        return {
            "metric_name": name,
            "time_range_minutes": time_range_minutes,
            "data_points": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted(values)[int(0.95 * len(values))] if len(values) > 0 else 0,
            "p99": sorted(values)[int(0.99 * len(values))] if len(values) > 0 else 0,
            "stddev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Export counters
        for metric_key, value in self.counters.items():
            lines.append(f"{metric_key} {value}")
        
        # Export gauges
        for metric_key, value in self.gauges.items():
            lines.append(f"{metric_key} {value}")
        
        # Export histogram summaries
        for metric_key, values in self.histograms.items():
            if values:
                lines.append(f"{metric_key}_sum {sum(values)}")
                lines.append(f"{metric_key}_count {len(values)}")
                lines.append(f"{metric_key}_avg {statistics.mean(values)}")
        
        return "\n".join(lines)

class AlertManager:
    """
    Advanced alerting system with configurable thresholds
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        self.evaluation_interval = 30  # seconds
        self.running = False
        self.thread = None
    
    def add_alert(self, alert: Alert):
        """Add an alert rule"""
        self.alerts[alert.name] = alert
        logger.info(f"Added alert rule: {alert.name}")
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def start(self):
        """Start alert evaluation loop"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.thread.start()
        logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert evaluation loop"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Alert manager stopped")
    
    def _evaluation_loop(self):
        """Main alert evaluation loop"""
        while self.running:
            try:
                self._evaluate_alerts()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                time.sleep(5)  # Brief pause on error
    
    def _evaluate_alerts(self):
        """Evaluate all alert rules"""
        for alert_name, alert in self.alerts.items():
            try:
                self._evaluate_single_alert(alert)
            except Exception as e:
                logger.error(f"Error evaluating alert {alert_name}: {e}")
    
    def _evaluate_single_alert(self, alert: Alert):
        """Evaluate a single alert rule"""
        # Parse condition (simplified - could be expanded)
        if ">" in alert.condition:
            metric_name, operator, threshold_str = alert.condition.partition(">")
            metric_name = metric_name.strip()
            threshold = float(threshold_str.strip())
            
            summary = self.metrics_collector.get_metric_summary(metric_name, 5)
            
            if "error" not in summary:
                current_value = summary.get("mean", 0)
                should_fire = current_value > threshold
                
                if should_fire and not alert.active:
                    # Fire alert
                    alert.active = True
                    alert.triggered_at = time.time()
                    self._fire_alert(alert, current_value)
                    
                elif not should_fire and alert.active:
                    # Resolve alert
                    alert.active = False
                    alert.resolved_at = time.time()
                    self._resolve_alert(alert, current_value)
    
    def _fire_alert(self, alert: Alert, current_value: float):
        """Fire an alert"""
        logger.warning(f"ALERT FIRED: {alert.name} - {alert.description} (value: {current_value})")
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _resolve_alert(self, alert: Alert, current_value: float):
        """Resolve an alert"""
        logger.info(f"ALERT RESOLVED: {alert.name} (value: {current_value})")
        
        # Could add resolution handlers here

class PerformanceMonitor:
    """
    Production-grade performance monitoring system
    """
    
    def __init__(self, enable_alerts: bool = True):
        self.metrics = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics) if enable_alerts else None
        self.monitoring_active = False
        
        # Setup default alerts
        if self.alert_manager:
            self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        alerts = [
            Alert(
                name="high_latency",
                condition="benchmark_latency_ms > 10.0",
                threshold=10.0,
                severity="warning",
                description="Benchmark latency is above 10ms"
            ),
            Alert(
                name="low_throughput", 
                condition="benchmark_throughput_fps > 100.0",
                threshold=100.0,
                severity="warning",
                description="Benchmark throughput dropped below 100 FPS"
            ),
            Alert(
                name="high_error_rate",
                condition="benchmark_error_rate > 0.05",
                threshold=0.05,
                severity="critical",
                description="Benchmark error rate above 5%"
            )
        ]
        
        for alert in alerts:
            self.alert_manager.add_alert(alert)
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        if self.alert_manager:
            self.alert_manager.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.alert_manager:
            self.alert_manager.stop()
        logger.info("Performance monitoring stopped")
    
    def record_benchmark_result(self, 
                              latency_ms: float,
                              throughput_fps: float,
                              success: bool = True,
                              model_name: str = "unknown"):
        """Record benchmark results as metrics"""
        labels = {"model": model_name}
        
        self.metrics.histogram_observe("benchmark_latency_ms", latency_ms, labels)
        self.metrics.gauge_set("benchmark_throughput_fps", throughput_fps, labels)
        self.metrics.counter_inc("benchmark_total", 1.0, labels)
        
        if not success:
            self.metrics.counter_inc("benchmark_errors", 1.0, labels)
        
        # Calculate error rate
        total_key = self.metrics._build_metric_key("benchmark_total", labels)
        error_key = self.metrics._build_metric_key("benchmark_errors", labels)
        
        total_count = self.metrics.counters.get(total_key, 0)
        error_count = self.metrics.counters.get(error_key, 0)
        
        error_rate = error_count / total_count if total_count > 0 else 0
        self.metrics.gauge_set("benchmark_error_rate", error_rate, labels)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        status = {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "metrics_collected": len(self.metrics.metrics),
            "active_alerts": 0 if not self.alert_manager else len([a for a in self.alert_manager.alerts.values() if a.active])
        }
        
        # Get recent performance summary
        latency_summary = self.metrics.get_metric_summary("benchmark_latency_ms", 10)
        throughput_summary = self.metrics.get_metric_summary("benchmark_throughput_fps", 10)
        
        status["recent_performance"] = {
            "latency": latency_summary,
            "throughput": throughput_summary
        }
        
        return status
    
    def export_metrics_endpoint(self) -> str:
        """Export metrics in Prometheus format for scraping"""
        return self.metrics.export_prometheus_format()
    
    def generate_monitoring_report(self, output_path: Path):
        """Generate comprehensive monitoring report"""
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "monitoring_status": self.get_health_status(),
            "metric_summaries": {},
            "active_alerts": []
        }
        
        # Get summaries for all metrics
        for metric_name in self.metrics.metrics.keys():
            report["metric_summaries"][metric_name] = self.metrics.get_metric_summary(metric_name, 60)
        
        # Get active alerts
        if self.alert_manager:
            for alert in self.alert_manager.alerts.values():
                if alert.active:
                    report["active_alerts"].append({
                        "name": alert.name,
                        "severity": alert.severity,
                        "description": alert.description,
                        "triggered_at": alert.triggered_at
                    })
        
        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Monitoring report generated: {output_path}")

# Example usage and demo
def demo_enhanced_monitoring():
    """Demonstrate enhanced monitoring capabilities"""
    
    monitor = PerformanceMonitor(enable_alerts=True)
    monitor.start_monitoring()
    
    # Simulate benchmark results
    import random
    
    print("🔄 Generating sample benchmark data...")
    
    for i in range(50):
        # Simulate varying performance
        latency = random.uniform(1.0, 15.0)  # Some above alert threshold
        throughput = random.uniform(50, 200)  # Some below threshold
        success = random.random() > 0.02  # 2% error rate
        
        monitor.record_benchmark_result(
            latency_ms=latency,
            throughput_fps=throughput,
            success=success,
            model_name=f"model_{i % 3}"
        )
        
        time.sleep(0.1)  # Small delay
    
    # Wait for alerts to potentially fire
    time.sleep(2)
    
    # Generate reports
    health = monitor.get_health_status()
    print(f"📊 Health Status: {json.dumps(health, indent=2)}")
    
    metrics_export = monitor.export_metrics_endpoint()
    print(f"📈 Metrics (Prometheus format):\n{metrics_export[:500]}...")
    
    # Generate full report
    report_path = Path("monitoring_results/monitoring_report.json")
    monitor.generate_monitoring_report(report_path)
    
    monitor.stop_monitoring()
    
    return health

if __name__ == "__main__":
    demo_enhanced_monitoring()