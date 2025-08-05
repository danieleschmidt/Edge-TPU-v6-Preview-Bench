"""
Performance metrics collection and calculation utilities
Provides standardized metric definitions and calculation methods for Edge TPU benchmarking
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class MetricType(Enum):
    """Types of metrics that can be collected"""
    LATENCY = "latency"
    THROUGHPUT = "throughput" 
    ACCURACY = "accuracy"
    POWER = "power"
    THERMAL = "thermal"
    MEMORY = "memory"
    CUSTOM = "custom"

@dataclass
class MetricDefinition:
    """Definition of a performance metric"""
    name: str
    metric_type: MetricType
    unit: str
    description: str
    higher_is_better: bool = True
    decimal_places: int = 4

@dataclass
class MetricSample:
    """Single metric measurement sample"""
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class BenchmarkMetrics:
    """
    Centralized metrics collection and calculation
    
    Provides standardized metric definitions and calculation methods
    for consistent benchmarking across different Edge TPU devices
    """
    
    # Standard metric definitions
    STANDARD_METRICS = {
        'latency_mean_ms': MetricDefinition(
            'latency_mean_ms', MetricType.LATENCY, 'ms',
            'Mean inference latency', higher_is_better=False
        ),
        'latency_p50_ms': MetricDefinition(
            'latency_p50_ms', MetricType.LATENCY, 'ms', 
            '50th percentile (median) latency', higher_is_better=False
        ),
        'latency_p95_ms': MetricDefinition(
            'latency_p95_ms', MetricType.LATENCY, 'ms',
            '95th percentile latency', higher_is_better=False
        ),
        'latency_p99_ms': MetricDefinition(
            'latency_p99_ms', MetricType.LATENCY, 'ms',
            '99th percentile latency', higher_is_better=False
        ),
        'throughput_fps': MetricDefinition(
            'throughput_fps', MetricType.THROUGHPUT, 'fps',
            'Frames (inferences) per second', higher_is_better=True
        ),
        'throughput_ips': MetricDefinition(
            'throughput_ips', MetricType.THROUGHPUT, 'ips', 
            'Inferences per second', higher_is_better=True
        ),
        'power_mean_w': MetricDefinition(
            'power_mean_w', MetricType.POWER, 'W',
            'Mean power consumption', higher_is_better=False
        ),
        'energy_per_inference_mj': MetricDefinition(
            'energy_per_inference_mj', MetricType.POWER, 'mJ',
            'Energy per inference', higher_is_better=False
        ),
        'accuracy_top1': MetricDefinition(
            'accuracy_top1', MetricType.ACCURACY, '%',
            'Top-1 accuracy', higher_is_better=True
        ),
        'accuracy_top5': MetricDefinition(
            'accuracy_top5', MetricType.ACCURACY, '%', 
            'Top-5 accuracy', higher_is_better=True
        ),
        'thermal_max_c': MetricDefinition(
            'thermal_max_c', MetricType.THERMAL, '°C',
            'Maximum temperature', higher_is_better=False
        ),
        'memory_peak_mb': MetricDefinition(
            'memory_peak_mb', MetricType.MEMORY, 'MB',
            'Peak memory usage', higher_is_better=False
        ),
    }
    
    def __init__(self):
        self.samples: Dict[str, List[MetricSample]] = {}
        self.custom_metrics: Dict[str, MetricDefinition] = {}
        
    def register_metric(self, metric_def: MetricDefinition):
        """Register a custom metric definition"""
        self.custom_metrics[metric_def.name] = metric_def
        
    def record_sample(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a single metric sample"""
        if metric_name not in self.samples:
            self.samples[metric_name] = []
            
        sample = MetricSample(
            value=value,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.samples[metric_name].append(sample)
        
    def record_latency_batch(self, latencies_ms: List[float], batch_size: int = 1):
        """Record a batch of latency measurements"""
        for latency in latencies_ms:
            self.record_sample('latency_ms', latency, {'batch_size': batch_size})
    
    def calculate_latency_metrics(self, raw_latencies: List[float]) -> Dict[str, float]:
        """Calculate standard latency metrics from raw measurements"""
        if not raw_latencies:
            return {}
            
        return {
            'latency_mean_ms': statistics.mean(raw_latencies),
            'latency_median_ms': statistics.median(raw_latencies),
            'latency_p50_ms': np.percentile(raw_latencies, 50),
            'latency_p95_ms': np.percentile(raw_latencies, 95),
            'latency_p99_ms': np.percentile(raw_latencies, 99),
            'latency_std_ms': statistics.stdev(raw_latencies) if len(raw_latencies) > 1 else 0.0,
            'latency_min_ms': min(raw_latencies),
            'latency_max_ms': max(raw_latencies),
            'latency_count': len(raw_latencies)
        }
    
    def calculate_throughput_metrics(self, latencies_ms: List[float], batch_size: int = 1) -> Dict[str, float]:
        """Calculate throughput metrics from latency measurements"""
        if not latencies_ms:
            return {}
            
        mean_latency_s = statistics.mean(latencies_ms) / 1000.0
        
        if mean_latency_s <= 0:
            return {'throughput_fps': 0.0, 'throughput_ips': 0.0}
            
        # Throughput calculations
        throughput_per_batch = 1.0 / mean_latency_s
        throughput_per_sample = throughput_per_batch * batch_size
        
        return {
            'throughput_fps': throughput_per_sample,
            'throughput_ips': throughput_per_sample,
            'throughput_batch_fps': throughput_per_batch,
        }
    
    def calculate_power_metrics(self, power_samples_w: List[float], duration_s: float, num_inferences: int) -> Dict[str, float]:
        """Calculate power and energy metrics"""
        if not power_samples_w:
            return {}
            
        metrics = {
            'power_mean_w': statistics.mean(power_samples_w),
            'power_max_w': max(power_samples_w),
            'power_min_w': min(power_samples_w),
            'power_std_w': statistics.stdev(power_samples_w) if len(power_samples_w) > 1 else 0.0,
        }
        
        # Energy calculations
        if duration_s > 0 and num_inferences > 0:
            total_energy_j = metrics['power_mean_w'] * duration_s
            energy_per_inference_j = total_energy_j / num_inferences
            
            metrics.update({
                'energy_total_j': total_energy_j,
                'energy_per_inference_mj': energy_per_inference_j * 1000,
                'inferences_per_joule': 1.0 / energy_per_inference_j if energy_per_inference_j > 0 else 0.0,
                'inferences_per_watt': num_inferences / (metrics['power_mean_w'] * duration_s) if metrics['power_mean_w'] > 0 else 0.0
            })
            
        return metrics
    
    def calculate_accuracy_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray, top_k: List[int] = None) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        if top_k is None:
            top_k = [1, 5]
            
        metrics = {}
        
        for k in top_k:
            if k == 1:
                # Top-1 accuracy
                correct = np.sum(np.argmax(predictions, axis=1) == ground_truth)
                metrics[f'accuracy_top{k}'] = (correct / len(ground_truth)) * 100.0
            else:
                # Top-k accuracy  
                top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
                correct = np.sum([gt in pred_k for gt, pred_k in zip(ground_truth, top_k_preds)])
                metrics[f'accuracy_top{k}'] = (correct / len(ground_truth)) * 100.0
                
        return metrics
    
    def calculate_efficiency_metrics(self, latency_ms: float, power_w: float, accuracy_pct: float) -> Dict[str, float]:
        """Calculate efficiency composite metrics"""
        metrics = {}
        
        if latency_ms > 0:
            metrics['performance_fps_per_watt'] = (1000.0 / latency_ms) / power_w if power_w > 0 else 0.0
            
        if accuracy_pct > 0 and latency_ms > 0:
            metrics['accuracy_per_ms'] = accuracy_pct / latency_ms
            
        if accuracy_pct > 0 and power_w > 0:
            metrics['accuracy_per_watt'] = accuracy_pct / power_w
            
        # Composite efficiency score (higher is better)
        if all(v > 0 for v in [accuracy_pct, latency_ms, power_w]):
            metrics['efficiency_score'] = (accuracy_pct * 1000.0) / (latency_ms * power_w)
            
        return metrics
    
    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """Get definition for a metric"""
        return (self.STANDARD_METRICS.get(metric_name) or 
                self.custom_metrics.get(metric_name))
    
    def format_metric_value(self, metric_name: str, value: float) -> str:
        """Format metric value according to its definition"""
        metric_def = self.get_metric_definition(metric_name)
        
        if metric_def:
            decimal_places = metric_def.decimal_places
            unit = metric_def.unit
            formatted_value = f"{value:.{decimal_places}f}"
            return f"{formatted_value} {unit}" if unit else formatted_value
        else:
            return f"{value:.4f}"
    
    def compare_metrics(self, baseline: Dict[str, float], comparison: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Compare two sets of metrics and calculate improvements/regressions"""
        comparison_results = {}
        
        for metric_name in set(baseline.keys()) & set(comparison.keys()):
            baseline_val = baseline[metric_name]
            comparison_val = comparison[metric_name]
            
            if baseline_val == 0:
                continue
                
            metric_def = self.get_metric_definition(metric_name)
            higher_is_better = metric_def.higher_is_better if metric_def else True
            
            # Calculate percentage change
            pct_change = ((comparison_val - baseline_val) / baseline_val) * 100
            
            # Determine if this is an improvement
            is_improvement = (pct_change > 0) == higher_is_better
            
            comparison_results[metric_name] = {
                'baseline': baseline_val,
                'comparison': comparison_val,
                'absolute_change': comparison_val - baseline_val,
                'percent_change': pct_change,
                'is_improvement': is_improvement,
                'higher_is_better': higher_is_better,
            }
            
        return comparison_results
    
    def aggregate_metrics(self, metric_samples: Dict[str, List[float]]) -> Dict[str, float]:
        """Aggregate multiple metric samples into summary statistics"""
        aggregated = {}
        
        for metric_name, samples in metric_samples.items():
            if not samples:
                continue
                
            # Basic statistics
            aggregated[f"{metric_name}_mean"] = statistics.mean(samples)
            aggregated[f"{metric_name}_median"] = statistics.median(samples)
            aggregated[f"{metric_name}_min"] = min(samples)
            aggregated[f"{metric_name}_max"] = max(samples)
            
            if len(samples) > 1:
                aggregated[f"{metric_name}_std"] = statistics.stdev(samples)
                
                # Percentiles for latency-like metrics
                if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
                    aggregated[f"{metric_name}_p95"] = np.percentile(samples, 95)
                    aggregated[f"{metric_name}_p99"] = np.percentile(samples, 99)
            else:
                aggregated[f"{metric_name}_std"] = 0.0
                
        return aggregated
    
    def get_samples(self, metric_name: str) -> List[MetricSample]:
        """Get all samples for a specific metric"""
        return self.samples.get(metric_name, [])
    
    def clear_samples(self, metric_name: Optional[str] = None):
        """Clear samples for a specific metric or all metrics"""
        if metric_name:
            self.samples.pop(metric_name, None)
        else:
            self.samples.clear()
    
    def export_samples(self, metric_names: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Export samples in a serializable format"""
        export_data = {}
        
        metrics_to_export = metric_names or list(self.samples.keys())
        
        for metric_name in metrics_to_export:
            if metric_name in self.samples:
                export_data[metric_name] = [
                    {
                        'value': sample.value,
                        'timestamp': sample.timestamp,
                        'metadata': sample.metadata
                    }
                    for sample in self.samples[metric_name]
                ]
                
        return export_data