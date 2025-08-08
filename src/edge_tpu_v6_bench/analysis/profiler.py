"""
Latency Profiler for detailed performance analysis
Provides layer-wise profiling and optimization suggestions
"""

import time
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class LayerProfile:
    """Profile information for a single layer"""
    layer_name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    latency_ms: float
    memory_mb: float
    ops_count: int
    utilization_percent: float

@dataclass 
class ModelProfile:
    """Complete model profiling results"""
    model_name: str
    total_latency_ms: float
    total_memory_mb: float
    total_ops: int
    layer_profiles: List[LayerProfile] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    optimization_suggestions: List[Dict[str, Any]] = field(default_factory=list)

class LatencyProfiler:
    """
    Advanced latency profiler for Edge TPU models
    
    Features:
    - Layer-wise performance analysis
    - Bottleneck identification
    - Optimization recommendations
    - Comparative analysis across device types
    """
    
    def __init__(self, device: str = 'edge_tpu_v6'):
        self.device = device
        self.profiles: Dict[str, ModelProfile] = {}
        logger.info(f"LatencyProfiler initialized for {device}")
    
    def profile_model(self, 
                     model,
                     input_shape: Tuple[int, ...],
                     n_runs: int = 1000,
                     warmup_runs: int = 100) -> ModelProfile:
        """
        Profile model with detailed layer-wise analysis
        
        Args:
            model: TensorFlow Lite model or interpreter
            input_shape: Input tensor shape
            n_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Detailed model profile
        """
        logger.info(f"Profiling model with {n_runs} runs, {warmup_runs} warmup")
        
        # Generate dummy input data
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(warmup_runs):
            self._single_inference(model, dummy_input)
        
        # Profile individual layers (mock implementation for now)
        layer_profiles = self._profile_layers(model, dummy_input, n_runs)
        
        # Calculate total metrics
        total_latency = sum(layer.latency_ms for layer in layer_profiles)
        total_memory = sum(layer.memory_mb for layer in layer_profiles)
        total_ops = sum(layer.ops_count for layer in layer_profiles)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(layer_profiles)
        
        # Generate optimization suggestions
        suggestions = self._generate_suggestions(layer_profiles)
        
        profile = ModelProfile(
            model_name=f"model_{id(model)}",
            total_latency_ms=total_latency,
            total_memory_mb=total_memory,
            total_ops=total_ops,
            layer_profiles=layer_profiles,
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions
        )
        
        self.profiles[profile.model_name] = profile
        logger.info(f"Profile complete: {total_latency:.2f}ms total latency")
        
        return profile
    
    def _single_inference(self, model, input_data: np.ndarray) -> np.ndarray:
        """Run single inference (mock implementation)"""
        # In real implementation, this would use TensorFlow Lite interpreter
        time.sleep(0.001)  # Simulate inference time
        return np.random.randn(1, 1000)  # Mock output
    
    def _profile_layers(self, model, input_data: np.ndarray, n_runs: int) -> List[LayerProfile]:
        """Profile individual layers (mock implementation)"""
        # Mock layer profiling - in real implementation would use TF Lite profiling API
        mock_layers = [
            ("input", "InputLayer", (1, 224, 224, 3), (1, 224, 224, 3), 0.1),
            ("conv2d_1", "Conv2D", (1, 224, 224, 3), (1, 112, 112, 32), 2.5),
            ("bn_1", "BatchNormalization", (1, 112, 112, 32), (1, 112, 112, 32), 0.3),
            ("relu_1", "ReLU", (1, 112, 112, 32), (1, 112, 112, 32), 0.2),
            ("conv2d_2", "Conv2D", (1, 112, 112, 32), (1, 56, 56, 64), 4.2),
            ("global_pool", "GlobalAveragePooling2D", (1, 56, 56, 64), (1, 64), 0.5),
            ("dense", "Dense", (1, 64), (1, 1000), 0.8),
            ("softmax", "Softmax", (1, 1000), (1, 1000), 0.1)
        ]
        
        profiles = []
        for name, layer_type, in_shape, out_shape, base_latency in mock_layers:
            # Add some random variation
            latency_variation = np.random.normal(1.0, 0.1)
            latency_ms = base_latency * latency_variation
            
            # Estimate memory usage
            memory_mb = (np.prod(out_shape) * 4) / (1024 * 1024)  # 4 bytes per float32
            
            # Estimate operations count
            if layer_type == "Conv2D":
                ops_count = int(np.prod(out_shape) * 9)  # Rough estimate for 3x3 conv
            elif layer_type == "Dense":
                ops_count = int(in_shape[-1] * out_shape[-1])
            else:
                ops_count = int(np.prod(out_shape))
            
            # Mock utilization
            utilization = np.random.uniform(70, 95)
            
            profiles.append(LayerProfile(
                layer_name=name,
                layer_type=layer_type,
                input_shape=in_shape,
                output_shape=out_shape,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                ops_count=ops_count,
                utilization_percent=utilization
            ))
        
        return profiles
    
    def _identify_bottlenecks(self, layer_profiles: List[LayerProfile]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        total_latency = sum(layer.latency_ms for layer in layer_profiles)
        
        for layer in layer_profiles:
            # Layer consuming more than 20% of total time is a bottleneck
            if layer.latency_ms / total_latency > 0.20:
                bottlenecks.append(f"{layer.layer_name} ({layer.layer_type}): "
                                 f"{layer.latency_ms:.2f}ms "
                                 f"({layer.latency_ms/total_latency*100:.1f}% of total)")
        
        return bottlenecks
    
    def _generate_suggestions(self, layer_profiles: List[LayerProfile]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions"""
        suggestions = []
        
        for layer in layer_profiles:
            if layer.layer_type == "Conv2D" and layer.latency_ms > 2.0:
                suggestions.append({
                    'layer': layer.layer_name,
                    'type': 'quantization',
                    'description': f"Consider INT8 quantization for {layer.layer_name}",
                    'potential_speedup': '1.5-2.0x',
                    'accuracy_impact': 'Minimal (<1%)'
                })
            
            if layer.utilization_percent < 80:
                suggestions.append({
                    'layer': layer.layer_name,
                    'type': 'optimization',
                    'description': f"Low utilization ({layer.utilization_percent:.1f}%) in {layer.layer_name}",
                    'potential_speedup': '1.2x',
                    'accuracy_impact': 'None'
                })
        
        # Global suggestions
        conv_layers = [l for l in layer_profiles if l.layer_type == "Conv2D"]
        if len(conv_layers) > 3:
            suggestions.append({
                'layer': 'model',
                'type': 'architecture',
                'description': 'Consider depthwise separable convolutions',
                'potential_speedup': '2.0-3.0x',
                'accuracy_impact': 'Moderate (2-5%)'
            })
        
        return suggestions
    
    def plot_layer_latency(self, profile: ModelProfile):
        """Plot layer-wise latency breakdown"""
        try:
            import matplotlib.pyplot as plt
            
            layer_names = [layer.layer_name for layer in profile.layer_profiles]
            latencies = [layer.latency_ms for layer in profile.layer_profiles]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(layer_names, latencies)
            plt.title(f'Layer Latency Breakdown - {profile.model_name}')
            plt.xlabel('Layer')
            plt.ylabel('Latency (ms)')
            plt.xticks(rotation=45)
            
            # Color bottleneck layers differently
            bottleneck_layers = [b.split(' ')[0] for b in profile.bottlenecks]
            for i, bar in enumerate(bars):
                if layer_names[i] in bottleneck_layers:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    def plot_operation_breakdown(self, profile: ModelProfile):
        """Plot operation type breakdown"""
        try:
            import matplotlib.pyplot as plt
            
            # Group by layer type
            type_latencies = {}
            for layer in profile.layer_profiles:
                if layer.layer_type not in type_latencies:
                    type_latencies[layer.layer_type] = 0
                type_latencies[layer.layer_type] += layer.latency_ms
            
            plt.figure(figsize=(8, 8))
            plt.pie(type_latencies.values(), labels=type_latencies.keys(), autopct='%1.1f%%')
            plt.title(f'Operation Type Breakdown - {profile.model_name}')
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    def suggest_optimizations(self, profile: ModelProfile) -> List[str]:
        """Get optimization suggestions as formatted strings"""
        suggestions = []
        
        for suggestion in profile.optimization_suggestions:
            suggestions.append(
                f"• {suggestion['description']}\n"
                f"  Potential speedup: {suggestion['potential_speedup']}\n"
                f"  Accuracy impact: {suggestion['accuracy_impact']}"
            )
        
        return suggestions
    
    def compare_profiles(self, profile1: ModelProfile, profile2: ModelProfile) -> Dict[str, Any]:
        """Compare two model profiles"""
        comparison = {
            'latency_improvement': profile1.total_latency_ms / profile2.total_latency_ms,
            'memory_difference_mb': profile2.total_memory_mb - profile1.total_memory_mb,
            'ops_difference': profile2.total_ops - profile1.total_ops,
            'bottleneck_changes': {
                'added': set(profile2.bottlenecks) - set(profile1.bottlenecks),
                'removed': set(profile1.bottlenecks) - set(profile2.bottlenecks)
            }
        }
        
        return comparison