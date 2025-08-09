"""
Advanced auto-scaling system for Edge TPU v6 benchmarking
Intelligent scaling decisions, load prediction, and resource optimization
"""

import time
import threading
import logging
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import concurrent.futures

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingTrigger(Enum):
    """Scaling trigger types"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM_METRIC = "custom_metric"

@dataclass
class ScalingRule:
    """Configuration for scaling rules"""
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_adjustment: int = 1
    scale_down_adjustment: int = 1
    cooldown_period: float = 300.0  # 5 minutes
    evaluation_periods: int = 2
    last_scaling_action: float = 0.0
    enabled: bool = True

@dataclass
class ScalingEvent:
    """Record of a scaling event"""
    timestamp: float
    direction: ScalingDirection
    trigger: ScalingTrigger
    old_size: int
    new_size: int
    trigger_value: float
    threshold: float
    reason: str

@dataclass
class LoadPrediction:
    """Load prediction result"""
    predicted_load: float
    confidence: float
    time_horizon: float
    recommendation: ScalingDirection

class MetricCollector:
    """Collects and analyzes system metrics for scaling decisions"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics: Dict[ScalingTrigger, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.lock = threading.RLock()
    
    def add_metric(self, trigger: ScalingTrigger, value: float, timestamp: Optional[float] = None):
        """Add metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.metrics[trigger].append((timestamp, value))
    
    def get_recent_average(self, trigger: ScalingTrigger, window_seconds: float = 300.0) -> Optional[float]:
        """Get average value for recent window"""
        cutoff_time = time.time() - window_seconds
        
        with self.lock:
            recent_values = [
                value for timestamp, value in self.metrics[trigger]
                if timestamp > cutoff_time
            ]
            
            if recent_values:
                return statistics.mean(recent_values)
            return None
    
    def get_trend(self, trigger: ScalingTrigger, window_seconds: float = 600.0) -> Optional[float]:
        """Get trend direction for metric (-1 to 1, negative is decreasing)"""
        cutoff_time = time.time() - window_seconds
        
        with self.lock:
            recent_data = [
                (timestamp, value) for timestamp, value in self.metrics[trigger]
                if timestamp > cutoff_time
            ]
            
            if len(recent_data) < 2:
                return None
            
            # Simple linear regression for trend
            times = [t - cutoff_time for t, _ in recent_data]
            values = [v for _, v in recent_data]
            
            n = len(times)
            sum_t = sum(times)
            sum_v = sum(values)
            sum_tv = sum(t * v for t, v in zip(times, values))
            sum_t2 = sum(t * t for t in times)
            
            # Calculate slope
            denominator = n * sum_t2 - sum_t * sum_t
            if abs(denominator) < 1e-10:
                return 0.0
            
            slope = (n * sum_tv - sum_t * sum_v) / denominator
            
            # Normalize slope to [-1, 1] range
            max_value = max(values) if values else 1.0
            normalized_slope = slope / max(max_value, 1.0)
            
            return max(-1.0, min(1.0, normalized_slope))

class LoadPredictor:
    """Predicts future load based on historical patterns"""
    
    def __init__(self):
        self.seasonal_patterns: Dict[int, List[float]] = defaultdict(list)  # hour -> loads
        self.recent_trends: deque = deque(maxlen=100)
        self.lock = threading.RLock()
    
    def update_load_data(self, load: float, timestamp: Optional[float] = None):
        """Update load data for prediction"""
        if timestamp is None:
            timestamp = time.time()
        
        hour = int((timestamp % 86400) // 3600)  # Hour of day
        
        with self.lock:
            self.seasonal_patterns[hour].append(load)
            if len(self.seasonal_patterns[hour]) > 100:  # Keep bounded
                self.seasonal_patterns[hour].pop(0)
            
            self.recent_trends.append((timestamp, load))
    
    def predict_load(self, time_horizon: float = 300.0) -> LoadPrediction:
        """Predict load for given time horizon"""
        future_time = time.time() + time_horizon
        future_hour = int((future_time % 86400) // 3600)
        
        with self.lock:
            # Seasonal prediction based on historical patterns
            seasonal_loads = self.seasonal_patterns.get(future_hour, [])
            seasonal_prediction = statistics.mean(seasonal_loads) if seasonal_loads else 0.0
            
            # Trend-based prediction
            trend_prediction = self._predict_trend(time_horizon)
            
            # Combine predictions
            if seasonal_loads and len(self.recent_trends) > 5:
                # Weight both predictions
                combined_load = 0.6 * seasonal_prediction + 0.4 * trend_prediction
                confidence = min(len(seasonal_loads) / 50.0, 1.0)  # More data = higher confidence
            elif seasonal_loads:
                combined_load = seasonal_prediction
                confidence = min(len(seasonal_loads) / 100.0, 0.8)
            elif len(self.recent_trends) > 5:
                combined_load = trend_prediction
                confidence = 0.5
            else:
                combined_load = 0.0
                confidence = 0.1
            
            # Determine recommendation
            current_load = self.recent_trends[-1][1] if self.recent_trends else 0.0
            
            if combined_load > current_load * 1.2:
                recommendation = ScalingDirection.UP
            elif combined_load < current_load * 0.8:
                recommendation = ScalingDirection.DOWN
            else:
                recommendation = ScalingDirection.STABLE
            
            return LoadPrediction(
                predicted_load=combined_load,
                confidence=confidence,
                time_horizon=time_horizon,
                recommendation=recommendation
            )
    
    def _predict_trend(self, time_horizon: float) -> float:
        """Predict load based on recent trend"""
        if len(self.recent_trends) < 2:
            return 0.0
        
        # Linear extrapolation based on recent trend
        recent_data = list(self.recent_trends)[-20:]  # Last 20 points
        
        if len(recent_data) < 2:
            return recent_data[-1][1] if recent_data else 0.0
        
        # Calculate trend slope
        times = [t for t, _ in recent_data]
        loads = [l for _, l in recent_data]
        
        time_span = times[-1] - times[0]
        if time_span < 1.0:
            return loads[-1]
        
        load_change = loads[-1] - loads[0]
        slope = load_change / time_span
        
        # Extrapolate
        predicted_load = loads[-1] + slope * time_horizon
        
        return max(0.0, predicted_load)

class AutoScaler:
    """
    Intelligent auto-scaling system
    
    Features:
    - Multiple scaling triggers and rules
    - Load prediction and proactive scaling
    - Cooldown periods and stability controls
    - Comprehensive event logging and metrics
    - Integration with resource pools and monitoring
    """
    
    def __init__(self,
                 min_capacity: int = 1,
                 max_capacity: int = 100,
                 target_resource: Any = None,
                 scaling_function: Optional[Callable[[int, int], bool]] = None):
        """
        Initialize auto-scaler
        
        Args:
            min_capacity: Minimum allowed capacity
            max_capacity: Maximum allowed capacity
            target_resource: Target resource to scale
            scaling_function: Function to perform actual scaling
        """
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.current_capacity = min_capacity
        self.target_resource = target_resource
        self.scaling_function = scaling_function
        
        # Scaling rules
        self.scaling_rules: Dict[ScalingTrigger, ScalingRule] = {}
        self._setup_default_rules()
        
        # Monitoring and prediction
        self.metric_collector = MetricCollector()
        self.load_predictor = LoadPredictor()
        
        # Event tracking
        self.scaling_events: List[ScalingEvent] = []
        self.scaling_lock = threading.RLock()
        
        # Background processing
        self.running = False
        self.evaluation_thread: Optional[threading.Thread] = None
        self.evaluation_interval = 60.0  # 1 minute
        
        logger.info(f"AutoScaler initialized: capacity range [{min_capacity}-{max_capacity}]")
    
    def _setup_default_rules(self):
        """Setup default scaling rules"""
        
        # CPU utilization rule
        self.add_scaling_rule(ScalingRule(
            trigger=ScalingTrigger.CPU_UTILIZATION,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
            scale_up_adjustment=2,
            scale_down_adjustment=1,
            cooldown_period=300.0,
            evaluation_periods=2
        ))
        
        # Memory utilization rule
        self.add_scaling_rule(ScalingRule(
            trigger=ScalingTrigger.MEMORY_UTILIZATION,
            scale_up_threshold=80.0,
            scale_down_threshold=40.0,
            scale_up_adjustment=1,
            scale_down_adjustment=1,
            cooldown_period=600.0,
            evaluation_periods=3
        ))
        
        # Response time rule
        self.add_scaling_rule(ScalingRule(
            trigger=ScalingTrigger.RESPONSE_TIME,
            scale_up_threshold=1000.0,  # 1 second
            scale_down_threshold=200.0,  # 200ms
            scale_up_adjustment=2,
            scale_down_adjustment=1,
            cooldown_period=300.0,
            evaluation_periods=2
        ))
        
        # Error rate rule
        self.add_scaling_rule(ScalingRule(
            trigger=ScalingTrigger.ERROR_RATE,
            scale_up_threshold=5.0,  # 5% error rate
            scale_down_threshold=1.0,  # 1% error rate
            scale_up_adjustment=1,
            scale_down_adjustment=0,  # Don't scale down on error rate
            cooldown_period=180.0,
            evaluation_periods=1  # React quickly to errors
        ))
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add or update scaling rule"""
        self.scaling_rules[rule.trigger] = rule
        logger.info(f"Added scaling rule for {rule.trigger.value}")
    
    def start(self):
        """Start auto-scaling evaluation"""
        if self.running:
            logger.warning("AutoScaler already running")
            return
        
        self.running = True
        self.evaluation_thread = threading.Thread(
            target=self._evaluation_loop,
            daemon=True,
            name="AutoScaler-Evaluation"
        )
        self.evaluation_thread.start()
        logger.info("AutoScaler started")
    
    def stop(self):
        """Stop auto-scaling evaluation"""
        if not self.running:
            return
        
        self.running = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=10.0)
        
        logger.info("AutoScaler stopped")
    
    def _evaluation_loop(self):
        """Main evaluation loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Evaluate scaling rules
                scaling_decision = self._evaluate_scaling_rules()
                
                # Consider load predictions
                prediction = self.load_predictor.predict_load(300.0)  # 5 minutes ahead
                
                # Apply predictive scaling if high confidence
                if prediction.confidence > 0.7:
                    predictive_decision = self._evaluate_predictive_scaling(prediction)
                    if predictive_decision != ScalingDirection.STABLE:
                        scaling_decision = predictive_decision
                
                # Execute scaling decision
                if scaling_decision != ScalingDirection.STABLE:
                    self._execute_scaling_decision(scaling_decision)
                
                # Sleep until next evaluation
                elapsed = time.time() - start_time
                sleep_time = max(0, self.evaluation_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"AutoScaler evaluation error: {e}")
                time.sleep(min(60, self.evaluation_interval))
    
    def _evaluate_scaling_rules(self) -> ScalingDirection:
        """Evaluate all scaling rules and determine action"""
        scale_up_votes = 0
        scale_down_votes = 0
        current_time = time.time()
        
        for trigger, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if current_time - rule.last_scaling_action < rule.cooldown_period:
                continue
            
            # Get recent metric average
            recent_value = self.metric_collector.get_recent_average(
                trigger, window_seconds=rule.evaluation_periods * 60
            )
            
            if recent_value is None:
                continue
            
            # Update load predictor with current data
            if trigger in [ScalingTrigger.CPU_UTILIZATION, ScalingTrigger.THROUGHPUT]:
                self.load_predictor.update_load_data(recent_value)
            
            # Evaluate scaling conditions
            if recent_value > rule.scale_up_threshold:
                scale_up_votes += 1
                logger.debug(f"Scale up vote from {trigger.value}: {recent_value} > {rule.scale_up_threshold}")
            elif recent_value < rule.scale_down_threshold:
                scale_down_votes += 1
                logger.debug(f"Scale down vote from {trigger.value}: {recent_value} < {rule.scale_down_threshold}")
        
        # Determine overall decision
        if scale_up_votes > 0 and scale_up_votes >= scale_down_votes:
            return ScalingDirection.UP
        elif scale_down_votes > 0 and scale_down_votes > scale_up_votes:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def _evaluate_predictive_scaling(self, prediction: LoadPrediction) -> ScalingDirection:
        """Evaluate predictive scaling based on load prediction"""
        current_load = self.metric_collector.get_recent_average(ScalingTrigger.CPU_UTILIZATION, 60.0)
        
        if current_load is None:
            return ScalingDirection.STABLE
        
        # Scale proactively if predicted load change is significant
        load_change_threshold = 20.0  # 20% change
        
        if prediction.predicted_load > current_load + load_change_threshold:
            logger.info(f"Predictive scale up: predicted {prediction.predicted_load:.1f}% vs current {current_load:.1f}%")
            return ScalingDirection.UP
        elif prediction.predicted_load < current_load - load_change_threshold:
            logger.info(f"Predictive scale down: predicted {prediction.predicted_load:.1f}% vs current {current_load:.1f}%")
            return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _execute_scaling_decision(self, direction: ScalingDirection):
        """Execute scaling decision"""
        with self.scaling_lock:
            old_capacity = self.current_capacity
            
            # Calculate new capacity
            if direction == ScalingDirection.UP:
                # Find the rule that triggered scaling up
                adjustment = max([r.scale_up_adjustment for r in self.scaling_rules.values() if r.enabled], default=1)
                new_capacity = min(self.max_capacity, old_capacity + adjustment)
                
            else:  # ScalingDirection.DOWN
                # Find the rule that triggered scaling down
                adjustment = max([r.scale_down_adjustment for r in self.scaling_rules.values() if r.enabled], default=1)
                new_capacity = max(self.min_capacity, old_capacity - adjustment)
            
            # Check if scaling is needed
            if new_capacity == old_capacity:
                return
            
            # Attempt scaling
            success = self._perform_scaling(old_capacity, new_capacity)
            
            if success:
                self.current_capacity = new_capacity
                
                # Update rule last action times
                current_time = time.time()
                for rule in self.scaling_rules.values():
                    rule.last_scaling_action = current_time
                
                # Record event
                event = ScalingEvent(
                    timestamp=current_time,
                    direction=direction,
                    trigger=ScalingTrigger.CPU_UTILIZATION,  # Placeholder
                    old_size=old_capacity,
                    new_size=new_capacity,
                    trigger_value=0.0,  # Placeholder
                    threshold=0.0,  # Placeholder
                    reason=f"Auto-scaled {direction.value} from {old_capacity} to {new_capacity}"
                )
                
                self.scaling_events.append(event)
                
                # Keep event history bounded
                if len(self.scaling_events) > 1000:
                    self.scaling_events.pop(0)
                
                logger.info(f"Scaled {direction.value}: {old_capacity} -> {new_capacity}")
            else:
                logger.error(f"Failed to scale {direction.value} from {old_capacity} to {new_capacity}")
    
    def _perform_scaling(self, old_capacity: int, new_capacity: int) -> bool:
        """Perform actual scaling operation"""
        try:
            if self.scaling_function:
                return self.scaling_function(old_capacity, new_capacity)
            else:
                # Default scaling implementation
                logger.info(f"Mock scaling: {old_capacity} -> {new_capacity}")
                return True
                
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")
            return False
    
    def add_metric(self, trigger: ScalingTrigger, value: float):
        """Add metric value for scaling evaluation"""
        self.metric_collector.add_metric(trigger, value)
    
    def manual_scale(self, new_capacity: int, reason: str = "Manual scaling") -> bool:
        """Manually scale to specific capacity"""
        if new_capacity < self.min_capacity or new_capacity > self.max_capacity:
            logger.error(f"Capacity {new_capacity} outside allowed range [{self.min_capacity}-{self.max_capacity}]")
            return False
        
        with self.scaling_lock:
            old_capacity = self.current_capacity
            
            if old_capacity == new_capacity:
                return True
            
            success = self._perform_scaling(old_capacity, new_capacity)
            
            if success:
                direction = ScalingDirection.UP if new_capacity > old_capacity else ScalingDirection.DOWN
                
                event = ScalingEvent(
                    timestamp=time.time(),
                    direction=direction,
                    trigger=ScalingTrigger.CUSTOM_METRIC,
                    old_size=old_capacity,
                    new_size=new_capacity,
                    trigger_value=0.0,
                    threshold=0.0,
                    reason=reason
                )
                
                self.scaling_events.append(event)
                self.current_capacity = new_capacity
                
                logger.info(f"Manual scale: {old_capacity} -> {new_capacity} ({reason})")
            
            return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaler statistics"""
        recent_events = [e for e in self.scaling_events 
                        if time.time() - e.timestamp < 3600]  # Last hour
        
        scale_up_events = len([e for e in recent_events 
                              if e.direction == ScalingDirection.UP])
        scale_down_events = len([e for e in recent_events 
                                if e.direction == ScalingDirection.DOWN])
        
        # Get current metric values
        current_metrics = {}
        for trigger in ScalingTrigger:
            value = self.metric_collector.get_recent_average(trigger, 60.0)
            if value is not None:
                current_metrics[trigger.value] = value
        
        # Load prediction
        prediction = self.load_predictor.predict_load(300.0)
        
        return {
            'current_capacity': self.current_capacity,
            'min_capacity': self.min_capacity,
            'max_capacity': self.max_capacity,
            'total_scaling_events': len(self.scaling_events),
            'recent_scale_up_events': scale_up_events,
            'recent_scale_down_events': scale_down_events,
            'current_metrics': current_metrics,
            'prediction': {
                'predicted_load': prediction.predicted_load,
                'confidence': prediction.confidence,
                'recommendation': prediction.recommendation.value
            },
            'rules_enabled': len([r for r in self.scaling_rules.values() if r.enabled]),
            'is_running': self.running
        }
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling event history"""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            {
                'timestamp': event.timestamp,
                'direction': event.direction.value,
                'trigger': event.trigger.value,
                'old_size': event.old_size,
                'new_size': event.new_size,
                'trigger_value': event.trigger_value,
                'threshold': event.threshold,
                'reason': event.reason
            }
            for event in self.scaling_events
            if event.timestamp > cutoff_time
        ]

# Global auto-scaler instance
global_auto_scaler = AutoScaler()