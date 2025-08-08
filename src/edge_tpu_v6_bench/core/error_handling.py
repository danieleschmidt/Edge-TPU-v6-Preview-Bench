"""
Comprehensive error handling and recovery system for Edge TPU v6 benchmarking
Robust error handling, recovery strategies, and failure diagnostics
"""

import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"  # System failure, cannot continue
    HIGH = "high"         # Major feature failure, can continue with degraded functionality
    MEDIUM = "medium"     # Minor feature failure, workarounds available
    LOW = "low"          # Warning-level issues, minimal impact

class ErrorCategory(Enum):
    """Categories of errors for better handling"""
    HARDWARE = "hardware"           # Device/hardware related errors
    MODEL = "model"                # Model loading/format errors
    VALIDATION = "validation"      # Input validation errors
    PERFORMANCE = "performance"    # Benchmark execution errors
    MEMORY = "memory"             # Out of memory errors
    TIMEOUT = "timeout"           # Timeout errors
    NETWORK = "network"           # Network/communication errors
    FILESYSTEM = "filesystem"     # File system errors
    CONFIGURATION = "configuration" # Configuration errors
    UNKNOWN = "unknown"           # Uncategorized errors

@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class RecoveryStrategy:
    """Recovery strategy for specific error types"""
    strategy_name: str
    applicable_categories: List[ErrorCategory]
    recovery_function: Callable
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    success_rate: float = 0.0
    total_attempts: int = 0
    successful_recoveries: int = 0

class EdgeTPUError(Exception):
    """Base exception class for Edge TPU benchmarking errors"""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

class HardwareError(EdgeTPUError):
    """Hardware-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.HARDWARE, 
                        severity=ErrorSeverity.HIGH, **kwargs)

class ModelError(EdgeTPUError):
    """Model-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL, 
                        severity=ErrorSeverity.HIGH, **kwargs)

class ValidationError(EdgeTPUError):
    """Validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, 
                        severity=ErrorSeverity.MEDIUM, **kwargs)

class PerformanceError(EdgeTPUError):
    """Performance benchmark errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.PERFORMANCE, 
                        severity=ErrorSeverity.MEDIUM, **kwargs)

class TimeoutError(EdgeTPUError):
    """Timeout errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, 
                        severity=ErrorSeverity.MEDIUM, **kwargs)

class ErrorRecoverySystem:
    """
    Comprehensive error recovery and handling system
    
    Features:
    - Automatic error categorization and severity assessment
    - Context-aware recovery strategies  
    - Retry mechanisms with exponential backoff
    - Error pattern analysis and learning
    - Graceful degradation capabilities
    """
    
    def __init__(self):
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {}
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._setup_default_strategies()
        
        logger.info("ErrorRecoverySystem initialized with default recovery strategies")
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies for common error types"""
        
        # Hardware recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            strategy_name="device_reconnect",
            applicable_categories=[ErrorCategory.HARDWARE],
            recovery_function=self._recover_device_connection,
            max_attempts=3,
            backoff_seconds=2.0
        ))
        
        self.add_recovery_strategy(RecoveryStrategy(
            strategy_name="device_reset",
            applicable_categories=[ErrorCategory.HARDWARE],
            recovery_function=self._recover_device_reset,
            max_attempts=2,
            backoff_seconds=5.0
        ))
        
        # Memory recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            strategy_name="garbage_collection",
            applicable_categories=[ErrorCategory.MEMORY],
            recovery_function=self._recover_memory_cleanup,
            max_attempts=2,
            backoff_seconds=0.5
        ))
        
        self.add_recovery_strategy(RecoveryStrategy(
            strategy_name="reduce_batch_size",
            applicable_categories=[ErrorCategory.MEMORY],
            recovery_function=self._recover_reduce_batch_size,
            max_attempts=3,
            backoff_seconds=0.0
        ))
        
        # Performance recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            strategy_name="reduce_complexity",
            applicable_categories=[ErrorCategory.PERFORMANCE],
            recovery_function=self._recover_reduce_complexity,
            max_attempts=2,
            backoff_seconds=0.0
        ))
        
        # Timeout recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            strategy_name="increase_timeout",
            applicable_categories=[ErrorCategory.TIMEOUT],
            recovery_function=self._recover_increase_timeout,
            max_attempts=2,
            backoff_seconds=0.0
        ))
        
        # Model recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            strategy_name="fallback_device",
            applicable_categories=[ErrorCategory.MODEL, ErrorCategory.HARDWARE],
            recovery_function=self._recover_fallback_device,
            max_attempts=1,
            backoff_seconds=0.0
        ))
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy to the system"""
        for category in strategy.applicable_categories:
            if category not in self.recovery_strategies:
                self.recovery_strategies[category] = []
            self.recovery_strategies[category].append(strategy)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Handle error with automatic recovery attempts
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            Recovery result if successful, None if all recovery attempts failed
        """
        error_context = self._create_error_context(error, context)
        self.error_history.append(error_context)
        
        # Update error patterns
        error_key = f"{error_context.category.value}:{type(error).__name__}"
        self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1
        
        logger.error(f"Handling error: {error_context.severity.value} {error_context.category.value} - {error_context.message}")
        
        # Check circuit breaker
        if self._is_circuit_broken(error_context.category):
            logger.warning(f"Circuit breaker open for {error_context.category.value}, skipping recovery")
            return None
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_context)
        
        if recovery_result is not None:
            logger.info(f"Successfully recovered from {error_context.category.value} error")
            self._update_circuit_breaker(error_context.category, success=True)
        else:
            logger.error(f"All recovery attempts failed for {error_context.category.value} error")
            self._update_circuit_breaker(error_context.category, success=False)
        
        return recovery_result
    
    def _create_error_context(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorContext:
        """Create error context from exception"""
        error_id = f"err_{int(time.time() * 1000)}"
        
        # Determine category and severity
        if isinstance(error, EdgeTPUError):
            category = error.category
            severity = error.severity
            details = error.context.copy()
        else:
            category = self._categorize_error(error)
            severity = self._assess_severity(error)
            details = {}
        
        if context:
            details.update(context)
        
        # Generate recovery suggestions
        suggestions = self._generate_recovery_suggestions(error, category)
        
        return ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(error),
            details=details,
            traceback=traceback.format_exc(),
            recovery_suggestions=suggestions
        )
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on exception type and message"""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Hardware-related keywords
        hardware_keywords = ['device', 'tpu', 'hardware', 'coral', 'edgetpu', 'usb']
        if any(keyword in error_msg or keyword in error_type for keyword in hardware_keywords):
            return ErrorCategory.HARDWARE
        
        # Memory-related keywords
        memory_keywords = ['memory', 'allocation', 'oom', 'out of memory']
        if any(keyword in error_msg or keyword in error_type for keyword in memory_keywords):
            return ErrorCategory.MEMORY
        
        # Model-related keywords
        model_keywords = ['model', 'tflite', 'interpreter', 'graph', 'tensor']
        if any(keyword in error_msg or keyword in error_type for keyword in model_keywords):
            return ErrorCategory.MODEL
        
        # Timeout-related
        if 'timeout' in error_msg or 'timeout' in error_type:
            return ErrorCategory.TIMEOUT
        
        # Validation-related
        if 'validation' in error_msg or 'invalid' in error_msg:
            return ErrorCategory.VALIDATION
        
        # Performance-related
        performance_keywords = ['benchmark', 'performance', 'inference']
        if any(keyword in error_msg for keyword in performance_keywords):
            return ErrorCategory.PERFORMANCE
        
        # File system related
        filesystem_keywords = ['file', 'path', 'directory', 'permission']
        if any(keyword in error_msg for keyword in filesystem_keywords):
            return ErrorCategory.FILESYSTEM
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess error severity based on type and impact"""
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        # Critical errors that stop execution
        critical_keywords = ['system', 'fatal', 'abort', 'crash']
        if any(keyword in error_msg for keyword in critical_keywords):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_severity_types = ['runtimeerror', 'systemexit', 'keyboardinterrupt']
        if error_type in high_severity_types:
            return ErrorSeverity.HIGH
        
        # Memory errors are typically high severity
        if 'memory' in error_msg or 'allocation' in error_msg:
            return ErrorSeverity.HIGH
        
        # Device errors are high severity
        if 'device' in error_msg or 'hardware' in error_msg:
            return ErrorSeverity.HIGH
        
        # Medium severity for most runtime issues
        medium_severity_types = ['valueerror', 'typeerror', 'indexerror']
        if error_type in medium_severity_types:
            return ErrorSeverity.MEDIUM
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    def _generate_recovery_suggestions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate human-readable recovery suggestions"""
        suggestions = []
        
        if category == ErrorCategory.HARDWARE:
            suggestions.extend([
                "Check device connection and drivers",
                "Try reconnecting the Edge TPU device",
                "Verify device permissions",
                "Use fallback device (CPU/GPU)"
            ])
        
        elif category == ErrorCategory.MEMORY:
            suggestions.extend([
                "Reduce batch size",
                "Clear memory cache",
                "Close unused applications",
                "Use smaller model variant"
            ])
        
        elif category == ErrorCategory.MODEL:
            suggestions.extend([
                "Verify model file integrity",
                "Check model format compatibility",
                "Try re-quantizing the model",
                "Use alternative model"
            ])
        
        elif category == ErrorCategory.TIMEOUT:
            suggestions.extend([
                "Increase timeout duration",
                "Reduce workload complexity",
                "Check system load",
                "Use faster device"
            ])
        
        elif category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Check input data format",
                "Verify configuration parameters",
                "Review file paths",
                "Validate input ranges"
            ])
        
        return suggestions
    
    def _attempt_recovery(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt recovery using available strategies"""
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in strategies:
            logger.info(f"Attempting recovery strategy: {strategy.strategy_name}")
            
            strategy.total_attempts += 1
            
            try:
                # Attempt recovery with backoff
                for attempt in range(strategy.max_attempts):
                    if attempt > 0:
                        time.sleep(strategy.backoff_seconds * (2 ** attempt))  # Exponential backoff
                    
                    recovery_result = strategy.recovery_function(error_context)
                    
                    if recovery_result is not None:
                        strategy.successful_recoveries += 1
                        strategy.success_rate = strategy.successful_recoveries / strategy.total_attempts
                        return recovery_result
                
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.strategy_name} failed: {recovery_error}")
                continue
        
        return None
    
    def _is_circuit_broken(self, category: ErrorCategory) -> bool:
        """Check if circuit breaker is open for given category"""
        circuit_key = category.value
        circuit = self.circuit_breakers.get(circuit_key, {'failures': 0, 'last_failure': 0, 'open': False})
        
        # Circuit breaker logic: open if too many recent failures
        if circuit['open'] and time.time() - circuit['last_failure'] > 300:  # 5 minutes
            circuit['open'] = False
            circuit['failures'] = 0
        
        return circuit['open']
    
    def _update_circuit_breaker(self, category: ErrorCategory, success: bool):
        """Update circuit breaker state"""
        circuit_key = category.value
        
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {'failures': 0, 'last_failure': 0, 'open': False}
        
        circuit = self.circuit_breakers[circuit_key]
        
        if success:
            circuit['failures'] = 0
            circuit['open'] = False
        else:
            circuit['failures'] += 1
            circuit['last_failure'] = time.time()
            
            # Open circuit if too many failures
            if circuit['failures'] >= 5:
                circuit['open'] = True
                logger.warning(f"Circuit breaker opened for {category.value} after {circuit['failures']} failures")
    
    # Recovery strategy implementations
    def _recover_device_connection(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt to recover device connection"""
        logger.info("Attempting device reconnection...")
        # Mock implementation - in real code would attempt device reconnection
        time.sleep(1.0)
        return "device_reconnected"
    
    def _recover_device_reset(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt device reset"""
        logger.info("Attempting device reset...")
        # Mock implementation - in real code would reset device
        time.sleep(2.0)
        return "device_reset"
    
    def _recover_memory_cleanup(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt memory cleanup"""
        logger.info("Performing memory cleanup...")
        import gc
        gc.collect()
        return "memory_cleaned"
    
    def _recover_reduce_batch_size(self, error_context: ErrorContext) -> Optional[Any]:
        """Reduce batch size to fit in memory"""
        current_batch = error_context.details.get('batch_size', 32)
        new_batch = max(1, current_batch // 2)
        logger.info(f"Reducing batch size from {current_batch} to {new_batch}")
        return {'batch_size': new_batch}
    
    def _recover_reduce_complexity(self, error_context: ErrorContext) -> Optional[Any]:
        """Reduce benchmark complexity"""
        logger.info("Reducing benchmark complexity...")
        return {'reduced_complexity': True, 'measurement_runs': 10}
    
    def _recover_increase_timeout(self, error_context: ErrorContext) -> Optional[Any]:
        """Increase timeout duration"""
        current_timeout = error_context.details.get('timeout', 300)
        new_timeout = current_timeout * 2
        logger.info(f"Increasing timeout from {current_timeout} to {new_timeout}")
        return {'timeout': new_timeout}
    
    def _recover_fallback_device(self, error_context: ErrorContext) -> Optional[Any]:
        """Fallback to alternative device"""
        logger.info("Falling back to CPU device...")
        return {'device': 'cpu'}
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {'total_errors': 0}
        
        # Error distribution by category
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Recovery success rates
        recovery_rates = {}
        for category, strategies in self.recovery_strategies.items():
            total_attempts = sum(s.total_attempts for s in strategies)
            successful_recoveries = sum(s.successful_recoveries for s in strategies)
            
            if total_attempts > 0:
                recovery_rates[category.value] = successful_recoveries / total_attempts
        
        return {
            'total_errors': total_errors,
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'recovery_success_rates': recovery_rates,
            'most_common_patterns': sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        }

# Decorator for automatic error handling
def handle_errors(recovery_system: Optional[ErrorRecoverySystem] = None,
                 reraise: bool = False,
                 default_return: Any = None):
    """
    Decorator for automatic error handling with recovery
    
    Args:
        recovery_system: Error recovery system to use
        reraise: Whether to re-raise exceptions after recovery attempts
        default_return: Default return value if recovery fails
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if recovery_system:
                    recovery_result = recovery_system.handle_error(e, {
                        'function': func.__name__,
                        'args': str(args)[:200],  # Truncate for logging
                        'kwargs': str(kwargs)[:200]
                    })
                    
                    if recovery_result is not None:
                        # Recovery successful, retry function with recovered parameters
                        if isinstance(recovery_result, dict):
                            kwargs.update(recovery_result)
                        return func(*args, **kwargs)
                
                if reraise:
                    raise
                
                logger.error(f"Unrecoverable error in {func.__name__}: {e}")
                return default_return
        
        return wrapper
    return decorator

# Global error recovery system instance
global_recovery_system = ErrorRecoverySystem()