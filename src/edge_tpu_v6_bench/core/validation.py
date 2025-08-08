"""
Input validation and security module for Edge TPU v6 benchmarking
Comprehensive validation, sanitization, and security checks
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    sanitized_value: Optional[Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class SecurityValidator:
    """Security-focused validation for user inputs and file operations"""
    
    # Allowed file extensions for models
    ALLOWED_MODEL_EXTENSIONS = {'.tflite', '.pb', '.h5', '.onnx'}
    
    # Maximum file sizes (in bytes)
    MAX_MODEL_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_CONFIG_SIZE = 1 * 1024 * 1024   # 1MB
    
    # Path traversal protection patterns
    DANGEROUS_PATH_PATTERNS = [
        r'\.\./',  # Parent directory traversal
        r'\.\.\/',  # Alternative parent directory
        r'~/',     # Home directory
        r'/etc/',  # System directories
        r'/proc/', # Process filesystem
        r'/sys/',  # System filesystem
    ]
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate file path for security and accessibility
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validation result with security assessment
        """
        try:
            path = Path(file_path).resolve()
            path_str = str(path)
            
            # Check for path traversal attempts
            for pattern in SecurityValidator.DANGEROUS_PATH_PATTERNS:
                if re.search(pattern, str(file_path)):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Potentially unsafe path pattern detected: {pattern}"
                    )
            
            # Check if file exists
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File does not exist: {path}"
                )
            
            # Check if it's actually a file
            if not path.is_file():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Path is not a file: {path}"
                )
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > SecurityValidator.MAX_MODEL_SIZE:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large: {file_size} bytes (max: {SecurityValidator.MAX_MODEL_SIZE})"
                )
            
            # Check file extension
            if path.suffix.lower() not in SecurityValidator.ALLOWED_MODEL_EXTENSIONS:
                warnings = [f"Unusual file extension: {path.suffix}"]
            else:
                warnings = []
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=path,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Path validation error: {e}"
            )
    
    @staticmethod
    def validate_model_input(input_data: Any) -> ValidationResult:
        """
        Validate model input data for safety and format
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Validation result
        """
        try:
            if input_data is None:
                return ValidationResult(
                    is_valid=False,
                    error_message="Input data cannot be None"
                )
            
            # Convert to numpy array if not already
            if not isinstance(input_data, np.ndarray):
                try:
                    input_data = np.array(input_data)
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Cannot convert input to numpy array: {e}"
                    )
            
            # Check for reasonable array size (prevent memory attacks)
            max_elements = 100_000_000  # 100M elements max
            if input_data.size > max_elements:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Input array too large: {input_data.size} elements (max: {max_elements})"
                )
            
            # Check for valid numeric data
            if not np.isfinite(input_data).all():
                return ValidationResult(
                    is_valid=False,
                    error_message="Input contains non-finite values (NaN or Inf)"
                )
            
            # Check data type
            allowed_dtypes = {np.float32, np.float64, np.int8, np.int16, np.int32, np.uint8, np.uint16}
            if input_data.dtype.type not in allowed_dtypes:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported data type: {input_data.dtype}"
                )
            
            warnings = []
            
            # Warn about unusual shapes
            if len(input_data.shape) > 5:
                warnings.append(f"High-dimensional input: {len(input_data.shape)} dimensions")
            
            # Warn about unusual value ranges
            if input_data.dtype in {np.float32, np.float64}:
                if np.abs(input_data).max() > 1000:
                    warnings.append("Input values outside typical range [-1000, 1000]")
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=input_data,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Input validation error: {e}"
            )

class ConfigValidator:
    """Validator for configuration parameters"""
    
    @staticmethod
    def validate_benchmark_config(config: Dict[str, Any]) -> ValidationResult:
        """
        Validate benchmark configuration parameters
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validation result with sanitized config
        """
        try:
            sanitized_config = config.copy()
            warnings = []
            
            # Validate warmup_runs
            if 'warmup_runs' in config:
                warmup_runs = config['warmup_runs']
                if not isinstance(warmup_runs, int) or warmup_runs < 0:
                    return ValidationResult(
                        is_valid=False,
                        error_message="warmup_runs must be a non-negative integer"
                    )
                if warmup_runs > 1000:
                    warnings.append("Very high warmup_runs count may slow down benchmarking")
                    
            # Validate measurement_runs
            if 'measurement_runs' in config:
                measurement_runs = config['measurement_runs']
                if not isinstance(measurement_runs, int) or measurement_runs <= 0:
                    return ValidationResult(
                        is_valid=False,
                        error_message="measurement_runs must be a positive integer"
                    )
                if measurement_runs > 10000:
                    warnings.append("Very high measurement_runs count may take a long time")
            
            # Validate timeout_seconds
            if 'timeout_seconds' in config:
                timeout = config['timeout_seconds']
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    return ValidationResult(
                        is_valid=False,
                        error_message="timeout_seconds must be a positive number"
                    )
                if timeout > 3600:  # 1 hour
                    warnings.append("Very long timeout may cause issues")
            
            # Validate batch_sizes
            if 'batch_sizes' in config:
                batch_sizes = config['batch_sizes']
                if not isinstance(batch_sizes, list):
                    return ValidationResult(
                        is_valid=False,
                        error_message="batch_sizes must be a list"
                    )
                
                for batch_size in batch_sizes:
                    if not isinstance(batch_size, int) or batch_size <= 0:
                        return ValidationResult(
                            is_valid=False,
                            error_message="All batch sizes must be positive integers"
                        )
                    if batch_size > 1000:
                        warnings.append(f"Large batch size ({batch_size}) may exceed memory limits")
            
            # Validate concurrent_streams
            if 'concurrent_streams' in config:
                streams = config['concurrent_streams']
                if not isinstance(streams, int) or streams <= 0:
                    return ValidationResult(
                        is_valid=False,
                        error_message="concurrent_streams must be a positive integer"
                    )
                if streams > 32:
                    warnings.append("High concurrent_streams count may cause resource contention")
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized_config,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Configuration validation error: {e}"
            )
    
    @staticmethod
    def validate_device_specification(device_spec: Union[str, int]) -> ValidationResult:
        """
        Validate device specification
        
        Args:
            device_spec: Device specification to validate
            
        Returns:
            Validation result
        """
        try:
            if isinstance(device_spec, str):
                valid_strings = {'auto', 'edge_tpu_v6', 'edge_tpu_v5e', 'cpu', 'gpu'}
                if device_spec not in valid_strings:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid device string: {device_spec}. Valid options: {valid_strings}"
                    )
            elif isinstance(device_spec, int):
                if device_spec < 0:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Device index must be non-negative"
                    )
                if device_spec > 15:  # Reasonable upper bound
                    return ValidationResult(
                        is_valid=False,
                        error_message="Device index too high (max: 15)"
                    )
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message="Device specification must be string or integer"
                )
            
            return ValidationResult(is_valid=True, sanitized_value=device_spec)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Device specification validation error: {e}"
            )

class DataSanitizer:
    """Data sanitization utilities"""
    
    @staticmethod
    def sanitize_string_input(input_str: str, max_length: int = 1000) -> ValidationResult:
        """
        Sanitize string input to prevent injection attacks
        
        Args:
            input_str: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Validation result with sanitized string
        """
        try:
            if not isinstance(input_str, str):
                return ValidationResult(
                    is_valid=False,
                    error_message="Input must be a string"
                )
            
            if len(input_str) > max_length:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"String too long: {len(input_str)} chars (max: {max_length})"
                )
            
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>"\';\\]', '', input_str)
            
            # Remove excessive whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            warnings = []
            if sanitized != input_str:
                warnings.append("String was sanitized (removed potentially unsafe characters)")
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"String sanitization error: {e}"
            )
    
    @staticmethod
    def sanitize_numeric_input(value: Any, min_val: Optional[float] = None, 
                             max_val: Optional[float] = None) -> ValidationResult:
        """
        Sanitize numeric input with bounds checking
        
        Args:
            value: Numeric value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validation result with sanitized value
        """
        try:
            # Try to convert to float
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Cannot convert to numeric value: {value}"
                )
            
            # Check for special values
            if not np.isfinite(numeric_value):
                return ValidationResult(
                    is_valid=False,
                    error_message="Value must be finite (not NaN or Inf)"
                )
            
            # Check bounds
            warnings = []
            if min_val is not None and numeric_value < min_val:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Value {numeric_value} below minimum {min_val}"
                )
            
            if max_val is not None and numeric_value > max_val:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Value {numeric_value} above maximum {max_val}"
                )
            
            # Convert back to original type if it was integer
            if isinstance(value, int):
                sanitized_value = int(numeric_value)
            else:
                sanitized_value = numeric_value
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized_value,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Numeric sanitization error: {e}"
            )

def validate_and_sanitize(value: Any, validation_type: str, **kwargs) -> ValidationResult:
    """
    Main validation and sanitization entry point
    
    Args:
        value: Value to validate
        validation_type: Type of validation to perform
        **kwargs: Additional validation parameters
        
    Returns:
        Validation result
    """
    try:
        if validation_type == 'file_path':
            return SecurityValidator.validate_file_path(value)
        elif validation_type == 'model_input':
            return SecurityValidator.validate_model_input(value)
        elif validation_type == 'benchmark_config':
            return ConfigValidator.validate_benchmark_config(value)
        elif validation_type == 'device_spec':
            return ConfigValidator.validate_device_specification(value)
        elif validation_type == 'string':
            return DataSanitizer.sanitize_string_input(value, **kwargs)
        elif validation_type == 'numeric':
            return DataSanitizer.sanitize_numeric_input(value, **kwargs)
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown validation type: {validation_type}"
            )
            
    except Exception as e:
        logger.error(f"Validation error for type {validation_type}: {e}")
        return ValidationResult(
            is_valid=False,
            error_message=f"Validation system error: {e}"
        )