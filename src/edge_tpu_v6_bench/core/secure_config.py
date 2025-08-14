"""
Secure Configuration Management
Enhanced security configuration with environment variable support
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib
import secrets

logger = logging.getLogger(__name__)

@dataclass
class SecureConfig:
    """
    Secure configuration management with environment variable support
    """
    
    # Security settings
    enable_security_validation: bool = True
    max_file_size_mb: int = 100
    allowed_file_extensions: set = field(default_factory=lambda: {'.tflite', '.pb', '.json', '.txt'})
    
    # API keys and secrets (load from environment)
    api_key: Optional[str] = field(default=None)
    encryption_key: Optional[str] = field(default=None)
    
    # Paths and directories
    secure_temp_dir: Path = field(default_factory=lambda: Path('/tmp/edge_tpu_secure'))
    log_directory: Path = field(default_factory=lambda: Path('./logs'))
    
    def __post_init__(self):
        """Initialize secure configuration from environment variables"""
        # Load sensitive configuration from environment
        self.api_key = os.environ.get('EDGE_TPU_API_KEY')
        self.encryption_key = os.environ.get('EDGE_TPU_ENCRYPTION_KEY')
        
        # Generate secure encryption key if not provided
        if not self.encryption_key:
            self.encryption_key = secrets.token_urlsafe(32)
            logger.warning("Generated temporary encryption key. Set EDGE_TPU_ENCRYPTION_KEY environment variable for production.")
        
        # Ensure secure directories exist
        self.secure_temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log_directory.mkdir(parents=True, exist_ok=True, mode=0o755)
    
    def validate_file_path(self, file_path: Path) -> bool:
        """
        Validate file path for security compliance
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if path is secure
        """
        try:
            # Resolve path to prevent traversal attacks
            resolved_path = file_path.resolve()
            
            # Check for path traversal attempts
            if '..' in str(file_path) or str(resolved_path).startswith('/etc/') or str(resolved_path).startswith('/proc/'):
                logger.warning(f"Potentially unsafe path detected: {file_path}")
                return False
            
            # Check file extension
            if file_path.suffix not in self.allowed_file_extensions:
                logger.warning(f"File extension not allowed: {file_path.suffix}")
                return False
            
            # Check file size if it exists
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    logger.warning(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file path {file_path}: {e}")
            return False
    
    def get_secure_hash(self, data: str) -> str:
        """
        Generate secure hash of data using SHA-256
        
        Args:
            data: Data to hash
            
        Returns:
            Secure hash string
        """
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def sanitize_input(self, user_input: str) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            user_input: Raw user input
            
        Returns:
            Sanitized input
        """
        if not isinstance(user_input, str):
            return str(user_input)
        
        # Remove potentially dangerous characters
        dangerous_patterns = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')']
        sanitized = user_input
        
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
            logger.warning("Input truncated to 1000 characters")
        
        return sanitized.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)"""
        return {
            'enable_security_validation': self.enable_security_validation,
            'max_file_size_mb': self.max_file_size_mb,
            'allowed_file_extensions': list(self.allowed_file_extensions),
            'secure_temp_dir': str(self.secure_temp_dir),
            'log_directory': str(self.log_directory),
            # Note: API keys and encryption keys are NOT included for security
        }

# Global secure configuration instance
secure_config = SecureConfig()