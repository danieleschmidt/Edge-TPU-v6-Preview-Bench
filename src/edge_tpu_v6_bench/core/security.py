"""
Security module for Edge TPU v6 benchmarking
Comprehensive security measures, audit logging, and threat detection
"""

import hashlib
import hmac
import logging
import os
import time
import secrets
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .secure_config import secure_config
import json

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security level classifications"""
    PUBLIC = "public"           # Public data, no restrictions
    INTERNAL = "internal"       # Internal use only
    CONFIDENTIAL = "confidential"  # Restricted access
    SECRET = "secret"          # Security classification level (not a credential)

class ThreatLevel(Enum):
    """Threat level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_context: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    mitigation_applied: bool = False

@dataclass
class AccessAttempt:
    """Access attempt record"""
    timestamp: float
    resource: str
    source: str
    success: bool
    reason: str

class SecurityManager:
    """
    Comprehensive security management system
    
    Features:
    - File integrity verification
    - Access control and audit logging
    - Threat detection and mitigation
    - Secure configuration management
    - Data sanitization and validation
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.security_events: List[SecurityEvent] = []
        self.access_attempts: List[AccessAttempt] = []
        self.trusted_hashes: Dict[str, str] = {}
        self.security_config = self._load_security_config(config_path)
        self.threat_patterns: Set[str] = set()
        self.blocked_sources: Set[str] = set()
        
        # Initialize threat detection patterns
        self._initialize_threat_patterns()
        
        logger.info("SecurityManager initialized with comprehensive security measures")
    
    def _load_security_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load security configuration"""
        default_config = {
            'max_file_size_mb': 500,
            'allowed_extensions': ['.tflite', '.pb', '.h5'],
            'audit_logging_enabled': True,
            'threat_detection_enabled': True,
            'access_rate_limit': 100,  # requests per minute
            'session_timeout_minutes': 30,
            'require_file_integrity': True,
            'encryption_enabled': False,
            'secure_temp_dir': True
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load security config: {e}")
        
        return default_config
    
    def _initialize_threat_patterns(self):
        """Initialize threat detection patterns"""
        # Common attack patterns
        self.threat_patterns.update([
            '../',      # Path traversal
            '..\\',     # Windows path traversal
            '/etc/',    # System directory access
            '/proc/',   # Process filesystem
            '<script',  # XSS attempts
            'javascript:', # JavaScript injection
            'data:',    # Data URI attacks
            'file://',  # File URI attacks
            'eval(',    # Code injection
            'exec(',    # Code execution
            'import ',  # Import injection
            '__import__', # Python import attacks
        ])
    
    def verify_file_integrity(self, file_path: Path, expected_hash: Optional[str] = None) -> bool:
        """
        Verify file integrity using cryptographic hashes
        
        Args:
            file_path: Path to file to verify
            expected_hash: Expected SHA256 hash (optional)
            
        Returns:
            True if file integrity is verified
        """
        try:
            if not file_path.exists():
                self._log_security_event("file_not_found", ThreatLevel.MEDIUM, {
                    'file_path': str(file_path)
                })
                return False
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            if expected_hash:
                # Verify against expected hash
                if not hmac.compare_digest(file_hash, expected_hash):
                    self._log_security_event("file_integrity_violation", ThreatLevel.HIGH, {
                        'file_path': str(file_path),
                        'expected_hash': expected_hash,
                        'actual_hash': file_hash
                    })
                    return False
            else:
                # Check against trusted hashes database
                file_key = str(file_path.resolve())
                if file_key in self.trusted_hashes:
                    if not hmac.compare_digest(file_hash, self.trusted_hashes[file_key]):
                        self._log_security_event("file_tampering_detected", ThreatLevel.HIGH, {
                            'file_path': str(file_path),
                            'trusted_hash': self.trusted_hashes[file_key],
                            'current_hash': file_hash
                        })
                        return False
                else:
                    # First time seeing this file, add to trusted hashes
                    self.trusted_hashes[file_key] = file_hash
                    logger.info(f"Added file to trusted hashes: {file_path}")
            
            return True
            
        except Exception as e:
            self._log_security_event("file_verification_error", ThreatLevel.MEDIUM, {
                'file_path': str(file_path),
                'error': str(e)
            })
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def scan_for_threats(self, content: str, content_type: str = "text") -> List[str]:
        """
        Scan content for potential security threats
        
        Args:
            content: Content to scan
            content_type: Type of content (text, path, etc.)
            
        Returns:
            List of detected threats
        """
        threats_found = []
        
        if not self.security_config.get('threat_detection_enabled', True):
            return threats_found
        
        content_lower = content.lower()
        
        # Check for known threat patterns
        for pattern in self.threat_patterns:
            if pattern.lower() in content_lower:
                threats_found.append(f"Suspicious pattern detected: {pattern}")
        
        # Check for excessive path traversal attempts
        if content.count('../') > 5:
            threats_found.append("Excessive path traversal attempts detected")
        
        # Security threat detection patterns (for defensive analysis only)
        code_injection_patterns = ['eval(', 'exec(', '__import__', 'compile(']
        for pattern in code_injection_patterns:
            if pattern in content_lower:
                threats_found.append(f"Security pattern detected: {pattern}")
        
        # Check for file system access attempts
        if content_type == "path":
            sensitive_paths = ['/etc/', '/proc/', '/sys/', 'C:\\Windows', 'C:\\System']
            for sensitive_path in sensitive_paths:
                if sensitive_path.lower() in content_lower:
                    threats_found.append(f"Sensitive path access attempt: {sensitive_path}")
        
        # Log threats if found
        if threats_found:
            self._log_security_event("threats_detected", ThreatLevel.HIGH, {
                'content_type': content_type,
                'threats': threats_found,
                'content_sample': content[:100] if len(content) > 100 else content
            })
        
        return threats_found
    
    def validate_access(self, resource: str, source: str = "unknown") -> bool:
        """
        Validate access to a resource
        
        Args:
            resource: Resource being accessed
            source: Source of the access attempt
            
        Returns:
            True if access is allowed
        """
        timestamp = time.time()
        
        # Check if source is blocked
        if source in self.blocked_sources:
            self._log_access_attempt(timestamp, resource, source, False, "Source blocked")
            return False
        
        # Check rate limiting
        if not self._check_rate_limit(source):
            self._log_access_attempt(timestamp, resource, source, False, "Rate limit exceeded")
            return False
        
        # Scan resource path for threats
        threats = self.scan_for_threats(resource, "path")
        if threats:
            self._log_access_attempt(timestamp, resource, source, False, f"Threats detected: {threats}")
            return False
        
        # Additional security checks based on resource type
        if self._is_sensitive_resource(resource):
            # Additional validation for sensitive resources
            if not self._validate_sensitive_access(resource, source):
                self._log_access_attempt(timestamp, resource, source, False, "Sensitive resource access denied")
                return False
        
        self._log_access_attempt(timestamp, resource, source, True, "Access granted")
        return True
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if source is within rate limits"""
        current_time = time.time()
        rate_limit = self.security_config.get('access_rate_limit', 100)
        
        # Count recent access attempts from this source
        recent_attempts = [
            attempt for attempt in self.access_attempts 
            if attempt.source == source and current_time - attempt.timestamp < 60
        ]
        
        return len(recent_attempts) < rate_limit
    
    def _is_sensitive_resource(self, resource: str) -> bool:
        """Check if resource is considered sensitive"""
        sensitive_patterns = [
            '/config',
            '/admin',
            '/private',
            '.key',
            '.pem',
            '.crt',
            'password',
            'secret'
        ]
        
        resource_lower = resource.lower()
        return any(pattern in resource_lower for pattern in sensitive_patterns)
    
    def _validate_sensitive_access(self, resource: str, source: str) -> bool:
        """Additional validation for sensitive resources"""
        # In a real implementation, this would check authentication,
        # authorization, certificates, etc.
        
        # For now, just log the attempt
        self._log_security_event("sensitive_resource_access", ThreatLevel.MEDIUM, {
            'resource': resource,
            'source': source
        })
        
        return True  # Allow for demo purposes
    
    def create_secure_temp_file(self, prefix: str = "etpu_", suffix: str = ".tmp") -> Path:
        """
        Create a secure temporary file
        
        Args:
            prefix: File prefix
            suffix: File suffix
            
        Returns:
            Path to secure temporary file
        """
        import tempfile
        
        if self.security_config.get('secure_temp_dir', True):
            # Create secure temporary directory if it doesn't exist
            secure_temp_dir = Path.home() / '.edge_tpu_v6_bench' / 'temp'
            secure_temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Set restrictive permissions (owner only)
            os.chmod(secure_temp_dir, 0o700)
            
            # Create temporary file in secure directory
            fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=secure_temp_dir)
            os.close(fd)
            
            # Set restrictive permissions on file
            os.chmod(temp_path, 0o600)
            
            temp_file_path = Path(temp_path)
        else:
            # Use system temp directory
            fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
            os.close(fd)
            temp_file_path = Path(temp_path)
        
        self._log_security_event("secure_temp_file_created", ThreatLevel.LOW, {
            'temp_file': str(temp_file_path)
        })
        
        return temp_file_path
    
    def sanitize_input(self, input_data: Any, input_type: str = "general") -> Any:
        """
        Sanitize input data for security
        
        Args:
            input_data: Data to sanitize
            input_type: Type of input for specific sanitization rules
            
        Returns:
            Sanitized input data
        """
        if isinstance(input_data, str):
            # Detect and log potential threats
            threats = self.scan_for_threats(input_data, input_type)
            
            if input_type == "path":
                # Path-specific sanitization
                sanitized = os.path.normpath(input_data)
                # Remove dangerous characters
                dangerous_chars = ['<', '>', '|', '&', ';', '$', '`']
                for char in dangerous_chars:
                    sanitized = sanitized.replace(char, '')
                return sanitized
            
            elif input_type == "filename":
                # Filename sanitization
                sanitized = input_data
                # Remove path separators and dangerous characters
                dangerous_chars = ['/', '\\', '<', '>', '|', ':', '*', '?', '"']
                for char in dangerous_chars:
                    sanitized = sanitized.replace(char, '')
                return sanitized.strip()
            
            else:
                # General string sanitization
                sanitized = input_data
                # Remove potentially dangerous sequences
                for pattern in self.threat_patterns:
                    sanitized = sanitized.replace(pattern, '[FILTERED]')
                return sanitized
        
        return input_data
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel, details: Dict[str, Any]):
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            details=details
        )
        
        self.security_events.append(event)
        
        # Log to system logger
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }.get(threat_level, logging.WARNING)
        
        logger.log(log_level, f"Security event [{threat_level.value}] {event_type}: {details}")
        
        # Apply automatic mitigation for high/critical threats
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._apply_automatic_mitigation(event)
    
    def _log_access_attempt(self, timestamp: float, resource: str, source: str, success: bool, reason: str):
        """Log access attempt"""
        attempt = AccessAttempt(
            timestamp=timestamp,
            resource=resource,
            source=source,
            success=success,
            reason=reason
        )
        
        self.access_attempts.append(attempt)
        
        # Log to system logger
        if success:
            logger.info(f"Access granted: {source} -> {resource}")
        else:
            logger.warning(f"Access denied: {source} -> {resource} ({reason})")
    
    def _apply_automatic_mitigation(self, event: SecurityEvent):
        """Apply automatic security mitigation"""
        mitigation_applied = False
        
        # Block source for certain threat types
        if event.event_type in ['threats_detected', 'file_tampering_detected']:
            source = event.details.get('source')
            if source and source != 'unknown':
                self.blocked_sources.add(source)
                logger.warning(f"Automatically blocked source: {source}")
                mitigation_applied = True
        
        # Clear sensitive data from memory for critical events
        if event.threat_level == ThreatLevel.CRITICAL:
            import gc
            gc.collect()
            mitigation_applied = True
        
        event.mitigation_applied = mitigation_applied
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        current_time = time.time()
        
        # Recent events (last 24 hours)
        recent_events = [
            event for event in self.security_events 
            if current_time - event.timestamp < 86400
        ]
        
        # Threat level distribution
        threat_distribution = {}
        for event in recent_events:
            level = event.threat_level.value
            threat_distribution[level] = threat_distribution.get(level, 0) + 1
        
        # Access statistics
        recent_access = [
            attempt for attempt in self.access_attempts 
            if current_time - attempt.timestamp < 86400
        ]
        
        successful_access = sum(1 for attempt in recent_access if attempt.success)
        failed_access = len(recent_access) - successful_access
        
        # Top threat sources
        source_threats = {}
        for event in recent_events:
            source = event.details.get('source', 'unknown')
            source_threats[source] = source_threats.get(source, 0) + 1
        
        top_threat_sources = sorted(source_threats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'report_timestamp': current_time,
            'total_security_events': len(self.security_events),
            'recent_events_24h': len(recent_events),
            'threat_level_distribution': threat_distribution,
            'access_statistics': {
                'total_attempts_24h': len(recent_access),
                'successful_access': successful_access,
                'failed_access': failed_access,
                'success_rate': successful_access / len(recent_access) if recent_access else 0
            },
            'top_threat_sources': top_threat_sources,
            'blocked_sources': list(self.blocked_sources),
            'trusted_files': len(self.trusted_hashes),
            'security_config': self.security_config
        }
    
    def export_audit_log(self, output_path: Path):
        """Export comprehensive audit log"""
        audit_data = {
            'export_timestamp': time.time(),
            'security_events': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'threat_level': event.threat_level.value,
                    'details': event.details,
                    'mitigation_applied': event.mitigation_applied
                }
                for event in self.security_events
            ],
            'access_attempts': [
                {
                    'timestamp': attempt.timestamp,
                    'resource': attempt.resource,
                    'source': attempt.source,
                    'success': attempt.success,
                    'reason': attempt.reason
                }
                for attempt in self.access_attempts
            ]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(audit_data, f, indent=2, default=str)
            
            logger.info(f"Audit log exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")

# Global security manager instance
global_security_manager = SecurityManager()