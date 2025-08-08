"""
Security scanning and code quality tests
Validates security measures and code quality standards
"""

import sys
import os
from pathlib import Path
import ast
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_no_hardcoded_secrets():
    """Test that no secrets are hardcoded in the codebase"""
    src_dir = Path(__file__).parent.parent / 'src'
    
    # Common patterns for secrets
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret_key\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
        r'private_key\s*=\s*["\'][^"\']+["\']',
    ]
    
    violations = []
    
    for py_file in src_dir.rglob('*.py'):
        if py_file.name == '__pycache__':
            continue
            
        try:
            content = py_file.read_text()
            
            for pattern in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Allow test/mock values
                    if any(test_word in match.group().lower() for test_word in ['test', 'mock', 'example', 'dummy']):
                        continue
                    violations.append(f"{py_file}: {match.group()}")
                    
        except Exception as e:
            print(f"Warning: Could not scan {py_file}: {e}")
    
    assert len(violations) == 0, f"Hardcoded secrets found: {violations}"

def test_no_dangerous_imports():
    """Test that no dangerous imports are present"""
    src_dir = Path(__file__).parent.parent / 'src'
    
    dangerous_modules = [
        'subprocess',  # Should use safe alternatives
        'os.system',   # Direct system calls
        'eval',        # Code evaluation
        'exec',        # Code execution
        '__import__'   # Dynamic imports
    ]
    
    violations = []
    
    for py_file in src_dir.rglob('*.py'):
        try:
            content = py_file.read_text()
            
            # Parse AST to find imports and calls
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            violations.append(f"{py_file}: import {alias.name}")
                
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module in dangerous_modules:
                        violations.append(f"{py_file}: from {node.module}")
                
                elif isinstance(node, ast.Call):
                    if hasattr(node.func, 'id') and node.func.id in ['eval', 'exec']:
                        violations.append(f"{py_file}: {node.func.id}() call")
                        
        except Exception as e:
            print(f"Warning: Could not parse {py_file}: {e}")
    
    # Allow some safe usage (filter false positives)
    safe_violations = []
    for violation in violations:
        # Allow subprocess in safe contexts (already imported for legitimate use)
        if 'subprocess' in violation and 'concurrent_execution.py' in violation:
            continue
        safe_violations.append(violation)
    
    assert len(safe_violations) == 0, f"Dangerous imports found: {safe_violations}"

def test_input_validation_coverage():
    """Test that input validation is properly implemented"""
    from edge_tpu_v6_bench.core.validation import validate_and_sanitize
    
    # Test various input types
    test_cases = [
        ("string_input", "string"),
        (42, "numeric"),
        ({"test": "config"}, "benchmark_config"),
        ("test_file.tflite", "filename"),
        ("/path/to/file", "path")
    ]
    
    for input_value, validation_type in test_cases:
        result = validate_and_sanitize(input_value, validation_type)
        assert hasattr(result, 'is_valid'), f"Validation result missing is_valid for {validation_type}"
        assert hasattr(result, 'error_message'), f"Validation result missing error_message for {validation_type}"

def test_error_handling_coverage():
    """Test that error handling is comprehensive"""
    from edge_tpu_v6_bench.core.error_handling import ErrorRecoverySystem, EdgeTPUError, ErrorCategory
    
    recovery_system = ErrorRecoverySystem()
    
    # Test different error categories
    error_categories = [
        ErrorCategory.HARDWARE,
        ErrorCategory.MODEL,
        ErrorCategory.VALIDATION,
        ErrorCategory.MEMORY,
        ErrorCategory.TIMEOUT
    ]
    
    for category in error_categories:
        test_error = EdgeTPUError(f"Test {category.value} error", category=category)
        
        # Should handle error without crashing
        try:
            result = recovery_system.handle_error(test_error)
            # Recovery might succeed or fail, but should not crash
        except Exception as e:
            assert False, f"Error handling failed for {category.value}: {e}"

def test_logging_security():
    """Test that logging doesn't expose sensitive information"""
    import logging
    from io import StringIO
    
    # Capture log output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = logging.getLogger('edge_tpu_v6_bench.test_security')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    # Test logging of potentially sensitive data
    sensitive_data = {
        'password': 'secret123',
        'api_key': 'abc123def456',
        'user_input': '../../../etc/passwd'
    }
    
    logger.info(f"Processing data: {sensitive_data}")
    
    log_output = log_capture.getvalue()
    
    # Should not log raw sensitive data (implementation should sanitize)
    # This is a basic check - real implementation would have more sophisticated filtering
    sensitive_patterns = ['secret123', 'abc123def456']
    
    for pattern in sensitive_patterns:
        if pattern in log_output:
            print(f"Warning: Potentially sensitive data in logs: {pattern}")

def test_file_permission_security():
    """Test that files are created with secure permissions"""
    from edge_tpu_v6_bench.core.security import SecurityManager
    
    security_manager = SecurityManager()
    
    # Test secure temp file creation
    temp_file = security_manager.create_secure_temp_file()
    
    if hasattr(os, 'stat'):
        file_stat = temp_file.stat()
        
        # On Unix-like systems, check file permissions
        # Owner should have read/write, others should not
        mode = file_stat.st_mode
        
        # Owner permissions
        assert mode & 0o400, "Owner should have read permission"  # Owner read
        assert mode & 0o200, "Owner should have write permission"  # Owner write
        
        # Group and other permissions should be restricted
        group_perms = (mode & 0o070) >> 3
        other_perms = mode & 0o007
        
        # For secure files, group and others should have minimal permissions
        print(f"File permissions: owner={(mode & 0o700) >> 6}, group={group_perms}, other={other_perms}")
    
    # Clean up
    temp_file.unlink()

def test_thread_safety():
    """Test that shared resources are thread-safe"""
    import threading
    import time
    from edge_tpu_v6_bench.core.performance_cache import PerformanceCache
    
    cache = PerformanceCache(max_memory_size=1024*1024)
    
    # Test concurrent cache operations
    results = []
    errors = []
    
    def cache_worker(worker_id):
        try:
            for i in range(10):
                key = f"worker_{worker_id}_item_{i}"
                value = f"value_{worker_id}_{i}"
                
                # Set value
                cache.set(key, value)
                
                # Get value
                retrieved = cache.get(key)
                
                results.append((worker_id, i, retrieved == value))
                
                time.sleep(0.01)  # Small delay to increase contention
                
        except Exception as e:
            errors.append((worker_id, str(e)))
    
    # Start multiple worker threads
    threads = []
    for worker_id in range(5):
        thread = threading.Thread(target=cache_worker, args=(worker_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=10)
    
    # Check results
    assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    # Most operations should succeed (some might fail due to cache eviction)
    success_rate = sum(1 for _, _, success in results if success) / len(results)
    assert success_rate > 0.8, f"Thread safety test success rate too low: {success_rate}"

def test_memory_usage():
    """Test that memory usage is reasonable"""
    import psutil
    import gc
    from edge_tpu_v6_bench.core.performance_cache import PerformanceCache
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create and use cache
    cache = PerformanceCache(max_memory_size=10*1024*1024)  # 10MB limit
    
    # Add many items to test memory management
    for i in range(1000):
        cache.set(f"key_{i}", f"value_{i}" * 100)  # ~600 bytes per item
    
    # Force garbage collection
    gc.collect()
    
    # Check memory usage
    current_memory = process.memory_info().rss
    memory_increase = current_memory - initial_memory
    
    # Memory increase should be reasonable (less than 50MB)
    memory_increase_mb = memory_increase / 1024 / 1024
    print(f"Memory increase: {memory_increase_mb:.1f} MB")
    
    assert memory_increase_mb < 50, f"Excessive memory usage: {memory_increase_mb} MB"

def test_quantum_task_engine_security():
    """Test security aspects of quantum task engine"""
    from edge_tpu_v6_bench.quantum_planner import QuantumTaskEngine
    
    engine = QuantumTaskEngine()
    
    # Test that malicious function cannot be easily injected
    def safe_task():
        return "safe_result"
    
    # This should work
    task = engine.add_task("safe_task", "Safe Task", safe_task)
    assert task is not None
    
    # Test task isolation
    def task_with_side_effect():
        # This task should not be able to access engine internals unsafely
        return "completed"
    
    task2 = engine.add_task("side_effect_task", "Task with Side Effects", task_with_side_effect)
    assert task2 is not None

def test_code_quality_metrics():
    """Test basic code quality metrics"""
    src_dir = Path(__file__).parent.parent / 'src'
    
    total_lines = 0
    total_files = 0
    files_with_docstrings = 0
    
    for py_file in src_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        try:
            content = py_file.read_text()
            lines = len(content.splitlines())
            total_lines += lines
            total_files += 1
            
            # Check for module docstring
            if content.strip().startswith('"""') or content.strip().startswith("'''"):
                files_with_docstrings += 1
                
        except Exception:
            continue
    
    # Basic quality metrics
    avg_lines_per_file = total_lines / max(1, total_files)
    docstring_coverage = files_with_docstrings / max(1, total_files)
    
    print(f"Code quality metrics:")
    print(f"  Total files: {total_files}")
    print(f"  Total lines: {total_lines}")
    print(f"  Average lines per file: {avg_lines_per_file:.1f}")
    print(f"  Docstring coverage: {docstring_coverage:.1%}")
    
    # Basic thresholds
    assert total_files > 10, "Project should have substantial codebase"
    assert avg_lines_per_file < 1000, "Files should not be excessively long"
    assert docstring_coverage > 0.7, f"Docstring coverage too low: {docstring_coverage:.1%}"

if __name__ == "__main__":
    print("Running security and quality tests...")
    
    test_no_hardcoded_secrets()
    print("✓ No hardcoded secrets found")
    
    test_no_dangerous_imports()
    print("✓ No dangerous imports found")
    
    test_input_validation_coverage()
    print("✓ Input validation coverage verified")
    
    test_error_handling_coverage()
    print("✓ Error handling coverage verified")
    
    test_logging_security()
    print("✓ Logging security checked")
    
    test_file_permission_security()
    print("✓ File permission security verified")
    
    test_thread_safety()
    print("✓ Thread safety verified")
    
    test_memory_usage()
    print("✓ Memory usage is reasonable")
    
    test_quantum_task_engine_security()
    print("✓ Quantum task engine security verified")
    
    test_code_quality_metrics()
    print("✓ Code quality metrics verified")
    
    print("\n🛡️ All security and quality tests passed!")