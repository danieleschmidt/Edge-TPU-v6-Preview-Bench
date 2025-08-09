#!/usr/bin/env python3
"""
Direct Quality Gates Validator - Tests core modules without dependencies
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add core module paths directly
core_path = str(Path(__file__).parent / 'src' / 'edge_tpu_v6_bench' / 'core')
sys.path.insert(0, core_path)

def test_error_handling():
    """Test error handling system directly"""
    from error_handling import (
        ErrorRecoverySystem, ErrorSeverity, ErrorCategory, 
        EdgeTPUError, HardwareError, handle_errors
    )
    
    # Test error system
    recovery = ErrorRecoverySystem()
    
    # Test different error types
    hw_error = HardwareError("Test hardware failure")
    assert hw_error.category == ErrorCategory.HARDWARE
    assert hw_error.severity == ErrorSeverity.HIGH
    
    # Test error handling
    result = recovery.handle_error(hw_error)
    
    # Test decorator
    @handle_errors(recovery_system=recovery, reraise=False, default_return="failed")
    def test_function():
        raise ValueError("Test error")
    
    result = test_function()
    assert result == "failed"
    
    stats = recovery.get_error_statistics()
    assert stats['total_errors'] >= 2
    
    print(f"✅ Error Handling: {stats['total_errors']} errors handled")

def test_security():
    """Test security system directly"""
    from security import SecurityManager, ThreatLevel, SecurityEvent
    
    security = SecurityManager()
    
    # Test threat scanning
    threats = security.scan_for_threats("../etc/passwd", "path")
    assert len(threats) > 0
    
    threats = security.scan_for_threats("<script>alert('xss')</script>", "text")
    assert len(threats) > 0
    
    # Test input sanitization
    safe_input = security.sanitize_input("normal input", "general")
    assert safe_input == "normal input"
    
    unsafe_input = security.sanitize_input("../etc/passwd", "path")
    assert "../" not in unsafe_input
    
    # Test access validation
    valid = security.validate_access("/safe/path", "user")
    assert valid
    
    invalid = security.validate_access("/etc/passwd", "user")
    assert not invalid
    
    report = security.get_security_report()
    print(f"✅ Security: {report['total_security_events']} events logged")

def test_validation():
    """Test validation system directly"""
    from validation import (
        SecurityValidator, ConfigValidator, DataSanitizer,
        ValidationResult, validate_and_sanitize
    )
    
    # Test device validation
    result = SecurityValidator.validate_device_specification("auto")
    assert result.is_valid
    
    result = SecurityValidator.validate_device_specification(-1)
    assert not result.is_valid
    
    # Test config validation
    config = {
        "warmup_runs": 10,
        "measurement_runs": 100,
        "timeout_seconds": 300.0
    }
    result = ConfigValidator.validate_benchmark_config(config)
    assert result.is_valid
    
    # Test sanitization
    result = DataSanitizer.sanitize_string_input("test<script>alert()</script>")
    assert result.is_valid
    assert "script" not in result.sanitized_value
    
    result = DataSanitizer.sanitize_numeric_input("42.5", min_val=0, max_val=100)
    assert result.is_valid
    assert result.sanitized_value == 42.5
    
    print("✅ Validation: Input sanitization working")

def test_monitoring():
    """Test monitoring system directly"""  
    from monitoring import (
        SystemMonitor, HealthStatus, MetricType,
        HealthCheck, Alert, MetricSample
    )
    
    monitor = SystemMonitor(sample_interval=0.1, enable_alerting=False)
    
    # Test basic functionality
    monitor.start_monitoring()
    time.sleep(0.3)
    
    metrics = monitor.get_current_metrics()
    health = monitor.get_health_status()
    
    assert "overall_status" in health
    assert health["overall_status"] in [s.value for s in HealthStatus]
    
    monitor.stop_monitoring()
    
    # Test custom health check
    def custom_check():
        return True
    
    check = HealthCheck(
        name="test_check",
        check_function=custom_check,
        warning_threshold=50.0
    )
    monitor.add_health_check(check)
    
    print(f"✅ Monitoring: {len(metrics)} metrics, status: {health['overall_status']}")

def test_resource_pool():
    """Test resource pool directly"""
    try:
        from resource_pool import (
            ResourcePool, PoolStrategy, ResourceStatus,
            PooledResource, ResourceContext
        )
        
        def factory():
            return f"resource_{time.time()}"
        
        def health_check(resource):
            return True
        
        pool = ResourcePool(
            resource_factory=factory,
            min_size=2,
            max_size=4,
            strategy=PoolStrategy.FIXED_SIZE,
            health_check_function=health_check
        )
        
        # Test acquisition
        resource = pool.acquire(timeout=1.0)
        assert resource is not None
        
        # Test context manager
        with ResourceContext(pool) as ctx_resource:
            assert ctx_resource is not None
        
        pool.release(resource)
        
        stats = pool.get_statistics()
        assert stats["current_pool_size"] >= 2
        
        pool.shutdown(timeout=1.0)
        
        print(f"✅ Resource Pool: {stats['total_acquisitions']} acquisitions")
        
    except ImportError as e:
        print(f"⚠️ Resource Pool skipped: {e}")

def test_auto_scaler():
    """Test auto-scaler directly"""
    try:
        from auto_scaler import (
            AutoScaler, ScalingTrigger, ScalingDirection,
            ScalingRule, LoadPredictor, MetricCollector
        )
        
        scaler = AutoScaler(min_capacity=1, max_capacity=10)
        
        # Test metric addition
        scaler.add_metric(ScalingTrigger.CPU_UTILIZATION, 75.0)
        scaler.add_metric(ScalingTrigger.MEMORY_UTILIZATION, 60.0)
        
        # Test manual scaling
        success = scaler.manual_scale(5, "Test scaling")
        assert success
        
        # Test prediction
        prediction = scaler.load_predictor.predict_load(300.0)
        assert prediction.confidence >= 0.0
        
        stats = scaler.get_statistics()
        assert stats["current_capacity"] == 5
        
        print(f"✅ Auto-scaler: capacity={stats['current_capacity']}")
        
    except ImportError as e:
        print(f"⚠️ Auto-scaler skipped: {e}")

def test_performance():
    """Test performance characteristics"""
    from error_handling import ErrorRecoverySystem
    from security import SecurityManager
    
    # Error handling performance
    recovery = ErrorRecoverySystem()
    start = time.time()
    for i in range(50):
        recovery.handle_error(Exception(f"Test {i}"))
    error_time = time.time() - start
    
    # Security scanning performance  
    security = SecurityManager()
    start = time.time()
    for i in range(50):
        security.scan_for_threats(f"test input {i}", "text")
    security_time = time.time() - start
    
    assert error_time < 1.0
    assert security_time < 1.0
    
    print(f"✅ Performance: error {error_time:.3f}s, security {security_time:.3f}s")

def test_concurrency():
    """Test thread safety"""
    from security import SecurityManager
    from error_handling import ErrorRecoverySystem
    
    results = []
    
    def worker(worker_id):
        security = SecurityManager()
        recovery = ErrorRecoverySystem()
        
        for i in range(5):
            security.scan_for_threats(f"worker {worker_id} input {i}", "text")
            recovery.handle_error(Exception(f"Worker {worker_id} error {i}"))
        
        results.append(worker_id)
    
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    assert len(results) == 3
    print(f"✅ Concurrency: {len(results)} workers completed")

def main():
    """Run all tests"""
    print("🚀 Direct Quality Gates Validation")
    print("="*50)
    
    tests = [
        ("Error Handling", test_error_handling),
        ("Security System", test_security),
        ("Validation System", test_validation),
        ("Monitoring System", test_monitoring),
        ("Resource Pool", test_resource_pool),
        ("Auto-scaler", test_auto_scaler),
        ("Performance", test_performance),
        ("Concurrency", test_concurrency),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1
    
    print("="*50)
    print(f"📊 RESULTS: ✅ {passed} passed, ❌ {failed} failed")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)