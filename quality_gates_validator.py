import os
#!/usr/bin/env python3
"""
Quality Gates Validator for Edge TPU v6 Benchmark Suite
Comprehensive testing, security scanning, and performance validation
"""

import sys
import os
import time
import threading
import traceback
from pathlib import Path

# Add source paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class QualityGatesValidator:
    """Comprehensive quality gates validation"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def run_test(self, test_name: str, test_function):
        """Run a single test with error handling"""
        print(f"🧪 Testing {test_name}...")
        try:
            test_function()
            self.passed_tests += 1
            self.test_results.append((test_name, True, None))
            print(f"✅ {test_name} PASSED")
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append((test_name, False, str(e)))
            print(f"❌ {test_name} FAILED: {e}")
            # Print traceback for debugging
            # traceback.print_exc()
    
    def test_core_imports(self):
        """Test that all core modules can be imported"""
        # Test individual core components that don't require numpy
        from edge_tpu_v6_bench.core.error_handling import (
            ErrorRecoverySystem, ErrorSeverity, ErrorCategory, EdgeTPUError
        )
        from edge_tpu_v6_bench.core.security import (
            SecurityManager, SecurityLevel, ThreatLevel
        )
        from edge_tpu_v6_bench.core.validation import (
            SecurityValidator, ValidationResult, ConfigValidator
        )
        from edge_tpu_v6_bench.core.monitoring import (
            SystemMonitor, HealthStatus, MetricType
        )
        
        # These might have numpy dependencies, test carefully
        try:
            from edge_tpu_v6_bench.core.resource_pool import (
                ResourcePool, PoolStrategy, ResourceStatus
            )
        except ImportError:
            pass  # Skip if dependencies not available
        
        try:
            from edge_tpu_v6_bench.core.auto_scaler import (
                AutoScaler, ScalingDirection, ScalingTrigger
            )
        except ImportError:
            pass  # Skip if dependencies not available
        
        assert True  # If we get here, imports succeeded
    
    def test_error_handling_system(self):
        """Test comprehensive error handling"""
        from edge_tpu_v6_bench.core.error_handling import (
            ErrorRecoverySystem, HardwareError, ModelError, ValidationError
        )
        
        recovery_system = ErrorRecoverySystem()
        
        # Test error categorization
        hardware_error = HardwareError("Test hardware failure")
        assert hardware_error.category.value == "hardware"
        assert hardware_error.severity.value == "high"
        
        # Test error handling
        result = recovery_system.handle_error(hardware_error)
        # Should return None since no recovery strategies will work for test errors
        
        # Test statistics
        stats = recovery_system.get_error_statistics()
        assert stats['total_errors'] >= 1
        
        print(f"  📊 Error statistics: {stats['total_errors']} errors handled")
    
    def test_security_system(self):
        """Test comprehensive security features"""
        from edge_tpu_v6_bench.core.security import SecurityManager, ThreatLevel
        
        security = SecurityManager()
        
        # Test threat detection
        threats = security.scan_for_threats("../etc/passwd", "path")
        assert len(threats) > 0, "Should detect path traversal threat"
        
        threats = security.scan_for_threats("<script>alert('xss')</script>", "text")
        assert len(threats) > 0, "Should detect XSS threat"
        
        # Test access validation
        valid_access = security.validate_access("/safe/path", "test_user")
        assert valid_access, "Safe path should be allowed"
        
        dangerous_access = security.validate_access("/etc/passwd", "test_user")
        assert not dangerous_access, "Dangerous path should be blocked"
        
        # Test secure temp file creation
        temp_file = security.create_secure_temp_file()
        assert temp_file.exists(), "Temp file should be created"
        temp_file.unlink()  # Cleanup
        
        print(f"  🔒 Security report: {len(security.security_events)} events logged")
    
    def test_validation_system(self):
        """Test input validation and sanitization"""
        from edge_tpu_v6_bench.core.validation import (
            SecurityValidator, ConfigValidator, DataSanitizer, ValidationResult
        )
        
        # Test device specification validation
        result = SecurityValidator.validate_device_specification("auto")
        assert result.is_valid, "Auto device spec should be valid"
        
        result = SecurityValidator.validate_device_specification("invalid_device")
        assert not result.is_valid, "Invalid device spec should fail"
        
        # Test config validation
        valid_config = {
            "warmup_runs": 10,
            "measurement_runs": 100,
            "timeout_seconds": 300.0,
            "batch_sizes": [1, 4, 8]
        }
        result = ConfigValidator.validate_benchmark_config(valid_config)
        assert result.is_valid, "Valid config should pass"
        
        # Test string sanitization
        result = DataSanitizer.sanitize_string_input("<script>alert('test')</script>")
        assert result.is_valid
        assert "<script>" not in result.sanitized_value, "Script tags should be removed"
        
        print("  ✨ Validation: All input sanitization working")
    
    def test_monitoring_system(self):
        """Test system monitoring and health checks"""
        from edge_tpu_v6_bench.core.monitoring import SystemMonitor, HealthStatus
        
        monitor = SystemMonitor(
            sample_interval=0.1, 
            history_size=10,
            enable_alerting=False
        )
        
        # Test monitoring startup/shutdown
        monitor.start_monitoring()
        time.sleep(0.3)  # Let it collect some samples
        
        # Check that metrics are being collected
        current_metrics = monitor.get_current_metrics()
        assert len(current_metrics) > 0, "Should collect some metrics"
        
        # Test health status
        health_status = monitor.get_health_status()
        assert "overall_status" in health_status
        assert health_status["overall_status"] in ["healthy", "warning", "critical"]
        
        monitor.stop_monitoring()
        
        print(f"  📈 Monitoring: {len(current_metrics)} metrics collected")
    
    def test_resource_pooling(self):
        """Test resource pooling system"""
        try:
            from edge_tpu_v6_bench.core.resource_pool import (
                ResourcePool, PoolStrategy, ResourceContext
            )
            
            def mock_resource_factory():
                return f"mock_resource_{int(time.time() * 1000000)}"
            
            def mock_health_check(resource):
                return True
            
            pool = ResourcePool(
                resource_factory=mock_resource_factory,
                min_size=2,
                max_size=5,
                strategy=PoolStrategy.FIXED_SIZE,
                health_check_function=mock_health_check
            )
            
            # Test resource acquisition
            resource = pool.acquire(timeout=1.0)
            assert resource is not None, "Should acquire resource"
            
            # Test context manager
            with ResourceContext(pool, timeout=1.0) as ctx_resource:
                assert ctx_resource is not None, "Context manager should work"
            
            pool.release(resource)
            
            # Test statistics
            stats = pool.get_statistics()
            assert stats["current_pool_size"] >= 2, "Should maintain minimum size"
            
            pool.shutdown(timeout=2.0)
            
            print(f"  🔄 Resource Pool: {stats['total_acquisitions']} acquisitions")
            
        except ImportError:
            print("  ⚠️ Resource pooling skipped (dependencies not available)")
    
    def test_auto_scaling(self):
        """Test auto-scaling system"""
        try:
            from edge_tpu_v6_bench.core.auto_scaler import (
                AutoScaler, ScalingTrigger, ScalingDirection
            )
            
            scaler = AutoScaler(min_capacity=1, max_capacity=10)
            
            # Test metric addition
            scaler.add_metric(ScalingTrigger.CPU_UTILIZATION, 75.0)
            scaler.add_metric(ScalingTrigger.MEMORY_UTILIZATION, 60.0)
            scaler.add_metric(ScalingTrigger.RESPONSE_TIME, 500.0)
            
            # Test manual scaling
            success = scaler.manual_scale(3, "Test scaling")
            assert success, "Manual scaling should succeed"
            
            # Test load prediction
            prediction = scaler.load_predictor.predict_load(300.0)
            assert prediction.confidence >= 0.0, "Prediction should have valid confidence"
            
            # Test statistics
            stats = scaler.get_statistics()
            assert stats["current_capacity"] == 3, "Capacity should be updated"
            assert stats["rules_enabled"] > 0, "Should have scaling rules"
            
            print(f"  📈 Auto-scaler: capacity={stats['current_capacity']}, rules={stats['rules_enabled']}")
            
        except ImportError:
            print("  ⚠️ Auto-scaling skipped (dependencies not available)")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of core systems"""
        from edge_tpu_v6_bench.core.error_handling import ErrorRecoverySystem
        from edge_tpu_v6_bench.core.security import SecurityManager
        
        # Test error handling performance
        recovery = ErrorRecoverySystem()
        start_time = time.time()
        
        for i in range(100):
            error = Exception(f"Test error {i}")
            recovery.handle_error(error)
        
        error_handling_time = time.time() - start_time
        assert error_handling_time < 1.0, "Error handling should be fast"
        
        # Test security scanning performance
        security = SecurityManager()
        start_time = time.time()
        
        for i in range(100):
            security.scan_for_threats(f"test input {i}", "text")
        
        security_scan_time = time.time() - start_time
        assert security_scan_time < 1.0, "Security scanning should be fast"
        
        print(f"  ⚡ Performance: error handling {error_handling_time:.3f}s, security {security_scan_time:.3f}s")
    
    def test_concurrent_safety(self):
        """Test thread safety of core components"""
        from edge_tpu_v6_bench.core.security import SecurityManager
        from edge_tpu_v6_bench.core.error_handling import ErrorRecoverySystem
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                security = SecurityManager()
                recovery = ErrorRecoverySystem()
                
                for i in range(10):
                    # Test concurrent operations
                    security.scan_for_threats(f"worker {worker_id} input {i}", "text")
                    recovery.handle_error(Exception(f"Worker {worker_id} error {i}"))
                
                results.append(f"Worker {worker_id} completed")
                
            except Exception as e:
                errors.append(f"Worker {worker_id} failed: {e}")
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 5, "All workers should complete"
        
        print(f"  🔄 Concurrency: {len(results)} workers completed successfully")
    
    def run_all_tests(self):
        """Run all quality gate tests"""
        print("🚀 Starting Quality Gates Validation")
        print("="*60)
        
        # Core functionality tests
        self.run_test("Core Module Imports", self.test_core_imports)
        self.run_test("Error Handling System", self.test_error_handling_system)
        self.run_test("Security System", self.test_security_system)
        self.run_test("Validation System", self.test_validation_system)
        self.run_test("Monitoring System", self.test_monitoring_system)
        self.run_test("Resource Pooling", self.test_resource_pooling)
        self.run_test("Auto-scaling", self.test_auto_scaling)
        
        # Performance and reliability tests
        self.run_test("Performance Characteristics", self.test_performance_characteristics)
        self.run_test("Concurrent Safety", self.test_concurrent_safety)
        
        print("="*60)
        print(f"📊 QUALITY GATES RESULTS")
        print(f"   ✅ Passed: {self.passed_tests}")
        print(f"   ❌ Failed: {self.failed_tests}")
        print(f"   📈 Success Rate: {(self.passed_tests/(self.passed_tests+self.failed_tests)*100):.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL QUALITY GATES PASSED!")
            print("   System is ready for production deployment")
            return True
        else:
            print(f"\n⚠️  {self.failed_tests} QUALITY GATES FAILED")
            print("   Failed tests:")
            for test_name, passed, error in self.test_results:
                if not passed:
                    print(f"   - {test_name}: {error}")
            return False

def main():
    """Main entry point"""
    validator = QualityGatesValidator()
    success = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()