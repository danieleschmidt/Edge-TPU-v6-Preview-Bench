#!/usr/bin/env python3
"""
Health check script for Edge TPU v6 Benchmark Suite Docker container
"""

import sys
import time
import socket
import subprocess
import requests
from pathlib import Path

def check_port(host: str, port: int, timeout: int = 5) -> bool:
    """Check if a port is open and responding"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error):
        return False

def check_http_endpoint(url: str, timeout: int = 5) -> bool:
    """Check if HTTP endpoint is responding"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

def check_service_health() -> bool:
    """Check if the main service is healthy"""
    try:
        # Try to import and initialize core components
        sys.path.insert(0, '/app/src')
        
        from edge_tpu_v6_bench.core.monitoring import SystemMonitor
        from edge_tpu_v6_bench.core.error_handling import ErrorRecoverySystem
        
        # Quick health check of core systems
        monitor = SystemMonitor(sample_interval=1.0, enable_alerting=False)
        recovery = ErrorRecoverySystem()
        
        # Check system resources
        health = monitor.get_health_status()
        if health['overall_status'] == 'down':
            return False
        
        return True
        
    except Exception as e:
        print(f"Service health check failed: {e}")
        return False

def check_file_permissions() -> bool:
    """Check file permissions and directories"""
    required_dirs = ['/app/data', '/app/logs', '/app/cache', '/app/results']
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Required directory missing: {dir_path}")
            return False
        
        if not path.is_dir():
            print(f"Path is not a directory: {dir_path}")
            return False
        
        # Check write permissions
        test_file = path / '.healthcheck_test'
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            print(f"Cannot write to directory {dir_path}: {e}")
            return False
    
    return True

def check_system_resources() -> bool:
    """Check system resource availability"""
    try:
        # Check memory usage
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = None
        mem_available = None
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1])
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1])
        
        if mem_total and mem_available:
            mem_usage_percent = ((mem_total - mem_available) / mem_total) * 100
            if mem_usage_percent > 95:  # More than 95% memory usage
                print(f"High memory usage: {mem_usage_percent:.1f}%")
                return False
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage('/app')
        disk_usage_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
        
        if disk_usage_percent > 95:  # More than 95% disk usage
            print(f"High disk usage: {disk_usage_percent:.1f}%")
            return False
        
        return True
        
    except Exception as e:
        print(f"System resource check failed: {e}")
        return False

def main():
    """Main health check"""
    print("🏥 Starting health check...")
    
    checks = [
        ("File Permissions", check_file_permissions),
        ("System Resources", check_system_resources),
        ("Service Health", check_service_health),
    ]
    
    # Optional network checks (if ports are expected to be open)
    if check_port('localhost', 8080, timeout=2):
        checks.append(("HTTP Endpoint", lambda: check_http_endpoint('http://localhost:8080/health', timeout=5)))
    
    failed_checks = []
    
    for check_name, check_func in checks:
        print(f"   Checking {check_name}...")
        try:
            if check_func():
                print(f"   ✅ {check_name} OK")
            else:
                print(f"   ❌ {check_name} FAILED")
                failed_checks.append(check_name)
        except Exception as e:
            print(f"   ❌ {check_name} ERROR: {e}")
            failed_checks.append(check_name)
    
    if failed_checks:
        print(f"\n❌ Health check FAILED. Failed checks: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        print(f"\n✅ Health check PASSED. All systems operational.")
        sys.exit(0)

if __name__ == "__main__":
    main()