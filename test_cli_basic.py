#!/usr/bin/env python3
"""
Basic CLI test for Edge TPU v6 benchmark suite
Tests that CLI commands can be invoked without errors
"""

import sys
import os
from pathlib import Path
import subprocess

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

print("Testing CLI basic functionality...")

# Test help command
try:
    result = subprocess.run([
        sys.executable, '-c', 
        f"""
import sys
sys.path.insert(0, '{src_path}')
from edge_tpu_v6_bench.cli import cli
cli(['--help'])
"""
    ], capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        print("✓ CLI help command works")
    else:
        print(f"✗ CLI help command failed: {result.stderr}")
        
except subprocess.TimeoutExpired:
    print("✗ CLI help command timed out")
except Exception as e:
    print(f"✗ CLI help command error: {e}")

# Test devices command
try:
    result = subprocess.run([
        sys.executable, '-c', 
        f"""
import sys
sys.path.insert(0, '{src_path}')
from edge_tpu_v6_bench.cli import cli
cli(['devices'])
"""
    ], capture_output=True, text=True, timeout=15)
    
    if result.returncode == 0:
        print("✓ CLI devices command works")
    else:
        print(f"✗ CLI devices command failed: {result.stderr}")
        
except subprocess.TimeoutExpired:
    print("✗ CLI devices command timed out")
except Exception as e:
    print(f"✗ CLI devices command error: {e}")

print("CLI basic test completed!")