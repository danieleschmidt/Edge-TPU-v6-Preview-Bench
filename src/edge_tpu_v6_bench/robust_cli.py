"""
Robust CLI for Edge TPU v6 Benchmark Suite - Generation 2
Enhanced command-line interface with comprehensive error handling, security, and monitoring
"""

import sys
import time
import argparse
import os
import json
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('robust_cli.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path for imports when run directly
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

try:
    from .core.robust_benchmark import (
        RobustEdgeTPUBenchmark,
        RobustBenchmarkConfig,
        SecurityConfig,
        SecurityLevel,
        save_robust_results
    )
except ImportError:
    # Direct import when run as script
    from edge_tpu_v6_bench.core.robust_benchmark import (
        RobustEdgeTPUBenchmark,
        RobustBenchmarkConfig,
        SecurityConfig,
        SecurityLevel,
        save_robust_results
    )

def main():
    """Main CLI entry point with robust error handling"""
    parser = argparse.ArgumentParser(
        description='Edge TPU v6 Benchmark Suite - Robust CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s benchmark --model model.tflite --device edge_tpu_v6 --runs 1000
  %(prog)s benchmark --model model.tflite --security strict --monitoring
  %(prog)s health --device edge_tpu_v6
  %(prog)s validate --model model.tflite --security paranoid
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', default='robust_cli.log',
                       help='Log file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run robust model benchmark')
    bench_parser.add_argument('--model', '-m', required=True,
                             help='Path to model file')
    bench_parser.add_argument('--device', '-d', default='edge_tpu_v6',
                             choices=['edge_tpu_v6', 'edge_tpu_v5e', 'cpu_fallback'],
                             help='Target device')
    bench_parser.add_argument('--runs', '-r', type=int, default=100,
                             help='Number of measurement runs')
    bench_parser.add_argument('--warmup', '-w', type=int, default=10,
                             help='Number of warmup runs')
    bench_parser.add_argument('--timeout', type=float, default=300.0,
                             help='Benchmark timeout in seconds')
    bench_parser.add_argument('--retries', type=int, default=3,
                             help='Maximum retry attempts per measurement')
    bench_parser.add_argument('--output', '-o', default='robust_results',
                             help='Output directory')
    bench_parser.add_argument('--security', 
                             choices=['disabled', 'basic', 'strict', 'paranoid'],
                             default='strict',
                             help='Security level')
    bench_parser.add_argument('--monitoring', action='store_true',
                             help='Enable system monitoring')
    bench_parser.add_argument('--checkpoints', action='store_true',
                             help='Enable progress checkpoints')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Perform health check')
    health_parser.add_argument('--device', '-d', default='edge_tpu_v6',
                              help='Target device')
    health_parser.add_argument('--security', 
                              choices=['disabled', 'basic', 'strict', 'paranoid'],
                              default='basic',
                              help='Security level for health check')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate model file')
    validate_parser.add_argument('--model', '-m', required=True,
                                help='Path to model file')
    validate_parser.add_argument('--security',
                                choices=['disabled', 'basic', 'strict', 'paranoid'],
                                default='strict',
                                help='Security level for validation')
    
    # Devices command
    device_parser = subparsers.add_parser('devices', help='List available devices')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Analyze benchmark results')
    analysis_parser.add_argument('--results-dir', '-r', default='robust_results',
                                help='Results directory to analyze')
    analysis_parser.add_argument('--format', '-f', choices=['table', 'json', 'report'],
                                default='table',
                                help='Output format')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    
    # Route to appropriate handler
    try:
        if args.command == 'benchmark':
            run_robust_benchmark(args)
        elif args.command == 'health':
            run_health_check(args)
        elif args.command == 'validate':
            run_validation(args)
        elif args.command == 'devices':
            list_devices()
        elif args.command == 'analyze':
            analyze_results(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def run_robust_benchmark(args):
    """Run robust benchmark with comprehensive error handling"""
    try:
        print(f"🚀 Edge TPU v6 Robust Benchmark Suite")
        print(f"📱 Device: {args.device}")
        print(f"📁 Model: {args.model}")
        print(f"🔒 Security: {args.security}")
        print(f"📊 Monitoring: {'enabled' if args.monitoring else 'disabled'}")
        
        # Validate model file exists
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Configure security
        security_config = SecurityConfig(
            level=SecurityLevel(args.security),
            enable_integrity_checks=args.security != 'disabled',
            enable_sandboxing=args.security in ['strict', 'paranoid']
        )
        
        # Initialize robust benchmark
        benchmark = RobustEdgeTPUBenchmark(
            device=args.device,
            security_config=security_config
        )
        
        # Configure benchmark parameters
        config = RobustBenchmarkConfig(
            warmup_runs=args.warmup,
            measurement_runs=args.runs,
            timeout_seconds=args.timeout,
            retry_attempts=args.retries,
            enable_monitoring=args.monitoring,
            enable_checkpoints=args.checkpoints
        )
        
        print(f"🔧 Starting robust benchmark...")
        print(f"   Runs: {args.runs} (warmup: {args.warmup})")
        print(f"   Timeout: {args.timeout}s")
        print(f"   Retries: {args.retries}")
        
        # Run benchmark
        start_time = time.time()
        result = benchmark.benchmark(model_path=model_path, config=config)
        end_time = time.time()
        
        # Display results
        if result.success:
            print(f"\n✅ Robust benchmark completed successfully!")
            print(f"⏱️  Total duration: {end_time - start_time:.1f}s")
            print(f"\n📊 Performance Metrics:")
            print(f"   Latency (mean):    {result.latency_mean_ms:.2f} ± {result.latency_std_ms:.2f} ms")
            print(f"   Latency (median):  {result.latency_median_ms:.2f} ms")
            print(f"   Latency (p95):     {result.latency_p95_ms:.2f} ms")
            print(f"   Latency (p99):     {result.latency_p99_ms:.2f} ms")
            print(f"   Latency (range):   {result.latency_min_ms:.2f} - {result.latency_max_ms:.2f} ms")
            print(f"   Throughput:        {result.throughput_fps:.1f} FPS")
            
            print(f"\n🔧 Reliability Metrics:")
            print(f"   Success rate:      {result.success_rate:.1%}")
            print(f"   Successful runs:   {result.successful_measurements}/{result.total_measurements}")
            print(f"   Failed runs:       {result.failed_measurements}")
            
            if args.monitoring:
                print(f"\n🖥️  System Metrics:")
                print(f"   CPU usage:         {result.cpu_usage_percent:.1f}%")
                print(f"   Memory usage:      {result.memory_usage_mb:.1f} MB")
            
            print(f"\n🔒 Security Status:")
            print(f"   Integrity verified: {'✅' if result.integrity_verified else '❌'}")
            print(f"   Security scan:      {'✅ passed' if result.security_scan_passed else '❌ failed'}")
            print(f"   Model hash:         {result.model_hash[:16] if result.model_hash else 'N/A'}...")
            
            if result.warnings:
                print(f"\n⚠️  Warnings ({len(result.warnings)}):")
                for warning in result.warnings[:5]:  # Show first 5 warnings
                    print(f"   • {warning}")
                if len(result.warnings) > 5:
                    print(f"   • ... and {len(result.warnings) - 5} more")
            
            # Save results
            output_path = save_robust_results(result, args.output)
            print(f"\n💾 Detailed results saved to: {output_path}")
            
        else:
            print(f"\n❌ Robust benchmark failed!")
            print(f"Error: {result.error_message}")
            if result.warnings:
                print(f"Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:
                    print(f"  • {warning}")
            
            # Save failure results too
            output_path = save_robust_results(result, args.output)
            print(f"💾 Failure details saved to: {output_path}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        print(f"❌ Benchmark execution failed: {e}")
        sys.exit(1)

def run_health_check(args):
    """Run comprehensive health check"""
    try:
        print(f"🏥 Edge TPU Health Check")
        print(f"📱 Device: {args.device}")
        print(f"🔒 Security: {args.security}")
        
        # Configure security
        security_config = SecurityConfig(
            level=SecurityLevel(args.security)
        )
        
        # Initialize benchmark for health check
        benchmark = RobustEdgeTPUBenchmark(
            device=args.device,
            security_config=security_config
        )
        
        print(f"🔍 Running health diagnostics...")
        
        # Get device info
        device_info = benchmark.get_device_info()
        print(f"\n📱 Device Information:")
        print(f"   Type: {device_info['device_type']}")
        print(f"   Benchmark ID: {device_info['benchmark_id']}")
        print(f"   Security Level: {device_info['security_level']}")
        print(f"   Circuit Breaker: {device_info['circuit_breaker_state']}")
        print(f"   Status: {device_info['status']}")
        
        # Run health check
        health = benchmark.health_check()
        print(f"\n🏥 Health Check Results:")
        print(f"   Overall Status: {health['status'].upper()}")
        print(f"   Basic Inference: {health.get('basic_inference', 'not tested')}")
        
        if health.get('circuit_breaker'):
            print(f"   Circuit Breaker: {health['circuit_breaker']}")
        
        if health.get('issues'):
            print(f"\n⚠️  Issues Found ({len(health['issues'])}):")
            for issue in health['issues']:
                print(f"   • {issue}")
        else:
            print(f"   ✅ No issues detected")
        
        # Overall health status
        if health['status'] == 'healthy':
            print(f"\n✅ Device is healthy and ready for benchmarking")
        elif health['status'] == 'degraded':
            print(f"\n⚠️  Device is functional but with degraded performance")
        else:
            print(f"\n❌ Device has health issues that may affect benchmarking")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

def run_validation(args):
    """Run model file validation"""
    try:
        print(f"🔍 Model Validation")
        print(f"📁 Model: {args.model}")
        print(f"🔒 Security: {args.security}")
        
        # Configure security
        security_config = SecurityConfig(
            level=SecurityLevel(args.security)
        )
        
        # Initialize benchmark for validation
        benchmark = RobustEdgeTPUBenchmark(
            device='cpu_fallback',  # Use CPU for validation only
            security_config=security_config
        )
        
        print(f"🔍 Running validation checks...")
        
        # Attempt to load model (this runs all validation)
        try:
            model_info = benchmark.load_model(args.model)
            
            print(f"\n✅ Model validation passed!")
            print(f"📁 Model Information:")
            print(f"   Path: {model_info['path']}")
            print(f"   Size: {model_info['size_mb']:.2f} MB")
            print(f"   Format: {model_info['format']}")
            print(f"   Hash: {model_info['hash'][:16]}...")
            
            print(f"\n🔒 Security Checks:")
            print(f"   Integrity verified: {'✅' if model_info['integrity_verified'] else '❌'}")
            print(f"   Security scan: {'✅ passed' if model_info['security_scan_passed'] else '❌ failed'}")
            
            print(f"\n✅ Model is ready for benchmarking")
            
        except Exception as e:
            print(f"\n❌ Model validation failed!")
            print(f"Error: {e}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

def list_devices():
    """List available devices with enhanced information"""
    print("📱 Available Edge TPU devices:")
    
    devices = [
        {
            'name': 'edge_tpu_v6',
            'description': 'Google Edge TPU v6 (Preview)',
            'performance': 'High',
            'features': ['INT4', 'Structured Sparsity', 'Dynamic Shapes', 'Enhanced Security'],
            'security_features': ['Hardware TEE', 'Secure Boot', 'Model Encryption'],
            'reliability': 'Production Ready'
        },
        {
            'name': 'edge_tpu_v5e',
            'description': 'Google Edge TPU v5e',
            'performance': 'Medium-High',
            'features': ['INT8', 'UINT8', 'Basic Security'],
            'security_features': ['Basic Validation', 'File Integrity'],
            'reliability': 'Stable'
        },
        {
            'name': 'cpu_fallback',
            'description': 'CPU Fallback',
            'performance': 'Low',
            'features': ['All quantizations', 'Software-based'],
            'security_features': ['OS-level Security'],
            'reliability': 'Always Available'
        }
    ]
    
    for device in devices:
        print(f"\n  🔹 {device['name']}")
        print(f"     Description: {device['description']}")
        print(f"     Performance: {device['performance']}")
        print(f"     Reliability: {device['reliability']}")
        print(f"     Features: {', '.join(device['features'])}")
        print(f"     Security: {', '.join(device['security_features'])}")

def analyze_results(args):
    """Analyze robust benchmark results"""
    try:
        print(f"📊 Analyzing Robust Benchmark Results")
        print(f"📁 Results directory: {args.results_dir}")
        
        results_path = Path(args.results_dir)
        if not results_path.exists():
            print(f"❌ Results directory not found: {results_path}")
            sys.exit(1)
        
        # Find all robust result files
        result_files = list(results_path.glob('robust_benchmark_*.json'))
        
        if not result_files:
            print(f"❌ No robust benchmark results found in {results_path}")
            sys.exit(1)
        
        print(f"🔍 Found {len(result_files)} result files")
        
        # Load and analyze results
        all_results = []
        successful_results = []
        failed_results = []
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                all_results.append(data)
                
                if data['benchmark_info']['success']:
                    successful_results.append(data)
                else:
                    failed_results.append(data)
            except Exception as e:
                print(f"⚠️  Failed to load {result_file}: {e}")
        
        if not all_results:
            print("❌ No valid results to analyze")
            sys.exit(1)
        
        # Display analysis based on format
        if args.format == 'table':
            display_table_analysis(successful_results, failed_results)
        elif args.format == 'json':
            display_json_analysis(successful_results, failed_results)
        elif args.format == 'report':
            generate_analysis_report(successful_results, failed_results, args.results_dir)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

def display_table_analysis(successful_results, failed_results):
    """Display tabular analysis"""
    print(f"\n📈 Robust Benchmark Analysis")
    print(f"=" * 80)
    print(f"Total benchmarks: {len(successful_results) + len(failed_results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Success rate: {len(successful_results) / (len(successful_results) + len(failed_results)):.1%}")
    
    if successful_results:
        print(f"\n📊 Performance Summary (Successful Runs)")
        print(f"-" * 80)
        print(f"{'Device':<15} {'Mean Lat':<10} {'P95 Lat':<10} {'Throughput':<12} {'Success Rate':<12}")
        print(f"-" * 80)
        
        for result in successful_results[:10]:  # Show top 10
            perf = result['performance_metrics']
            rel = result['reliability_metrics']
            device = result['benchmark_info']['device_type']
            
            print(f"{device:<15} {perf['latency_mean_ms']:<10.2f} "
                  f"{perf['latency_p95_ms']:<10.2f} {perf['throughput_fps']:<12.1f} "
                  f"{rel['success_rate']:<12.1%}")
    
    if failed_results:
        print(f"\n❌ Failed Benchmarks")
        print(f"-" * 80)
        for result in failed_results[:5]:  # Show first 5 failures
            device = result['benchmark_info']['device_type']
            error = result['error_info']['error_message']
            print(f"Device: {device}, Error: {error[:50]}...")

def display_json_analysis(successful_results, failed_results):
    """Display JSON analysis"""
    analysis = {
        'summary': {
            'total_benchmarks': len(successful_results) + len(failed_results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / (len(successful_results) + len(failed_results))
        },
        'successful_results': successful_results,
        'failed_results': failed_results
    }
    
    print(json.dumps(analysis, indent=2, default=str))

def generate_analysis_report(successful_results, failed_results, output_dir):
    """Generate comprehensive HTML analysis report"""
    report_path = Path(output_dir) / 'robust_analysis_report.html'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robust Edge TPU Benchmark Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .metrics {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .metric-card {{ flex: 1; background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #34495e; color: white; }}
            .success {{ color: #27ae60; }}
            .error {{ color: #e74c3c; }}
            .warning {{ color: #f39c12; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Robust Edge TPU Benchmark Analysis</h1>
                <p>Comprehensive analysis of {len(successful_results) + len(failed_results)} benchmark runs</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{len(successful_results) + len(failed_results)}</div>
                    <div class="metric-label">Total Benchmarks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value success">{len(successful_results)}</div>
                    <div class="metric-label">Successful</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value error">{len(failed_results)}</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(successful_results) / (len(successful_results) + len(failed_results)):.1%}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
            </div>
    """
    
    if successful_results:
        html_content += """
            <h2>📊 Performance Results</h2>
            <table>
                <tr>
                    <th>Device</th>
                    <th>Mean Latency (ms)</th>
                    <th>P95 Latency (ms)</th>
                    <th>Throughput (FPS)</th>
                    <th>Success Rate</th>
                    <th>Security Level</th>
                </tr>
        """
        
        for result in successful_results:
            perf = result['performance_metrics']
            rel = result['reliability_metrics']
            device = result['benchmark_info']['device_type']
            security = result['benchmark_info']['security_level']
            
            html_content += f"""
                <tr>
                    <td>{device}</td>
                    <td>{perf['latency_mean_ms']:.2f}</td>
                    <td>{perf['latency_p95_ms']:.2f}</td>
                    <td>{perf['throughput_fps']:.1f}</td>
                    <td class="success">{rel['success_rate']:.1%}</td>
                    <td>{security}</td>
                </tr>
            """
        
        html_content += "</table>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Comprehensive analysis report generated: {report_path}")

if __name__ == '__main__':
    main()