"""
Scalable CLI for Edge TPU v6 Benchmark Suite - Generation 3
Advanced command-line interface with intelligent scaling, concurrency, and performance optimization
"""

import sys
import time
import argparse
import os
import json
import logging
from pathlib import Path
from typing import Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(thread)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scalable_cli.log')
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
    from .core.scalable_benchmark import (
        ScalableEdgeTPUBenchmark,
        ScalableConfig,
        ExecutionMode,
        ScalingStrategy,
        save_scalable_results
    )
except ImportError:
    # Direct import when run as script
    from edge_tpu_v6_bench.core.scalable_benchmark import (
        ScalableEdgeTPUBenchmark,
        ScalableConfig,
        ExecutionMode,
        ScalingStrategy,
        save_scalable_results
    )

def main():
    """Main CLI entry point with scalable features"""
    parser = argparse.ArgumentParser(
        description='Edge TPU v6 Benchmark Suite - Scalable CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s benchmark --model model.tflite --device edge_tpu_v6 --runs 1000 --mode hybrid
  %(prog)s benchmark --model model.tflite --workers 8 --scaling adaptive --cache
  %(prog)s multi-benchmark --models model1.tflite model2.tflite --concurrent
  %(prog)s performance-test --model model.tflite --stress-test --duration 300
  %(prog)s optimization-report --results-dir scalable_results
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-file', default='scalable_cli.log',
                       help='Log file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run scalable model benchmark')
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
    bench_parser.add_argument('--workers', type=int, default=0,
                             help='Maximum number of workers (0 = auto)')
    bench_parser.add_argument('--mode', 
                             choices=['sequential', 'threaded', 'async', 'multiprocess', 'hybrid'],
                             default='hybrid',
                             help='Execution mode')
    bench_parser.add_argument('--scaling',
                             choices=['conservative', 'aggressive', 'adaptive', 'predictive'],
                             default='adaptive',
                             help='Auto-scaling strategy')
    bench_parser.add_argument('--target-latency', type=float, default=5.0,
                             help='Target latency in milliseconds')
    bench_parser.add_argument('--target-throughput', type=float, default=200.0,
                             help='Target throughput in FPS')
    bench_parser.add_argument('--cache', action='store_true',
                             help='Enable performance caching')
    bench_parser.add_argument('--memory-pool', action='store_true',
                             help='Enable memory pooling')
    bench_parser.add_argument('--jit', action='store_true',
                             help='Enable JIT optimization')
    bench_parser.add_argument('--output', '-o', default='scalable_results',
                             help='Output directory')
    
    # Multi-benchmark command
    multi_parser = subparsers.add_parser('multi-benchmark', help='Benchmark multiple models')
    multi_parser.add_argument('--models', '-m', nargs='+', required=True,
                             help='Paths to model files')
    multi_parser.add_argument('--device', '-d', default='edge_tpu_v6',
                             help='Target device')
    multi_parser.add_argument('--concurrent', action='store_true',
                             help='Run models concurrently')
    multi_parser.add_argument('--workers', type=int, default=0,
                             help='Maximum number of workers')
    multi_parser.add_argument('--output', '-o', default='scalable_results',
                             help='Output directory')
    
    # Performance test command
    perf_parser = subparsers.add_parser('performance-test', help='Run comprehensive performance test')
    perf_parser.add_argument('--model', '-m', required=True,
                            help='Path to model file')
    perf_parser.add_argument('--device', '-d', default='edge_tpu_v6',
                            help='Target device')
    perf_parser.add_argument('--stress-test', action='store_true',
                            help='Run stress test')
    perf_parser.add_argument('--duration', type=int, default=60,
                            help='Test duration in seconds')
    perf_parser.add_argument('--ramp-up', type=int, default=10,
                            help='Ramp-up duration in seconds')
    perf_parser.add_argument('--output', '-o', default='scalable_results',
                            help='Output directory')
    
    # Optimization report command
    opt_parser = subparsers.add_parser('optimization-report', help='Generate optimization report')
    opt_parser.add_argument('--results-dir', '-r', default='scalable_results',
                           help='Results directory to analyze')
    opt_parser.add_argument('--format', '-f', choices=['table', 'json', 'html'],
                           default='html',
                           help='Output format')
    
    # Devices command
    device_parser = subparsers.add_parser('devices', help='List available devices with scaling info')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.INFO)
    
    # Route to appropriate handler
    try:
        if args.command == 'benchmark':
            run_scalable_benchmark(args)
        elif args.command == 'multi-benchmark':
            run_multi_benchmark(args)
        elif args.command == 'performance-test':
            run_performance_test(args)
        elif args.command == 'optimization-report':
            generate_optimization_report(args)
        elif args.command == 'devices':
            list_scalable_devices()
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose or args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def run_scalable_benchmark(args):
    """Run scalable benchmark with advanced features"""
    try:
        print(f"🚀 Edge TPU v6 Scalable Benchmark Suite")
        print(f"📱 Device: {args.device}")
        print(f"📁 Model: {args.model}")
        print(f"⚙️  Execution mode: {args.mode}")
        print(f"📈 Scaling strategy: {args.scaling}")
        print(f"🔧 Workers: {args.workers if args.workers > 0 else 'auto'}")
        
        # Validate model file
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Configure scalable benchmark
        import multiprocessing
        max_workers = args.workers if args.workers > 0 else min(8, multiprocessing.cpu_count())
        
        config = ScalableConfig(
            warmup_runs=args.warmup,
            measurement_runs=args.runs,
            timeout_seconds=args.timeout,
            execution_mode=ExecutionMode(args.mode),
            max_workers=max_workers,
            scaling_strategy=ScalingStrategy(args.scaling),
            target_latency_ms=args.target_latency,
            target_throughput_fps=args.target_throughput,
            enable_caching=args.cache,
            enable_memory_pooling=args.memory_pool,
            enable_jit_optimization=args.jit
        )
        
        # Initialize scalable benchmark
        benchmark = ScalableEdgeTPUBenchmark(device=args.device, config=config)
        
        print(f"🔧 Starting scalable benchmark...")
        print(f"   Runs: {args.runs} (warmup: {args.warmup})")
        print(f"   Target latency: {args.target_latency}ms")
        print(f"   Target throughput: {args.target_throughput} FPS")
        print(f"   Caching: {'enabled' if args.cache else 'disabled'}")
        print(f"   Memory pooling: {'enabled' if args.memory_pool else 'disabled'}")
        print(f"   JIT optimization: {'enabled' if args.jit else 'disabled'}")
        
        # Run benchmark
        start_time = time.time()
        result = benchmark.benchmark(model_path=model_path, config=config)
        end_time = time.time()
        
        # Display comprehensive results
        if result.success:
            print(f"\n✅ Scalable benchmark completed successfully!")
            print(f"⏱️  Total duration: {end_time - start_time:.1f}s")
            
            print(f"\n📊 Performance Metrics:")
            print(f"   Latency (mean):      {result.latency_mean_ms:.2f} ± {result.latency_std_ms:.2f} ms")
            print(f"   Latency (median):    {result.latency_median_ms:.2f} ms")
            print(f"   Latency (p95/p99):   {result.latency_p95_ms:.2f} / {result.latency_p99_ms:.2f} ms")
            print(f"   Latency (range):     {result.latency_min_ms:.2f} - {result.latency_max_ms:.2f} ms")
            
            print(f"\n🚀 Throughput Metrics:")
            print(f"   Average throughput:  {result.throughput_fps:.1f} FPS")
            print(f"   Peak throughput:     {result.throughput_peak_fps:.1f} FPS")
            print(f"   Sustained throughput: {result.throughput_sustained_fps:.1f} FPS")
            
            print(f"\n📈 Scalability Metrics:")
            print(f"   Parallel speedup:    {result.parallel_speedup:.1f}x")
            print(f"   Parallel efficiency: {result.parallel_efficiency:.1%}")
            print(f"   Scaling factor:      {result.scaling_factor:.1f}")
            print(f"   Optimal workers:     {result.optimal_workers}")
            
            print(f"\n🔧 Concurrency Metrics:")
            print(f"   Tasks executed:      {result.total_tasks_executed}")
            print(f"   Peak concurrency:    {result.concurrent_tasks_peak}")
            print(f"   Task success rate:   {result.task_success_rate:.1%}")
            print(f"   Execution mode:      {result.execution_mode_used}")
            
            print(f"\n💾 Cache & Memory:")
            print(f"   Cache hit rate:      {result.cache_hit_rate:.1%}")
            print(f"   Cache efficiency:    {result.cache_efficiency:.2f}")
            print(f"   Memory utilization:  {result.memory_utilization_mb:.1f} MB")
            print(f"   Memory pool efficiency: {result.memory_pool_efficiency:.1%}")
            
            print(f"\n🖥️  Resource Utilization:")
            print(f"   CPU utilization:     {result.cpu_utilization_percent:.1f}%")
            
            if result.warnings:
                print(f"\n⚠️  Warnings ({len(result.warnings)}):")
                for warning in result.warnings[:3]:
                    print(f"   • {warning}")
                if len(result.warnings) > 3:
                    print(f"   • ... and {len(result.warnings) - 3} more")
            
            # Performance recommendations
            print(f"\n💡 Performance Recommendations:")
            if result.parallel_efficiency < 0.7:
                print(f"   • Consider reducing worker count for better efficiency")
            if result.cache_hit_rate < 0.5:
                print(f"   • Increase cache TTL or enable model caching")
            if result.cpu_utilization_percent > 90:
                print(f"   • System is CPU-bound, consider scaling up hardware")
            if result.throughput_fps < args.target_throughput * 0.8:
                print(f"   • Consider hybrid execution mode or more aggressive scaling")
            
            # Save comprehensive results
            output_path = save_scalable_results(result, args.output)
            print(f"\n💾 Detailed results saved to: {output_path}")
            
        else:
            print(f"\n❌ Scalable benchmark failed!")
            print(f"Error: {result.error_message}")
            if result.warnings:
                print(f"Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:
                    print(f"  • {warning}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Scalable benchmark execution failed: {e}")
        print(f"❌ Scalable benchmark execution failed: {e}")
        sys.exit(1)

def run_multi_benchmark(args):
    """Run multiple model benchmarks"""
    try:
        print(f"🔄 Multi-Model Scalable Benchmark")
        print(f"📁 Models: {len(args.models)}")
        print(f"📱 Device: {args.device}")
        print(f"🔄 Concurrent: {'yes' if args.concurrent else 'no'}")
        
        # Validate all model files
        model_paths = []
        for model in args.models:
            path = Path(model)
            if not path.exists():
                print(f"❌ Model file not found: {path}")
                sys.exit(1)
            model_paths.append(path)
        
        # Configure benchmark
        import multiprocessing
        max_workers = args.workers if args.workers > 0 else min(len(model_paths), multiprocessing.cpu_count())
        
        config = ScalableConfig(
            max_workers=max_workers,
            execution_mode=ExecutionMode.THREADED if args.concurrent else ExecutionMode.SEQUENTIAL
        )
        
        # Initialize benchmark
        benchmark = ScalableEdgeTPUBenchmark(device=args.device, config=config)
        
        start_time = time.time()
        
        if args.concurrent:
            print(f"🚀 Running {len(model_paths)} models concurrently...")
            results = benchmark.benchmark_multiple_models(model_paths)
        else:
            print(f"🔄 Running {len(model_paths)} models sequentially...")
            results = []
            for i, model_path in enumerate(model_paths):
                print(f"   Benchmarking model {i+1}/{len(model_paths)}: {model_path.name}")
                result = benchmark.benchmark(model_path=model_path)
                results.append(result)
        
        end_time = time.time()
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        print(f"\n✅ Multi-model benchmark completed!")
        print(f"⏱️  Total duration: {end_time - start_time:.1f}s")
        print(f"📊 Results: {len(successful_results)}/{len(results)} successful")
        
        if successful_results:
            # Calculate aggregate metrics
            avg_latency = sum(r.latency_mean_ms for r in successful_results) / len(successful_results)
            total_throughput = sum(r.throughput_fps for r in successful_results)
            avg_efficiency = sum(r.parallel_efficiency for r in successful_results) / len(successful_results)
            
            print(f"\n📈 Aggregate Metrics:")
            print(f"   Average latency:     {avg_latency:.2f} ms")
            print(f"   Total throughput:    {total_throughput:.1f} FPS")
            print(f"   Average efficiency:  {avg_efficiency:.1%}")
            
            # Top performers
            top_performers = sorted(successful_results, key=lambda r: r.throughput_fps, reverse=True)[:3]
            print(f"\n🏆 Top Performers:")
            for i, result in enumerate(top_performers):
                model_name = Path(result.model_path).name
                print(f"   {i+1}. {model_name}: {result.throughput_fps:.1f} FPS")
        
        if failed_results:
            print(f"\n❌ Failed Models ({len(failed_results)}):")
            for result in failed_results:
                model_name = Path(result.model_path).name if result.model_path else "unknown"
                print(f"   • {model_name}: {result.error_message}")
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_path = Path(args.output) / f'multi_benchmark_{timestamp}.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        multi_results = {
            'summary': {
                'total_models': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'concurrent_execution': args.concurrent,
                'total_duration_s': end_time - start_time
            },
            'individual_results': [
                {
                    'model_path': r.model_path,
                    'success': r.success,
                    'latency_mean_ms': r.latency_mean_ms,
                    'throughput_fps': r.throughput_fps,
                    'parallel_efficiency': r.parallel_efficiency,
                    'error_message': r.error_message
                }
                for r in results
            ]
        }
        
        with open(results_path, 'w') as f:
            json.dump(multi_results, f, indent=2, default=str)
        
        print(f"\n💾 Multi-benchmark results saved to: {results_path}")
    
    except Exception as e:
        print(f"❌ Multi-benchmark failed: {e}")
        sys.exit(1)

def run_performance_test(args):
    """Run comprehensive performance test"""
    try:
        print(f"⚡ Comprehensive Performance Test")
        print(f"📁 Model: {args.model}")
        print(f"📱 Device: {args.device}")
        print(f"⏱️  Duration: {args.duration}s")
        print(f"🔥 Stress test: {'yes' if args.stress_test else 'no'}")
        
        # Validate model
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Configure for performance testing
        config = ScalableConfig(
            measurement_runs=args.duration * 10,  # ~10 runs per second
            timeout_seconds=args.duration + 60,
            execution_mode=ExecutionMode.HYBRID,
            scaling_strategy=ScalingStrategy.AGGRESSIVE if args.stress_test else ScalingStrategy.ADAPTIVE,
            enable_caching=True,
            enable_memory_pooling=True,
            enable_jit_optimization=True
        )
        
        benchmark = ScalableEdgeTPUBenchmark(device=args.device, config=config)
        
        print(f"🚀 Starting performance test...")
        
        # Ramp-up phase
        if args.ramp_up > 0:
            print(f"📈 Ramp-up phase: {args.ramp_up}s")
            ramp_config = ScalableConfig(
                measurement_runs=args.ramp_up * 5,
                execution_mode=ExecutionMode.THREADED
            )
            benchmark.benchmark(model_path=model_path, config=ramp_config)
        
        # Main performance test
        print(f"⚡ Main performance test: {args.duration}s")
        start_time = time.time()
        result = benchmark.benchmark(model_path=model_path, config=config)
        
        if result.success:
            print(f"\n✅ Performance test completed!")
            print(f"⏱️  Actual duration: {result.total_duration_s:.1f}s")
            
            # Performance analysis
            print(f"\n📊 Performance Analysis:")
            print(f"   Sustained throughput: {result.throughput_sustained_fps:.1f} FPS")
            print(f"   Peak throughput:      {result.throughput_peak_fps:.1f} FPS")
            print(f"   Latency stability:    {result.latency_std_ms:.2f}ms std dev")
            print(f"   System efficiency:    {result.parallel_efficiency:.1%}")
            
            # Stress test specific metrics
            if args.stress_test:
                print(f"\n🔥 Stress Test Results:")
                reliability_score = result.task_success_rate * result.parallel_efficiency
                print(f"   Reliability score:    {reliability_score:.1%}")
                print(f"   Resource utilization: {result.cpu_utilization_percent:.1f}% CPU")
                print(f"   Memory pressure:      {result.memory_utilization_mb:.1f} MB")
                
                if reliability_score > 0.9:
                    print(f"   ✅ System handles stress well")
                elif reliability_score > 0.7:
                    print(f"   ⚠️  System shows some stress")
                else:
                    print(f"   ❌ System is struggling under stress")
            
            # Performance recommendations
            print(f"\n💡 Performance Insights:")
            if result.parallel_speedup < 2.0:
                print(f"   • Limited parallelization benefit observed")
            if result.cache_hit_rate > 0.8:
                print(f"   • Excellent cache performance")
            if result.memory_pool_efficiency > 0.9:
                print(f"   • Efficient memory management")
            
            # Generate performance report
            perf_report = benchmark.get_performance_report()
            
            # Save comprehensive results
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = Path(args.output) / f'performance_test_{timestamp}.json'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            perf_test_results = {
                'test_configuration': {
                    'duration_s': args.duration,
                    'stress_test': args.stress_test,
                    'ramp_up_s': args.ramp_up,
                    'model_path': str(model_path),
                    'device': args.device
                },
                'benchmark_result': {
                    'latency_mean_ms': result.latency_mean_ms,
                    'throughput_fps': result.throughput_fps,
                    'parallel_efficiency': result.parallel_efficiency,
                    'task_success_rate': result.task_success_rate,
                    'total_duration_s': result.total_duration_s
                },
                'performance_report': perf_report
            }
            
            with open(output_path, 'w') as f:
                json.dump(perf_test_results, f, indent=2, default=str)
            
            print(f"\n💾 Performance test results saved to: {output_path}")
            
        else:
            print(f"\n❌ Performance test failed!")
            print(f"Error: {result.error_message}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        sys.exit(1)

def generate_optimization_report(args):
    """Generate comprehensive optimization report"""
    try:
        print(f"📊 Generating Optimization Report")
        print(f"📁 Results directory: {args.results_dir}")
        
        results_path = Path(args.results_dir)
        if not results_path.exists():
            print(f"❌ Results directory not found: {results_path}")
            sys.exit(1)
        
        # Find all scalable result files
        result_files = list(results_path.glob('scalable_benchmark_*.json'))
        
        if not result_files:
            print(f"❌ No scalable benchmark results found")
            sys.exit(1)
        
        print(f"🔍 Analyzing {len(result_files)} result files...")
        
        # Load and analyze results
        all_results = []
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                all_results.append(data)
            except Exception as e:
                print(f"⚠️  Failed to load {result_file}: {e}")
        
        if not all_results:
            print("❌ No valid results to analyze")
            sys.exit(1)
        
        # Generate report based on format
        if args.format == 'table':
            display_optimization_table(all_results)
        elif args.format == 'json':
            display_optimization_json(all_results)
        elif args.format == 'html':
            generate_optimization_html(all_results, args.results_dir)
        
    except Exception as e:
        print(f"❌ Optimization report generation failed: {e}")
        sys.exit(1)

def display_optimization_table(results):
    """Display optimization analysis in table format"""
    print(f"\n📈 Scalability Analysis")
    print(f"=" * 100)
    
    successful_results = [r for r in results if r['benchmark_info']['success']]
    
    if successful_results:
        print(f"{'Model':<25} {'Device':<15} {'Mode':<12} {'Speedup':<8} {'Efficiency':<10} {'Throughput':<12}")
        print(f"-" * 100)
        
        for result in successful_results[:15]:  # Show top 15
            model_name = Path(result['benchmark_info']['model_path']).stem[:20]
            device = result['benchmark_info']['device_type']
            mode = result['benchmark_info']['execution_mode'][:10]
            speedup = result['scalability_metrics']['parallel_speedup']
            efficiency = result['scalability_metrics']['parallel_efficiency']
            throughput = result['throughput_metrics']['throughput_fps']
            
            print(f"{model_name:<25} {device:<15} {mode:<12} {speedup:<8.1f} "
                  f"{efficiency:<10.1%} {throughput:<12.1f}")
        
        # Summary statistics
        avg_speedup = sum(r['scalability_metrics']['parallel_speedup'] for r in successful_results) / len(successful_results)
        avg_efficiency = sum(r['scalability_metrics']['parallel_efficiency'] for r in successful_results) / len(successful_results)
        total_throughput = sum(r['throughput_metrics']['throughput_fps'] for r in successful_results)
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Average parallel speedup: {avg_speedup:.1f}x")
        print(f"   Average efficiency:       {avg_efficiency:.1%}")
        print(f"   Total throughput:         {total_throughput:.1f} FPS")

def display_optimization_json(results):
    """Display optimization analysis in JSON format"""
    analysis = {
        'summary': {
            'total_benchmarks': len(results),
            'successful': len([r for r in results if r['benchmark_info']['success']]),
            'analysis_timestamp': time.time()
        },
        'optimization_insights': [],
        'detailed_results': results
    }
    
    print(json.dumps(analysis, indent=2, default=str))

def generate_optimization_html(results, output_dir):
    """Generate comprehensive HTML optimization report"""
    successful_results = [r for r in results if r['benchmark_info']['success']]
    
    if not successful_results:
        print("❌ No successful results for HTML report")
        return
    
    report_path = Path(output_dir) / 'optimization_report.html'
    
    # Calculate key metrics
    avg_speedup = sum(r['scalability_metrics']['parallel_speedup'] for r in successful_results) / len(successful_results)
    avg_efficiency = sum(r['scalability_metrics']['parallel_efficiency'] for r in successful_results) / len(successful_results)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Edge TPU Scalability Optimization Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
            .container {{ max-width: 1200px; margin: 20px auto; background: white; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
            .header {{ background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
            .header h1 {{ margin: 0; font-size: 2.5em; }}
            .header p {{ margin: 10px 0 0; opacity: 0.9; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 30px; }}
            .metric-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 25px; border-radius: 10px; text-align: center; border: 1px solid #dee2e6; }}
            .metric-value {{ font-size: 3em; font-weight: bold; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            .metric-label {{ color: #6c757d; font-weight: 500; margin-top: 10px; }}
            .section {{ padding: 30px; border-top: 1px solid #dee2e6; }}
            .section h2 {{ color: #2c3e50; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #dee2e6; }}
            th {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; font-weight: 600; }}
            .performance-high {{ color: #28a745; font-weight: bold; }}
            .performance-medium {{ color: #ffc107; font-weight: bold; }}
            .performance-low {{ color: #dc3545; font-weight: bold; }}
            .chart-container {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Edge TPU Scalability Report</h1>
                <p>Comprehensive analysis of {len(successful_results)} scalable benchmark runs</p>
                <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{len(successful_results)}</div>
                    <div class="metric-label">Successful Benchmarks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_speedup:.1f}x</div>
                    <div class="metric-label">Average Speedup</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_efficiency:.0%}</div>
                    <div class="metric-label">Average Efficiency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sum(r['throughput_metrics']['throughput_fps'] for r in successful_results):.0f}</div>
                    <div class="metric-label">Total Throughput (FPS)</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Performance Results</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Device</th>
                        <th>Execution Mode</th>
                        <th>Parallel Speedup</th>
                        <th>Efficiency</th>
                        <th>Throughput (FPS)</th>
                        <th>Cache Hit Rate</th>
                    </tr>
    """
    
    for result in successful_results:
        model_name = Path(result['benchmark_info']['model_path']).stem
        device = result['benchmark_info']['device_type']
        mode = result['benchmark_info']['execution_mode']
        speedup = result['scalability_metrics']['parallel_speedup']
        efficiency = result['scalability_metrics']['parallel_efficiency']
        throughput = result['throughput_metrics']['throughput_fps']
        cache_hit = result['cache_metrics']['cache_hit_rate']
        
        # Performance classification
        perf_class = "performance-high" if efficiency > 0.8 else "performance-medium" if efficiency > 0.5 else "performance-low"
        
        html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{device}</td>
                        <td>{mode}</td>
                        <td class="{perf_class}">{speedup:.1f}x</td>
                        <td class="{perf_class}">{efficiency:.1%}</td>
                        <td>{throughput:.1f}</td>
                        <td>{cache_hit:.1%}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>💡 Optimization Recommendations</h2>
                <div class="chart-container">
    """
    
    # Add optimization recommendations
    high_performers = [r for r in successful_results if r['scalability_metrics']['parallel_efficiency'] > 0.8]
    low_performers = [r for r in successful_results if r['scalability_metrics']['parallel_efficiency'] < 0.5]
    
    html_content += f"""
                    <h3>✅ High Performance Configurations ({len(high_performers)} found)</h3>
                    <ul>
    """
    
    for result in high_performers[:5]:
        model_name = Path(result['benchmark_info']['model_path']).stem
        mode = result['benchmark_info']['execution_mode']
        workers = result['scalability_metrics']['optimal_workers']
        html_content += f"<li>{model_name}: {mode} mode with {workers} workers</li>"
    
    html_content += "</ul>"
    
    if low_performers:
        html_content += f"""
                    <h3>⚠️ Optimization Opportunities ({len(low_performers)} found)</h3>
                    <ul>
        """
        
        for result in low_performers[:5]:
            model_name = Path(result['benchmark_info']['model_path']).stem
            efficiency = result['scalability_metrics']['parallel_efficiency']
            html_content += f"<li>{model_name}: {efficiency:.1%} efficiency - consider mode/worker optimization</li>"
        
        html_content += "</ul>"
    
    html_content += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Comprehensive optimization report generated: {report_path}")

def list_scalable_devices():
    """List devices with scalability information"""
    print("📱 Edge TPU Devices - Scalability Features:")
    
    devices = [
        {
            'name': 'edge_tpu_v6',
            'description': 'Google Edge TPU v6 (Preview)',
            'performance': 'Ultra High',
            'max_parallel_streams': 16,
            'execution_modes': ['threaded', 'async', 'multiprocess', 'hybrid'],
            'scaling_strategies': ['adaptive', 'predictive', 'aggressive'],
            'features': ['Auto-scaling', 'JIT Optimization', 'Memory Pooling', 'Intelligent Caching'],
            'recommended_workers': '4-8'
        },
        {
            'name': 'edge_tpu_v5e',
            'description': 'Google Edge TPU v5e',
            'performance': 'High',
            'max_parallel_streams': 8,
            'execution_modes': ['threaded', 'async', 'hybrid'],
            'scaling_strategies': ['adaptive', 'conservative'],
            'features': ['Basic Auto-scaling', 'Memory Pooling'],
            'recommended_workers': '2-4'
        },
        {
            'name': 'cpu_fallback',
            'description': 'CPU Fallback',
            'performance': 'Medium',
            'max_parallel_streams': 32,
            'execution_modes': ['threaded', 'multiprocess'],
            'scaling_strategies': ['conservative'],
            'features': ['Process-level Parallelism'],
            'recommended_workers': 'CPU cores'
        }
    ]
    
    for device in devices:
        print(f"\n  🔹 {device['name']}")
        print(f"     Description: {device['description']}")
        print(f"     Performance: {device['performance']}")
        print(f"     Max parallel streams: {device['max_parallel_streams']}")
        print(f"     Execution modes: {', '.join(device['execution_modes'])}")
        print(f"     Scaling strategies: {', '.join(device['scaling_strategies'])}")
        print(f"     Features: {', '.join(device['features'])}")
        print(f"     Recommended workers: {device['recommended_workers']}")

if __name__ == '__main__':
    main()