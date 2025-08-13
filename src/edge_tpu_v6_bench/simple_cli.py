"""
Simple CLI for Edge TPU v6 Benchmark Suite - Generation 1
Basic command-line interface that works without complex dependencies
"""

import sys
import time
import argparse
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports when run directly
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

try:
    from .core.simple_benchmark import (
        SimpleEdgeTPUBenchmark, 
        SimpleAutoQuantizer,
        SimpleBenchmarkConfig,
        save_benchmark_results
    )
except ImportError:
    # Direct import when run as script
    from edge_tpu_v6_bench.core.simple_benchmark import (
        SimpleEdgeTPUBenchmark, 
        SimpleAutoQuantizer,
        SimpleBenchmarkConfig,
        save_benchmark_results
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Edge TPU v6 Benchmark Suite - Simple CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run model benchmark')
    bench_parser.add_argument('--model', '-m', required=True, 
                             help='Path to model file')
    bench_parser.add_argument('--device', '-d', default='edge_tpu_v6',
                             choices=['edge_tpu_v6', 'edge_tpu_v5e', 'cpu_fallback'],
                             help='Target device')
    bench_parser.add_argument('--runs', '-r', type=int, default=100,
                             help='Number of measurement runs')
    bench_parser.add_argument('--warmup', '-w', type=int, default=10,
                             help='Number of warmup runs')
    bench_parser.add_argument('--output', '-o', default='simple_results',
                             help='Output directory')
    bench_parser.add_argument('--quantize', action='store_true',
                             help='Apply automatic quantization')
    
    # Quantize command
    quant_parser = subparsers.add_parser('quantize', help='Quantize a model')
    quant_parser.add_argument('--input', '-i', required=True,
                             help='Input model path')
    quant_parser.add_argument('--device', '-d', default='edge_tpu_v6',
                             help='Target device')
    quant_parser.add_argument('--output', '-o', default='simple_results',
                             help='Output directory')
    
    # Device info command
    device_parser = subparsers.add_parser('devices', help='List available devices')
    
    args = parser.parse_args()
    
    if args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'quantize':
        run_quantization(args)
    elif args.command == 'devices':
        list_devices()
    else:
        parser.print_help()

def run_benchmark(args):
    """Run benchmark command"""
    try:
        print(f"🚀 Edge TPU v6 Benchmark Suite")
        print(f"📱 Device: {args.device}")
        print(f"📁 Model: {args.model}")
        
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Apply quantization if requested
        model_to_benchmark = str(model_path)
        if args.quantize:
            print("⚡ Applying automatic quantization...")
            quantizer = SimpleAutoQuantizer(target_device=args.device)
            quant_result = quantizer.quantize(str(model_path))
            
            if quant_result['success']:
                model_to_benchmark = quant_result['model_path']
                print(f"✅ Quantization applied: {quant_result['strategy_used']}")
                print(f"   Compression: {quant_result['compression_ratio']:.1f}x")
                print(f"   Size: {quant_result['quantized_size_mb']:.1f} MB")
            else:
                print(f"⚠️  Quantization failed: {quant_result['error_message']}")
        
        # Initialize benchmark
        benchmark = SimpleEdgeTPUBenchmark(device=args.device)
        
        # Configure benchmark
        config = SimpleBenchmarkConfig(
            warmup_runs=args.warmup,
            measurement_runs=args.runs
        )
        
        print(f"🔧 Running benchmark: {args.runs} runs")
        
        # Run benchmark
        result = benchmark.benchmark(model_path=model_to_benchmark, config=config)
        
        if result.success:
            print("✅ Benchmark completed successfully!")
            print(f"📊 Results:")
            print(f"   Latency (mean): {result.latency_mean_ms:.2f} ms")
            print(f"   Latency (p95):  {result.latency_p95_ms:.2f} ms") 
            print(f"   Latency (p99):  {result.latency_p99_ms:.2f} ms")
            print(f"   Throughput:     {result.throughput_fps:.1f} FPS")
            print(f"   Measurements:   {result.total_measurements}")
            
            # Save results
            output_path = save_benchmark_results(result, args.output)
            print(f"💾 Results saved to: {output_path}")
        else:
            print(f"❌ Benchmark failed: {result.error_message}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_quantization(args):
    """Run quantization command"""
    try:
        print(f"⚡ Quantizing model for {args.device}")
        print(f"📁 Input: {args.input}")
        
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Input model not found: {input_path}")
            sys.exit(1)
        
        quantizer = SimpleAutoQuantizer(target_device=args.device)
        result = quantizer.quantize(str(input_path))
        
        if result['success']:
            print(f"✅ Quantization successful!")
            print(f"   Strategy: {result['strategy_used']}")
            print(f"   Compression: {result['compression_ratio']:.1f}x")
            print(f"   Original size: {result['original_size_mb']:.1f} MB")
            print(f"   Quantized size: {result['quantized_size_mb']:.1f} MB")
            print(f"   Estimated speedup: {result['estimated_speedup']:.1f}x")
            print(f"   Output: {result['model_path']}")
        else:
            print(f"❌ Quantization failed: {result['error_message']}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def list_devices():
    """List available devices"""
    print("📱 Available Edge TPU devices:")
    
    devices = [
        {
            'name': 'edge_tpu_v6',
            'description': 'Google Edge TPU v6 (Preview)',
            'performance': 'High',
            'features': ['INT4', 'Structured Sparsity', 'Dynamic Shapes']
        },
        {
            'name': 'edge_tpu_v5e',
            'description': 'Google Edge TPU v5e',
            'performance': 'Medium-High', 
            'features': ['INT8', 'UINT8']
        },
        {
            'name': 'cpu_fallback',
            'description': 'CPU Fallback',
            'performance': 'Low',
            'features': ['All quantizations']
        }
    ]
    
    for device in devices:
        print(f"  🔹 {device['name']}")
        print(f"     {device['description']}")
        print(f"     Performance: {device['performance']}")
        print(f"     Features: {', '.join(device['features'])}")
        print()

if __name__ == '__main__':
    main()