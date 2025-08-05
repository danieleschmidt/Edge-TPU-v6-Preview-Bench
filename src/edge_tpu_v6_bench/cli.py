"""
Command-line interface for Edge TPU v6 Benchmark Suite
High-performance CLI with global-first design and comprehensive functionality
"""

import sys
import os
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

import click
import yaml

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('edge_tpu_bench.log')
    ]
)
logger = logging.getLogger(__name__)

# Import core components
from .core.device_manager import DeviceManager
from .core.benchmark import EdgeTPUBenchmark, BenchmarkConfig
from .quantization.auto_quantizer import AutoQuantizer
from .benchmarks.micro import MicroBenchmarkSuite
from .benchmarks.standard import StandardBenchmark

# Global configuration
CONFIG_FILE = 'edge_tpu_bench_config.yaml'
DEFAULT_OUTPUT_DIR = 'edge_tpu_results'

# I18n support - Global-first implementation
TRANSLATIONS = {
    'en': {
        'device_detected': 'Device detected',
        'benchmark_started': 'Benchmark started',
        'benchmark_completed': 'Benchmark completed',
        'error_occurred': 'Error occurred',
        'results_saved': 'Results saved to',
        'quantization_applied': 'Quantization applied',
        'help_device': 'Target device (auto, edge_tpu_v6, edge_tpu_v5e, cpu)',
        'help_output': 'Output directory for results',
        'help_verbose': 'Enable verbose logging',
        'help_config': 'Configuration file path',
    },
    'es': {
        'device_detected': 'Dispositivo detectado',
        'benchmark_started': 'Benchmark iniciado',
        'benchmark_completed': 'Benchmark completado',
        'error_occurred': 'Ocurrió un error',
        'results_saved': 'Resultados guardados en',
        'quantization_applied': 'Cuantización aplicada',
        'help_device': 'Dispositivo objetivo (auto, edge_tpu_v6, edge_tpu_v5e, cpu)',
        'help_output': 'Directorio de salida para resultados',
        'help_verbose': 'Habilitar registro detallado',
        'help_config': 'Ruta del archivo de configuración',
    },
    'fr': {
        'device_detected': 'Périphérique détecté',
        'benchmark_started': 'Benchmark démarré',
        'benchmark_completed': 'Benchmark terminé',
        'error_occurred': 'Une erreur s\'est produite',
        'results_saved': 'Résultats sauvegardés dans',
        'quantization_applied': 'Quantification appliquée',
        'help_device': 'Périphérique cible (auto, edge_tpu_v6, edge_tpu_v5e, cpu)',
        'help_output': 'Répertoire de sortie pour les résultats',
        'help_verbose': 'Activer la journalisation détaillée',
        'help_config': 'Chemin du fichier de configuration',
    },
    'de': {
        'device_detected': 'Gerät erkannt',
        'benchmark_started': 'Benchmark gestartet',
        'benchmark_completed': 'Benchmark abgeschlossen',
        'error_occurred': 'Fehler aufgetreten',
        'results_saved': 'Ergebnisse gespeichert in',
        'quantization_applied': 'Quantisierung angewendet',
        'help_device': 'Zielgerät (auto, edge_tpu_v6, edge_tpu_v5e, cpu)',
        'help_output': 'Ausgabeverzeichnis für Ergebnisse',
        'help_verbose': 'Ausführliche Protokollierung aktivieren',
        'help_config': 'Konfigurationsdateipfad',
    },
    'ja': {
        'device_detected': 'デバイスが検出されました',
        'benchmark_started': 'ベンチマークが開始されました',
        'benchmark_completed': 'ベンチマークが完了しました',
        'error_occurred': 'エラーが発生しました',
        'results_saved': '結果が保存されました',
        'quantization_applied': '量子化が適用されました',
        'help_device': 'ターゲットデバイス (auto, edge_tpu_v6, edge_tpu_v5e, cpu)',
        'help_output': '結果の出力ディレクトリ',
        'help_verbose': '詳細ログを有効にする',
        'help_config': '設定ファイルのパス',
    },
    'zh': {
        'device_detected': '检测到设备',
        'benchmark_started': '基准测试已开始',
        'benchmark_completed': '基准测试已完成',
        'error_occurred': '发生错误',
        'results_saved': '结果已保存到',
        'quantization_applied': '已应用量化',
        'help_device': '目标设备 (auto, edge_tpu_v6, edge_tpu_v5e, cpu)',
        'help_output': '结果输出目录',
        'help_verbose': '启用详细日志记录',
        'help_config': '配置文件路径',
    }
}

# Current language (can be set via environment variable)
CURRENT_LANG = os.environ.get('EDGE_TPU_LANG', 'en')

def t(key: str) -> str:
    """Translation function"""
    return TRANSLATIONS.get(CURRENT_LANG, TRANSLATIONS['en']).get(key, key)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = CONFIG_FILE
    
    try:
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        else:
            logger.info(f"No configuration file found at {config_path}, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}

def save_results(results: Dict[str, Any], output_dir: str):
    """Save benchmark results to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    json_path = output_path / f'benchmark_results_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Save YAML results (more human-readable)
    yaml_path = output_path / f'benchmark_results_{timestamp}.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    click.echo(f"{t('results_saved')}: {json_path}")
    return str(json_path)

@click.group()
@click.option('--device', '-d', default='auto', help=t('help_device'))
@click.option('--output', '-o', default=DEFAULT_OUTPUT_DIR, help=t('help_output'))
@click.option('--verbose', '-v', is_flag=True, help=t('help_verbose'))
@click.option('--config', '-c', help=t('help_config'))
@click.option('--lang', default='en', help='Language (en, es, fr, de, ja, zh)')
@click.pass_context
def cli(ctx, device, output, verbose, config, lang):
    """
    Edge TPU v6 Benchmark Suite
    
    Comprehensive benchmarking and optimization for Edge TPU devices.
    """
    # Set language
    global CURRENT_LANG
    CURRENT_LANG = lang
    
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    
    # Load configuration
    config_data = load_config(config)
    
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj.update({
        'device': device,
        'output': output,
        'verbose': verbose,
        'config': config_data,
        'lang': lang
    })
    
    click.echo(f"🚀 Edge TPU v6 Benchmark Suite - Language: {lang}")

@cli.command()
@click.pass_context
def devices(ctx):
    """List available Edge TPU devices"""
    try:
        device_manager = DeviceManager()
        devices = device_manager.list_devices()
        
        if not devices:
            click.echo("⚠️  No Edge TPU devices detected")
            return
        
        click.echo("📱 Available devices:")
        for device in devices:
            status = "✅" if device.get('connected', True) else "❌"
            click.echo(f"  {status} {device['id']} ({device['type']})")
            click.echo(f"     Performance: {device['performance']}")
            click.echo(f"     Power limit: {device['power_limit_w']}W")
            click.echo(f"     Memory: {device['memory_mb']}MB")
            
            if device.get('features'):
                features = device['features']
                feature_list = []
                if features.get('int4'): feature_list.append('INT4')
                if features.get('structured_sparsity'): feature_list.append('Sparsity')
                if features.get('dynamic_shapes'): feature_list.append('Dynamic')
                
                if feature_list:
                    click.echo(f"     Features: {', '.join(feature_list)}")
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ {t('error_occurred')}: {e}")
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', required=True, help='Path to model file (TFLite, SavedModel, or Keras)')
@click.option('--runs', '-r', default=100, help='Number of measurement runs')
@click.option('--warmup', '-w', default=10, help='Number of warmup runs')
@click.option('--batch-size', '-b', multiple=True, type=int, help='Batch sizes to test (can specify multiple)')
@click.option('--quantize', is_flag=True, help='Apply automatic quantization')
@click.option('--power', is_flag=True, help='Measure power consumption')
@click.pass_context
def benchmark(ctx, model, runs, warmup, batch_size, quantize, power):
    """Run comprehensive benchmark on a model"""
    
    try:
        model_path = Path(model)
        if not model_path.exists():
            click.echo(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Initialize benchmark
        click.echo(f"🔧 {t('benchmark_started')} on device: {ctx.obj['device']}")
        
        edge_tpu_bench = EdgeTPUBenchmark(
            device=ctx.obj['device'],
            power_monitoring=power
        )
        
        click.echo(f"📱 {t('device_detected')}: {edge_tpu_bench.device_info.device_type.value}")
        
        # Apply quantization if requested
        model_to_benchmark = str(model_path)
        if quantize:
            click.echo("⚡ Applying automatic quantization...")
            quantizer = AutoQuantizer(target_device=edge_tpu_bench.device_info.device_type.value)
            
            # Load model for quantization
            if model_path.suffix == '.tflite':
                click.echo("⚠️  Model is already in TFLite format, skipping quantization")
            else:
                quantization_result = quantizer.quantize(str(model_path))
                if quantization_result.success:
                    model_to_benchmark = quantization_result.model_path
                    click.echo(f"✅ {t('quantization_applied')}: {quantization_result.strategy_used}")
                    click.echo(f"   Compression: {quantization_result.compression_ratio:.1f}x")
                    click.echo(f"   Size: {quantization_result.quantized_size_mb:.1f} MB")
                else:
                    click.echo(f"⚠️  Quantization failed: {quantization_result.error_message}")
        
        # Configure benchmark
        batch_sizes = list(batch_size) if batch_size else [1]
        config = BenchmarkConfig(
            warmup_runs=warmup,
            measurement_runs=runs,
            batch_sizes=batch_sizes,
            measure_power=power
        )
        
        # Run benchmark
        click.echo(f"🚀 Running benchmark: {runs} runs, batch sizes {batch_sizes}")
        
        result = edge_tpu_bench.benchmark(
            model_path=model_to_benchmark,
            config=config,
            metrics=['latency', 'throughput', 'power'] if power else ['latency', 'throughput']
        )
        
        if not result.success:
            click.echo(f"❌ Benchmark failed: {result.error_message}")
            sys.exit(1)
        
        # Display results
        click.echo(f"✅ {t('benchmark_completed')}")
        click.echo("\n📊 Results:")
        
        metrics = result.metrics
        click.echo(f"  Latency (mean): {metrics.get('latency_mean_ms', 0):.2f} ms")
        click.echo(f"  Latency (p95):  {metrics.get('latency_p95_ms', 0):.2f} ms")
        click.echo(f"  Latency (p99):  {metrics.get('latency_p99_ms', 0):.2f} ms")
        click.echo(f"  Throughput:     {metrics.get('throughput_fps', 0):.1f} FPS")
        
        if power and 'power_mean_w' in metrics:
            click.echo(f"  Power (mean):   {metrics['power_mean_w']:.2f} W")
            click.echo(f"  Energy/inf:     {metrics.get('energy_per_inference_mj', 0):.1f} mJ")
        
        # Save results
        results_data = {
            'timestamp': time.time(),
            'model_path': str(model_path),
            'device': edge_tpu_bench.device_info.device_type.value,
            'configuration': {
                'runs': runs,
                'warmup': warmup,
                'batch_sizes': batch_sizes,
                'quantization': quantize,
                'power_monitoring': power
            },
            'results': result.metrics,
            'raw_measurements': result.raw_measurements
        }
        
        output_path = save_results(results_data, ctx.obj['output'])
        
    except Exception as e:
        click.echo(f"❌ {t('error_occurred')}: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--operations', '-o', multiple=True, 
              help='Operations to benchmark (conv2d, matmul, elementwise)')
@click.option('--shapes', '-s', multiple=True,
              help='Input shapes to test (e.g., "1,224,224,3")')
@click.option('--quantization', '-q', default='int8',
              help='Quantization type (int8, uint8, float32)')
@click.pass_context
def micro(ctx, operations, shapes, quantization):
    """Run micro-benchmarks for individual operations"""
    
    try:
        click.echo(f"🔬 Starting micro-benchmarks on device: {ctx.obj['device']}")
        
        micro_bench = MicroBenchmarkSuite(device=ctx.obj['device'])
        
        # Parse input shapes
        parsed_shapes = []
        for shape_str in shapes:
            try:
                shape = tuple(map(int, shape_str.split(',')))
                parsed_shapes.append(shape)
            except ValueError:
                click.echo(f"⚠️  Invalid shape format: {shape_str}")
        
        results = {}
        
        # Run convolution benchmarks
        if not operations or 'conv2d' in operations:
            click.echo("🔄 Benchmarking convolution operations...")
            conv_results = micro_bench.benchmark_convolutions(
                input_shapes=parsed_shapes if parsed_shapes else None,
                quantization=quantization
            )
            results['convolutions'] = conv_results
        
        # Run matrix multiplication benchmarks
        if not operations or 'matmul' in operations:
            click.echo("🔄 Benchmarking matrix multiplication...")
            matmul_results = micro_bench.benchmark_matmul(quantization=quantization)
            results['matrix_multiplication'] = matmul_results
        
        # Run element-wise operation benchmarks
        if not operations or 'elementwise' in operations:
            click.echo("🔄 Benchmarking element-wise operations...")
            elementwise_results = micro_bench.benchmark_elementwise(
                tensor_sizes=parsed_shapes if parsed_shapes else None,
                quantization=quantization
            )
            results['elementwise_operations'] = elementwise_results
        
        # Generate report
        report_path = Path(ctx.obj['output']) / 'micro_benchmarks_report.html'
        micro_bench.generate_report(results, str(report_path))
        
        click.echo(f"✅ Micro-benchmarks completed")
        click.echo(f"📄 Report generated: {report_path}")
        
        # Save raw results
        results_data = {
            'timestamp': time.time(),
            'device': micro_bench.device_info.device_type.value,
            'configuration': {
                'operations': list(operations) if operations else 'all',
                'shapes': list(shapes) if shapes else 'default',
                'quantization': quantization
            },
            'results': results
        }
        
        save_results(results_data, ctx.obj['output'])
        
        # Cleanup
        micro_bench.cleanup()
        
    except Exception as e:
        click.echo(f"❌ {t('error_occurred')}: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--models', '-m', multiple=True,
              help='Models to benchmark (mobilenet_v3_small, efficientnet_lite0, etc.)')
@click.option('--category', '-cat', type=click.Choice(['vision', 'nlp', 'audio', 'all']),
              default='vision', help='Model category to benchmark')
@click.option('--quantize', is_flag=True, help='Apply automatic quantization')
@click.option('--compare-devices', is_flag=True, help='Compare across multiple devices')
@click.pass_context  
def standard(ctx, models, category, quantize, compare_devices):
    """Run standard ML model benchmarks"""
    
    try:
        click.echo(f"📈 Starting standard benchmarks on device: {ctx.obj['device']}")
        
        if compare_devices:
            # Cross-device comparison
            devices = ['edge_tpu_v6', 'edge_tpu_v5e', 'cpu_fallback']
            model_list = list(models) if models else ['mobilenet_v3_small', 'efficientnet_lite0']
            
            click.echo(f"🔄 Comparing {len(devices)} devices on {len(model_list)} models...")
            
            standard_bench = StandardBenchmark(device='auto')
            comparison = standard_bench.compare_devices(devices, model_list)
            
            # Display comparison results
            click.echo("\n📊 Device Comparison Results:")
            for device, ranking in comparison.get('rankings', {}).items():
                click.echo(f"  {ranking['rank']}. {device}: {ranking['score']:.2f}")
            
            # Save comparison plot
            plot_path = Path(ctx.obj['output']) / 'device_comparison.png'
            standard_bench.plot_comparison(comparison, str(plot_path))
            click.echo(f"📊 Comparison plot saved: {plot_path}")
            
            # Save results
            results_data = {
                'timestamp': time.time(),
                'type': 'device_comparison',
                'configuration': {
                    'models': model_list,
                    'devices': devices,
                    'quantization': quantize
                },
                'comparison': comparison
            }
            
            save_results(results_data, ctx.obj['output'])
            standard_bench.cleanup()
            
        else:
            # Single device benchmarking
            standard_bench = StandardBenchmark(device=ctx.obj['device'])
            
            model_list = list(models) if models else None
            results = []
            
            if category == 'vision' or category == 'all':
                click.echo("🔄 Benchmarking vision models...")
                vision_results = standard_bench.benchmark_vision_models(
                    model_list or ['mobilenet_v3_small', 'efficientnet_lite0'],
                    enable_quantization=quantize
                )
                results.extend(vision_results)
            
            if category == 'nlp' or category == 'all':
                click.echo("🔄 Benchmarking NLP models...")
                nlp_results = standard_bench.benchmark_nlp_models(
                    model_list or ['bert_tiny', 'distilbert'],
                    enable_quantization=quantize
                )
                results.extend(nlp_results)
            
            if category == 'audio' or category == 'all':
                click.echo("🔄 Benchmarking audio models...")
                audio_results = standard_bench.benchmark_audio_models(
                    model_list or ['yamnet'],
                    enable_quantization=quantize
                )
                results.extend(audio_results)
            
            # Display results summary
            click.echo(f"\n✅ Standard benchmarks completed: {len(results)} models")
            
            successful_results = [r for r in results if r.success]
            click.echo(f"📊 Success rate: {len(successful_results)}/{len(results)}")
            
            if successful_results:
                avg_latency = sum(r.performance_metrics.get('latency_mean_ms', 0) 
                                for r in successful_results) / len(successful_results)
                avg_throughput = sum(r.performance_metrics.get('throughput_fps', 0)
                                   for r in successful_results) / len(successful_results)
                
                click.echo(f"  Average latency: {avg_latency:.2f} ms")
                click.echo(f"  Average throughput: {avg_throughput:.1f} FPS")
            
            # Save results
            results_data = {
                'timestamp': time.time(),
                'device': standard_bench.device_info.device_type.value,
                'configuration': {
                    'category': category,
                    'models': model_list,
                    'quantization': quantize
                },
                'results': [
                    {
                        'model_name': r.model_name,
                        'model_type': r.model_type,
                        'framework': r.framework,
                        'success': r.success,
                        'performance_metrics': r.performance_metrics,
                        'power_metrics': r.power_metrics,
                        'error_message': r.error_message
                    }
                    for r in results
                ]
            }
            
            save_results(results_data, ctx.obj['output'])
            standard_bench.cleanup()
            
    except Exception as e:
        click.echo(f"❌ {t('error_occurred')}: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--input', '-i', required=True, help='Input model path')
@click.option('--output', '-o', help='Output quantized model path')
@click.option('--strategy', '-s', 
              type=click.Choice(['auto', 'int8', 'uint8', 'int4_mixed', 'hybrid']),
              default='auto', help='Quantization strategy')
@click.option('--calibration-data', help='Path to calibration dataset')
@click.option('--accuracy-threshold', default=0.02, help='Maximum accuracy drop allowed')
@click.pass_context
def quantize(ctx, input, output, strategy, calibration_data, accuracy_threshold):
    """Apply automatic quantization to a model"""
    
    try:
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"❌ Input model not found: {input_path}")
            sys.exit(1)
        
        click.echo(f"⚡ Quantizing model: {input_path}")
        click.echo(f"   Strategy: {strategy}")
        click.echo(f"   Target device: {ctx.obj['device']}")
        
        # Initialize quantizer
        quantizer = AutoQuantizer(
            target_device=ctx.obj['device'],
            optimization_target='latency'
        )
        
        # Load calibration data if provided
        calibration_dataset = None
        if calibration_data:
            # TODO: Implement calibration data loading
            click.echo(f"   Calibration data: {calibration_data}")
        
        # Configure quantization
        from edge_tpu_v6_bench.quantization.auto_quantizer import QuantizationConfig
        config = QuantizationConfig(
            target_device=ctx.obj['device'],
            optimization_target='latency',
            quantization_strategies=[strategy] if strategy != 'auto' else None,
            accuracy_threshold=accuracy_threshold
        )
        
        # Apply quantization
        result = quantizer.quantize(
            str(input_path),
            calibration_data=calibration_dataset,
            config=config
        )
        
        if result.success:
            final_output_path = output or f"quantized_{input_path.stem}.tflite"
            
            # Copy quantized model to desired output location
            import shutil
            shutil.copy2(result.model_path, final_output_path)
            
            click.echo(f"✅ Quantization successful!")
            click.echo(f"   Strategy used: {result.strategy_used}")
            click.echo(f"   Compression: {result.compression_ratio:.1f}x")
            click.echo(f"   Size: {result.original_size_mb:.1f} MB → {result.quantized_size_mb:.1f} MB")
            click.echo(f"   Estimated speedup: {result.estimated_speedup:.1f}x")
            if result.accuracy_drop > 0:
                click.echo(f"   Accuracy drop: {result.accuracy_drop:.1%}")
            click.echo(f"   Output: {final_output_path}")
            
            # Save quantization report
            report_data = {
                'timestamp': time.time(),
                'input_model': str(input_path),
                'output_model': final_output_path,
                'quantization_result': {
                    'success': result.success,
                    'strategy_used': result.strategy_used,
                    'compression_ratio': result.compression_ratio,
                    'original_size_mb': result.original_size_mb,
                    'quantized_size_mb': result.quantized_size_mb,
                    'estimated_speedup': result.estimated_speedup,
                    'accuracy_drop': result.accuracy_drop,
                    'metadata': result.metadata
                }
            }
            
            save_results(report_data, ctx.obj['output'])
            
        else:
            click.echo(f"❌ Quantization failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ {t('error_occurred')}: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--results-dir', '-r', default=DEFAULT_OUTPUT_DIR, help='Results directory to analyze')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'html']), 
              default='table', help='Output format')
@click.option('--metric', '-m', default='latency_mean_ms', help='Primary metric for analysis')
@click.pass_context
def analyze(ctx, results_dir, format, metric):
    """Analyze and compare benchmark results"""
    
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            click.echo(f"❌ Results directory not found: {results_path}")
            sys.exit(1)
        
        # Find all result files
        result_files = list(results_path.glob('benchmark_results_*.json'))
        
        if not result_files:
            click.echo(f"❌ No benchmark results found in {results_path}")
            sys.exit(1)
        
        click.echo(f"📊 Analyzing {len(result_files)} result files...")
        
        # Load and aggregate results
        all_results = []
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_results.append(data)
            except Exception as e:
                click.echo(f"⚠️  Failed to load {result_file}: {e}")
        
        if not all_results:
            click.echo("❌ No valid results to analyze")
            sys.exit(1)
        
        # Analyze results based on format
        if format == 'table':
            # Display tabular analysis
            click.echo(f"\n📈 Results Analysis (sorted by {metric}):")
            click.echo("-" * 80)
            click.echo(f"{'Model':<25} {'Device':<15} {'Metric':<15} {'Value':<15}")
            click.echo("-" * 80)
            
            # Extract and sort results
            analyzed_results = []
            for result in all_results:
                if 'results' in result:
                    analyzed_results.append({
                        'model': Path(result.get('model_path', 'unknown')).stem,
                        'device': result.get('device', 'unknown'),
                        'metric_value': result['results'].get(metric, 0),
                        'timestamp': result.get('timestamp', 0)
                    })
            
            # Sort by metric value
            analyzed_results.sort(key=lambda x: x['metric_value'])
            
            for result in analyzed_results:
                click.echo(f"{result['model']:<25} {result['device']:<15} "
                          f"{metric:<15} {result['metric_value']:<15.2f}")
            
        elif format == 'json':
            # Output JSON analysis
            analysis = {
                'total_results': len(all_results),
                'metric': metric,
                'summary': {
                    'count': len(all_results),
                    'devices': list(set(r.get('device', 'unknown') for r in all_results)),
                    'models': list(set(Path(r.get('model_path', 'unknown')).stem for r in all_results)),
                },
                'detailed_results': all_results
            }
            
            print(json.dumps(analysis, indent=2, default=str))
            
        elif format == 'html':
            # Generate HTML report
            html_path = results_path / 'analysis_report.html'
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Edge TPU Benchmark Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Edge TPU Benchmark Analysis</h1>
                <p>Analysis of {len(all_results)} benchmark results</p>
                <p>Primary metric: {metric}</p>
                
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Device</th>
                        <th>{metric}</th>
                        <th>Timestamp</th>
                    </tr>
            """
            
            for result in all_results:
                if 'results' in result:
                    model_name = Path(result.get('model_path', 'unknown')).stem
                    device = result.get('device', 'unknown')
                    metric_value = result['results'].get(metric, 0)
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', 
                                           time.localtime(result.get('timestamp', 0)))
                    
                    html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{device}</td>
                        <td>{metric_value:.2f}</td>
                        <td>{timestamp}</td>
                    </tr>
                    """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            click.echo(f"📄 HTML report generated: {html_path}")
        
        click.echo(f"✅ Analysis completed")
        
    except Exception as e:
        click.echo(f"❌ {t('error_occurred')}: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point for CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()