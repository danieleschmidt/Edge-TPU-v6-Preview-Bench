"""
Performance Benchmark for Quantum Task Planner
Comprehensive performance testing and optimization validation
"""

import asyncio
import time
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Callable
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

# Add src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'edge_tpu_v6_bench', 'quantum_planner'))

@dataclass
class BenchmarkResult:
    name: str
    duration: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    quantum_efficiency: float
    metadata: Dict[str, Any]

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        print("🚀 Quantum Task Planner Performance Benchmark Suite")
        print("=" * 60)
    
    def benchmark_task_creation(self) -> BenchmarkResult:
        """Benchmark task creation performance"""
        print("\n📋 Benchmarking Task Creation Performance...")
        
        from quantum_task_engine import QuantumTaskEngine, Priority
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        engine = QuantumTaskEngine(max_workers=4)
        
        def dummy_task():
            return "completed"
        
        # Create 1000 tasks
        num_tasks = 1000
        for i in range(num_tasks):
            engine.add_task(
                task_id=f"task_{i}",
                name=f"Task {i}",
                function=dummy_task,
                description=f"Benchmark task {i}",
                priority=Priority.MEDIUM,
                estimated_duration=0.01,
                quantum_weight=0.5
            )
        
        duration = time.time() - start_time
        end_memory = self._get_memory_usage()
        memory_usage = end_memory - start_memory
        
        throughput = num_tasks / duration
        
        print(f"  ✅ Created {num_tasks} tasks in {duration:.3f}s")
        print(f"  📊 Throughput: {throughput:.1f} tasks/second")
        print(f"  💾 Memory usage: {memory_usage:.2f} MB")
        
        return BenchmarkResult(
            name="Task Creation",
            duration=duration,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # Not measured for creation
            success_rate=1.0,
            quantum_efficiency=min(10.0, throughput / 100),
            metadata={"num_tasks": num_tasks}
        )
    
    async def benchmark_task_execution(self) -> BenchmarkResult:
        """Benchmark task execution performance"""
        print("\n⚡ Benchmarking Task Execution Performance...")
        
        from quantum_task_engine import QuantumTaskEngine, Priority
        
        engine = QuantumTaskEngine(max_workers=8)
        
        # Create compute-intensive tasks
        def cpu_task(task_id: int):
            # Simulate CPU work
            start = time.time()
            result = 0
            for i in range(10000):
                result += i * task_id
            duration = time.time() - start
            return f"Task {task_id} computed {result} in {duration:.4f}s"
        
        # Add tasks
        num_tasks = 100
        start_memory = self._get_memory_usage()
        
        for i in range(num_tasks):
            engine.add_task(
                task_id=f"compute_{i}",
                name=f"Compute Task {i}",
                function=lambda i=i: cpu_task(i),
                estimated_duration=0.01
            )
        
        start_time = time.time()
        result = await engine.execute_quantum_plan()
        duration = time.time() - start_time
        
        end_memory = self._get_memory_usage()
        memory_usage = end_memory - start_memory
        
        success_rate = result['executed_tasks'] / result['total_tasks']
        throughput = result['executed_tasks'] / duration
        
        print(f"  ✅ Executed {result['executed_tasks']}/{result['total_tasks']} tasks")
        print(f"  ⏱️  Total time: {duration:.3f}s")
        print(f"  📊 Throughput: {throughput:.1f} tasks/second") 
        print(f"  🎯 Success rate: {success_rate:.1%}")
        print(f"  🌌 Quantum efficiency: {result['quantum_efficiency_score']:.3f}")
        
        return BenchmarkResult(
            name="Task Execution",
            duration=duration,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=0.0,
            success_rate=success_rate,
            quantum_efficiency=result['quantum_efficiency_score'],
            metadata={
                "num_tasks": num_tasks,
                "avg_task_duration": result.get('average_task_duration', 0)
            }
        )
    
    async def benchmark_dependency_resolution(self) -> BenchmarkResult:
        """Benchmark dependency resolution performance"""
        print("\n🔗 Benchmarking Dependency Resolution Performance...")
        
        from quantum_task_engine import QuantumTaskEngine
        
        engine = QuantumTaskEngine(max_workers=4)
        
        def simple_task(task_id: str):
            time.sleep(0.001)  # Minimal work
            return f"Task {task_id} completed"
        
        # Create complex dependency chain
        num_chains = 10
        chain_length = 20
        total_tasks = num_chains * chain_length
        
        start_time = time.time()
        
        # Create dependency chains
        for chain in range(num_chains):
            previous_task = None
            
            for step in range(chain_length):
                task_id = f"chain_{chain}_step_{step}"
                dependencies = {previous_task} if previous_task else set()
                
                engine.add_task(
                    task_id=task_id,
                    name=f"Chain {chain} Step {step}",
                    function=lambda tid=task_id: simple_task(tid),
                    dependencies=dependencies,
                    estimated_duration=0.001
                )
                
                previous_task = task_id
        
        creation_time = time.time() - start_time
        
        # Execute with dependency resolution
        execution_start = time.time()
        result = await engine.execute_quantum_plan()
        execution_duration = time.time() - execution_start
        
        total_duration = time.time() - start_time
        success_rate = result['executed_tasks'] / result['total_tasks']
        throughput = result['executed_tasks'] / execution_duration
        
        print(f"  ✅ Resolved dependencies for {total_tasks} tasks")
        print(f"  🏗️  Creation time: {creation_time:.3f}s")
        print(f"  ⚡ Execution time: {execution_duration:.3f}s")
        print(f"  📊 Throughput: {throughput:.1f} tasks/second")
        print(f"  🎯 Success rate: {success_rate:.1%}")
        
        return BenchmarkResult(
            name="Dependency Resolution",
            duration=total_duration,
            throughput=throughput,
            memory_usage=self._get_memory_usage(),
            cpu_usage=0.0,
            success_rate=success_rate,
            quantum_efficiency=result['quantum_efficiency_score'],
            metadata={
                "num_chains": num_chains,
                "chain_length": chain_length,
                "creation_time": creation_time,
                "execution_time": execution_duration
            }
        )
    
    async def benchmark_parallel_scaling(self) -> BenchmarkResult:
        """Benchmark parallel scaling performance"""
        print("\n⚡ Benchmarking Parallel Scaling Performance...")
        
        from quantum_task_engine import QuantumTaskEngine
        
        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        results = []
        
        def cpu_intensive_task(task_id: int):
            # CPU-bound work
            result = 0
            for i in range(50000):
                result += i * task_id % 1000
            return result
        
        num_tasks = 50
        
        for workers in worker_counts:
            print(f"  🔧 Testing with {workers} workers...")
            
            engine = QuantumTaskEngine(max_workers=workers)
            
            # Add tasks
            for i in range(num_tasks):
                engine.add_task(
                    task_id=f"parallel_{i}",
                    name=f"Parallel Task {i}",
                    function=lambda i=i: cpu_intensive_task(i),
                    estimated_duration=0.1
                )
            
            start_time = time.time()
            result = await engine.execute_quantum_plan()
            duration = time.time() - start_time
            
            throughput = result['executed_tasks'] / duration
            efficiency = throughput / workers  # Efficiency per worker
            
            results.append({
                'workers': workers,
                'duration': duration,
                'throughput': throughput,
                'efficiency': efficiency,
                'success_rate': result['executed_tasks'] / result['total_tasks']
            })
            
            print(f"     ✅ {workers} workers: {throughput:.1f} tasks/s (efficiency: {efficiency:.1f})")
        
        # Calculate scaling efficiency
        baseline_throughput = results[0]['throughput']
        scaling_efficiency = []
        
        for r in results:
            expected_throughput = baseline_throughput * r['workers']
            actual_efficiency = r['throughput'] / expected_throughput
            scaling_efficiency.append(actual_efficiency)
        
        avg_duration = statistics.mean([r['duration'] for r in results])
        avg_throughput = statistics.mean([r['throughput'] for r in results])
        avg_success_rate = statistics.mean([r['success_rate'] for r in results])
        
        print(f"  📊 Scaling efficiency: {scaling_efficiency}")
        print(f"  🎯 Average throughput: {avg_throughput:.1f} tasks/second")
        
        return BenchmarkResult(
            name="Parallel Scaling",
            duration=avg_duration,
            throughput=avg_throughput,
            memory_usage=self._get_memory_usage(),
            cpu_usage=0.0,
            success_rate=avg_success_rate,
            quantum_efficiency=statistics.mean(scaling_efficiency),
            metadata={
                "worker_results": results,
                "scaling_efficiency": scaling_efficiency
            }
        )
    
    async def benchmark_quantum_algorithms(self) -> BenchmarkResult:
        """Benchmark quantum-inspired algorithms"""
        print("\n🌌 Benchmarking Quantum Algorithm Performance...")
        
        from quantum_task_engine import QuantumTaskEngine
        
        engine = QuantumTaskEngine(max_workers=6, quantum_coherence_time=5.0)
        
        # Create tasks with varying quantum properties
        def quantum_task(task_id: str, complexity: float):
            # Simulate quantum computation
            start = time.time()
            result = 0
            iterations = int(complexity * 10000)
            
            for i in range(iterations):
                # Simulate quantum superposition calculations
                result += (i ** 0.5) * complexity
            
            duration = time.time() - start
            return f"Quantum task {task_id}: {result:.2f} computed in {duration:.4f}s"
        
        # Add quantum tasks with different weights
        num_tasks = 75
        quantum_weights = [0.2, 0.5, 0.8, 1.0]
        complexities = [0.1, 0.3, 0.7, 1.0]
        
        for i in range(num_tasks):
            weight = quantum_weights[i % len(quantum_weights)]
            complexity = complexities[i % len(complexities)]
            
            engine.add_task(
                task_id=f"quantum_{i}",
                name=f"Quantum Task {i}",
                function=lambda tid=f"quantum_{i}", comp=complexity: quantum_task(tid, comp),
                quantum_weight=weight,
                estimated_duration=complexity * 0.1
            )
        
        start_time = time.time()
        result = await engine.execute_quantum_plan()
        duration = time.time() - start_time
        
        throughput = result['executed_tasks'] / duration
        success_rate = result['executed_tasks'] / result['total_tasks']
        quantum_efficiency = result['quantum_efficiency_score']
        
        print(f"  ✅ Quantum algorithm execution completed")
        print(f"  ⏱️  Duration: {duration:.3f}s") 
        print(f"  📊 Throughput: {throughput:.1f} tasks/second")
        print(f"  🌌 Quantum efficiency: {quantum_efficiency:.3f}")
        print(f"  🎯 Success rate: {success_rate:.1%}")
        
        return BenchmarkResult(
            name="Quantum Algorithms",
            duration=duration,
            throughput=throughput,
            memory_usage=self._get_memory_usage(),
            cpu_usage=0.0,
            success_rate=success_rate,
            quantum_efficiency=quantum_efficiency,
            metadata={
                "num_tasks": num_tasks,
                "coherence_maintained": result.get('quantum_coherence_maintained', False)
            }
        )
    
    def benchmark_memory_efficiency(self) -> BenchmarkResult:
        """Benchmark memory efficiency"""
        print("\n💾 Benchmarking Memory Efficiency...")
        
        from quantum_task_engine import QuantumTaskEngine
        
        # Measure memory usage during different operations
        initial_memory = self._get_memory_usage()
        
        engine = QuantumTaskEngine(max_workers=4)
        
        def memory_task(size_mb: float):
            # Allocate some memory
            data = [0] * int(size_mb * 1024 * 128)  # Rough MB allocation
            result = sum(data[:1000])  # Do some work
            del data  # Clean up
            return result
        
        # Add memory-intensive tasks
        num_tasks = 50
        memory_sizes = [0.1, 0.5, 1.0, 2.0]  # MB per task
        
        start_time = time.time()
        
        for i in range(num_tasks):
            size = memory_sizes[i % len(memory_sizes)]
            engine.add_task(
                task_id=f"memory_{i}",
                name=f"Memory Task {i}",
                function=lambda s=size: memory_task(s),
                estimated_duration=0.01
            )
        
        creation_memory = self._get_memory_usage()
        
        # Execute tasks
        result = asyncio.run(engine.execute_quantum_plan())
        
        execution_memory = self._get_memory_usage()
        duration = time.time() - start_time
        
        # Clean up and measure final memory
        del engine
        import gc
        gc.collect()
        
        final_memory = self._get_memory_usage()
        
        peak_memory = max(creation_memory, execution_memory) - initial_memory
        memory_efficiency = (final_memory - initial_memory) / peak_memory if peak_memory > 0 else 1.0
        
        print(f"  📊 Memory usage analysis:")
        print(f"     Initial: {initial_memory:.1f} MB")
        print(f"     Peak: {peak_memory:.1f} MB")
        print(f"     Final: {final_memory - initial_memory:.1f} MB")
        print(f"     Efficiency: {memory_efficiency:.2f}")
        
        throughput = result['executed_tasks'] / duration
        success_rate = result['executed_tasks'] / result['total_tasks']
        
        return BenchmarkResult(
            name="Memory Efficiency",
            duration=duration,
            throughput=throughput,
            memory_usage=peak_memory,
            cpu_usage=0.0,
            success_rate=success_rate,
            quantum_efficiency=memory_efficiency,
            metadata={
                "initial_memory": initial_memory,
                "peak_memory": peak_memory,
                "final_memory": final_memory,
                "memory_efficiency": memory_efficiency
            }
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback for systems without psutil
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert KB to MB on Linux
        except:
            return 0.0
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("🏁 Starting comprehensive performance benchmark...")
        
        benchmarks = [
            ("Task Creation", self.benchmark_task_creation),
            ("Task Execution", self.benchmark_task_execution),
            ("Dependency Resolution", self.benchmark_dependency_resolution), 
            ("Parallel Scaling", self.benchmark_parallel_scaling),
            ("Quantum Algorithms", self.benchmark_quantum_algorithms),
            ("Memory Efficiency", self.benchmark_memory_efficiency)
        ]
        
        total_start = time.time()
        
        for name, benchmark_func in benchmarks:
            print(f"\n🔧 Running {name} benchmark...")
            
            try:
                if asyncio.iscoroutinefunction(benchmark_func):
                    result = await benchmark_func()
                else:
                    result = benchmark_func()
                
                self.results.append(result)
                print(f"  ✅ {name} benchmark completed successfully")
                
            except Exception as e:
                print(f"  ❌ {name} benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        
        total_duration = time.time() - total_start
        
        # Generate comprehensive report
        report = self._generate_performance_report(total_duration)
        
        return report
    
    def _generate_performance_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Calculate aggregate metrics
        avg_throughput = statistics.mean([r.throughput for r in self.results])
        avg_memory = statistics.mean([r.memory_usage for r in self.results])
        avg_success_rate = statistics.mean([r.success_rate for r in self.results])
        avg_quantum_efficiency = statistics.mean([r.quantum_efficiency for r in self.results])
        
        # Performance score calculation
        performance_score = min(100, (
            avg_throughput * 0.3 +
            avg_success_rate * 100 * 0.3 +  
            avg_quantum_efficiency * 10 * 0.2 +
            (1.0 / max(0.1, avg_memory)) * 100 * 0.2
        ))
        
        report = {
            'summary': {
                'total_benchmarks': len(self.results),
                'total_duration': total_duration,
                'performance_score': performance_score,
                'avg_throughput': avg_throughput,
                'avg_memory_usage': avg_memory,
                'avg_success_rate': avg_success_rate,
                'avg_quantum_efficiency': avg_quantum_efficiency
            },
            'detailed_results': [
                {
                    'name': r.name,
                    'duration': r.duration,
                    'throughput': r.throughput,
                    'memory_usage': r.memory_usage,
                    'success_rate': r.success_rate,
                    'quantum_efficiency': r.quantum_efficiency,
                    'metadata': r.metadata
                }
                for r in self.results
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze results and provide recommendations
        throughputs = [r.throughput for r in self.results]
        memory_usages = [r.memory_usage for r in self.results]
        
        if statistics.mean(throughputs) < 50:
            recommendations.append("Consider increasing worker count for better throughput")
        
        if max(memory_usages) > 100:  # > 100MB
            recommendations.append("Optimize memory usage with caching and cleanup")
        
        success_rates = [r.success_rate for r in self.results]
        if min(success_rates) < 0.95:
            recommendations.append("Improve error handling and retry mechanisms")
        
        quantum_efficiencies = [r.quantum_efficiency for r in self.results]
        if statistics.mean(quantum_efficiencies) < 5:
            recommendations.append("Tune quantum algorithm parameters for better efficiency")
        
        if not recommendations:
            recommendations.append("Performance is excellent - no specific optimizations needed")
        
        return recommendations
    
    def print_summary_report(self, report: Dict[str, Any]):
        """Print formatted summary report"""
        print("\n" + "=" * 60)
        print("🏆 QUANTUM TASK PLANNER PERFORMANCE REPORT")
        print("=" * 60)
        
        summary = report['summary']
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  • Performance Score: {summary['performance_score']:.1f}/100")
        print(f"  • Total Benchmarks: {summary['total_benchmarks']}")
        print(f"  • Total Duration: {summary['total_duration']:.2f}s")
        
        print(f"\n📈 KEY METRICS:")
        print(f"  • Average Throughput: {summary['avg_throughput']:.1f} tasks/second")
        print(f"  • Average Memory Usage: {summary['avg_memory_usage']:.1f} MB")
        print(f"  • Average Success Rate: {summary['avg_success_rate']:.1%}")
        print(f"  • Average Quantum Efficiency: {summary['avg_quantum_efficiency']:.2f}")
        
        print(f"\n📋 BENCHMARK RESULTS:")
        for result in report['detailed_results']:
            print(f"  🔧 {result['name']}:")
            print(f"     Throughput: {result['throughput']:.1f} tasks/s")
            print(f"     Duration: {result['duration']:.3f}s")
            print(f"     Success Rate: {result['success_rate']:.1%}")
            print(f"     Quantum Efficiency: {result['quantum_efficiency']:.2f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Performance grade
        score = summary['performance_score']
        if score >= 80:
            grade = "A - Excellent"
            emoji = "🏆"
        elif score >= 70:
            grade = "B - Good"
            emoji = "✅"
        elif score >= 60:
            grade = "C - Acceptable"
            emoji = "⚠️"
        else:
            grade = "D - Needs Improvement"
            emoji = "❌"
        
        print(f"\n{emoji} OVERALL GRADE: {grade} (Score: {score:.1f}/100)")
        
        return score >= 70  # Pass if score >= 70

async def main():
    """Run performance benchmark suite"""
    benchmark = PerformanceBenchmark()
    
    try:
        report = await benchmark.run_comprehensive_benchmark()
        
        # Save report
        report_file = Path(__file__).parent / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Print summary
        success = benchmark.print_summary_report(report)
        
        return success
        
    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)