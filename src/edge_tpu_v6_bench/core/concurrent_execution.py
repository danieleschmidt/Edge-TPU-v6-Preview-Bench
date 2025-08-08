"""
Advanced concurrent execution engine for Edge TPU v6 benchmarking
High-performance parallel execution, load balancing, and resource optimization
"""

import asyncio
import threading
import concurrent.futures
import multiprocessing
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import queue
import weakref
from collections import defaultdict
import psutil

logger = logging.getLogger(__name__)

class ExecutionStrategy(Enum):
    """Execution strategies for different workloads"""
    SEQUENTIAL = "sequential"        # Single-threaded execution
    THREADED = "threaded"           # Multi-threaded execution  
    PROCESS_POOL = "process_pool"   # Multi-process execution
    ASYNC_IO = "async_io"          # Async I/O execution
    HYBRID = "hybrid"              # Hybrid approach based on workload
    ADAPTIVE = "adaptive"          # Adaptive strategy selection

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"
    TPU = "tpu"

@dataclass
class TaskSpec:
    """Specification for an execution task"""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher is more important
    estimated_duration: float = 0.0
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ExecutionResult:
    """Result from task execution"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    worker_id: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class WorkerStats:
    """Statistics for a worker"""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    last_activity: float = field(default_factory=time.time)

class ResourceMonitor:
    """Monitor system resource usage"""
    
    def __init__(self):
        self.cpu_usage_history: List[float] = []
        self.memory_usage_history: List[float] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage_history.append(cpu_percent)
                
                # Memory usage
                memory_info = psutil.virtual_memory()
                self.memory_usage_history.append(memory_info.percent)
                
                # Keep history bounded
                if len(self.cpu_usage_history) > 100:
                    self.cpu_usage_history.pop(0)
                if len(self.memory_usage_history) > 100:
                    self.memory_usage_history.pop(0)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def get_current_usage(self) -> Dict[ResourceType, float]:
        """Get current resource usage"""
        try:
            return {
                ResourceType.CPU: psutil.cpu_percent(),
                ResourceType.MEMORY: psutil.virtual_memory().percent,
                ResourceType.DISK_IO: psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
            }
        except Exception:
            return {ResourceType.CPU: 0.0, ResourceType.MEMORY: 0.0, ResourceType.DISK_IO: 0.0}
    
    def get_average_usage(self, window: int = 10) -> Dict[ResourceType, float]:
        """Get average resource usage over window"""
        cpu_avg = sum(self.cpu_usage_history[-window:]) / min(len(self.cpu_usage_history), window) if self.cpu_usage_history else 0.0
        memory_avg = sum(self.memory_usage_history[-window:]) / min(len(self.memory_usage_history), window) if self.memory_usage_history else 0.0
        
        return {
            ResourceType.CPU: cpu_avg,
            ResourceType.MEMORY: memory_avg
        }

class LoadBalancer:
    """Intelligent load balancer for task distribution"""
    
    def __init__(self, max_workers: int = None):
        if max_workers is None:
            max_workers = min(32, (multiprocessing.cpu_count() or 1) * 2)
        
        self.max_workers = max_workers
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.task_queue = queue.PriorityQueue()
        self.resource_monitor = ResourceMonitor()
        
    def select_worker(self, task: TaskSpec) -> str:
        """Select optimal worker for task"""
        # Simple round-robin for now, can be enhanced with more sophisticated algorithms
        active_workers = [w for w in self.worker_stats.values() if time.time() - w.last_activity < 60]
        
        if not active_workers:
            return f"worker_0"
        
        # Select worker with lowest current load
        best_worker = min(active_workers, 
                         key=lambda w: w.tasks_completed / (w.total_execution_time + 1))
        
        return best_worker.worker_id
    
    def update_worker_stats(self, worker_id: str, result: ExecutionResult):
        """Update worker statistics"""
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
        
        stats = self.worker_stats[worker_id]
        stats.last_activity = time.time()
        
        if result.success:
            stats.tasks_completed += 1
        else:
            stats.tasks_failed += 1
        
        stats.total_execution_time += result.execution_time
        total_tasks = stats.tasks_completed + stats.tasks_failed
        if total_tasks > 0:
            stats.average_execution_time = stats.total_execution_time / total_tasks

class ConcurrentExecutor:
    """
    High-performance concurrent execution engine
    
    Features:
    - Multiple execution strategies (threads, processes, async)
    - Intelligent load balancing and resource management
    - Adaptive strategy selection based on workload characteristics
    - Resource monitoring and optimization
    - Fault tolerance with automatic retries
    """
    
    def __init__(self, 
                 strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
                 max_workers: Optional[int] = None,
                 enable_monitoring: bool = True):
        """
        Initialize concurrent executor
        
        Args:
            strategy: Execution strategy to use
            max_workers: Maximum number of workers
            enable_monitoring: Enable resource monitoring
        """
        self.strategy = strategy
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) * 2)
        
        # Execution pools
        self.thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        
        # Load balancing and monitoring
        self.load_balancer = LoadBalancer(self.max_workers)
        self.resource_monitor = ResourceMonitor()
        
        if enable_monitoring:
            self.resource_monitor.start_monitoring()
        
        # Execution tracking
        self.active_tasks: Dict[str, TaskSpec] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        # Performance metrics
        self.total_tasks_executed = 0
        self.total_execution_time = 0.0
        self.strategy_performance: Dict[ExecutionStrategy, List[float]] = defaultdict(list)
        
        logger.info(f"ConcurrentExecutor initialized: {strategy.value} strategy, {self.max_workers} max workers")
    
    def execute_batch(self, tasks: List[TaskSpec], 
                     strategy_override: Optional[ExecutionStrategy] = None) -> Dict[str, ExecutionResult]:
        """
        Execute a batch of tasks concurrently
        
        Args:
            tasks: List of tasks to execute
            strategy_override: Override default execution strategy
            
        Returns:
            Dictionary mapping task IDs to execution results
        """
        if not tasks:
            return {}
        
        strategy = strategy_override or self._select_optimal_strategy(tasks)
        logger.info(f"Executing {len(tasks)} tasks using {strategy.value} strategy")
        
        start_time = time.time()
        
        # Build dependency graph
        self._build_dependency_graph(tasks)
        
        # Execute tasks based on strategy
        if strategy == ExecutionStrategy.SEQUENTIAL:
            results = self._execute_sequential(tasks)
        elif strategy == ExecutionStrategy.THREADED:
            results = self._execute_threaded(tasks)
        elif strategy == ExecutionStrategy.PROCESS_POOL:
            results = self._execute_process_pool(tasks)
        elif strategy == ExecutionStrategy.ASYNC_IO:
            results = self._execute_async_io(tasks)
        elif strategy == ExecutionStrategy.HYBRID:
            results = self._execute_hybrid(tasks)
        else:  # ADAPTIVE
            results = self._execute_adaptive(tasks)
        
        batch_time = time.time() - start_time
        self.total_tasks_executed += len(tasks)
        self.total_execution_time += batch_time
        
        # Record strategy performance
        avg_task_time = batch_time / len(tasks)
        self.strategy_performance[strategy].append(avg_task_time)
        
        logger.info(f"Batch execution completed in {batch_time:.2f}s, "
                   f"avg {avg_task_time:.3f}s per task")
        
        return results
    
    def _select_optimal_strategy(self, tasks: List[TaskSpec]) -> ExecutionStrategy:
        """Select optimal execution strategy based on task characteristics"""
        if self.strategy != ExecutionStrategy.ADAPTIVE:
            return self.strategy
        
        # Analyze task characteristics
        total_estimated_time = sum(task.estimated_duration for task in tasks)
        io_heavy_tasks = sum(1 for task in tasks 
                           if ResourceType.DISK_IO in task.resource_requirements or 
                              ResourceType.NETWORK_IO in task.resource_requirements)
        cpu_heavy_tasks = sum(1 for task in tasks 
                            if task.resource_requirements.get(ResourceType.CPU, 0) > 0.5)
        
        # Simple heuristics for strategy selection
        if len(tasks) == 1:
            return ExecutionStrategy.SEQUENTIAL
        
        if io_heavy_tasks > len(tasks) * 0.7:  # Mostly I/O bound
            return ExecutionStrategy.ASYNC_IO if total_estimated_time > 1.0 else ExecutionStrategy.THREADED
        
        if cpu_heavy_tasks > len(tasks) * 0.7:  # Mostly CPU bound
            return ExecutionStrategy.PROCESS_POOL if len(tasks) > 4 else ExecutionStrategy.THREADED
        
        # Mixed workload
        return ExecutionStrategy.HYBRID
    
    def _build_dependency_graph(self, tasks: List[TaskSpec]):
        """Build task dependency graph"""
        self.task_dependencies.clear()
        
        for task in tasks:
            self.task_dependencies[task.task_id] = task.dependencies.copy()
    
    def _get_ready_tasks(self, tasks: List[TaskSpec], completed: set) -> List[TaskSpec]:
        """Get tasks that are ready to execute (dependencies satisfied)"""
        ready = []
        
        for task in tasks:
            if task.task_id in completed:
                continue
            
            dependencies_met = all(dep in completed for dep in task.dependencies)
            if dependencies_met:
                ready.append(task)
        
        return ready
    
    def _execute_sequential(self, tasks: List[TaskSpec]) -> Dict[str, ExecutionResult]:
        """Execute tasks sequentially"""
        results = {}
        completed = set()
        
        # Sort by priority and estimated duration
        sorted_tasks = sorted(tasks, key=lambda t: (-t.priority, t.estimated_duration))
        
        while len(completed) < len(tasks):
            ready_tasks = self._get_ready_tasks(sorted_tasks, completed)
            
            if not ready_tasks:
                # Handle circular dependencies or missing dependencies
                remaining = [t for t in sorted_tasks if t.task_id not in completed]
                if remaining:
                    ready_tasks = [remaining[0]]  # Force execute one task
            
            for task in ready_tasks:
                result = self._execute_single_task(task)
                results[task.task_id] = result
                completed.add(task.task_id)
                break  # Execute one at a time in sequential mode
        
        return results
    
    def _execute_threaded(self, tasks: List[TaskSpec]) -> Dict[str, ExecutionResult]:
        """Execute tasks using thread pool"""
        if not self.thread_pool:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="edge_tpu_worker"
            )
        
        results = {}
        completed = set()
        futures = {}
        
        try:
            while len(completed) < len(tasks):
                # Submit ready tasks
                ready_tasks = self._get_ready_tasks(tasks, completed)
                
                for task in ready_tasks:
                    if task.task_id not in futures:
                        future = self.thread_pool.submit(self._execute_single_task, task)
                        futures[task.task_id] = future
                
                # Wait for at least one task to complete
                if futures:
                    done_futures = concurrent.futures.as_completed(futures.values(), timeout=1.0)
                    
                    for future in done_futures:
                        # Find task ID for this future
                        task_id = None
                        for tid, f in futures.items():
                            if f == future:
                                task_id = tid
                                break
                        
                        if task_id:
                            try:
                                result = future.result()
                                results[task_id] = result
                                completed.add(task_id)
                                del futures[task_id]
                            except Exception as e:
                                error_result = ExecutionResult(
                                    task_id=task_id,
                                    success=False,
                                    error=e
                                )
                                results[task_id] = error_result
                                completed.add(task_id)
                                del futures[task_id]
                        
                        break  # Process one completion at a time
        
        except concurrent.futures.TimeoutError:
            # Handle timeout - continue with available results
            pass
        
        return results
    
    def _execute_process_pool(self, tasks: List[TaskSpec]) -> Dict[str, ExecutionResult]:
        """Execute tasks using process pool"""
        if not self.process_pool:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(self.max_workers, multiprocessing.cpu_count())
            )
        
        # Note: Process pool execution is more complex due to serialization requirements
        # For now, fall back to threaded execution
        logger.info("Process pool execution not fully implemented, falling back to threaded")
        return self._execute_threaded(tasks)
    
    def _execute_async_io(self, tasks: List[TaskSpec]) -> Dict[str, ExecutionResult]:
        """Execute tasks using async I/O"""
        async def _async_execute():
            results = {}
            completed = set()
            
            while len(completed) < len(tasks):
                ready_tasks = self._get_ready_tasks(tasks, completed)
                
                # Convert sync functions to async
                async_tasks = []
                for task in ready_tasks:
                    if task.task_id not in completed:
                        async_task = self._make_async(task)
                        async_tasks.append(async_task)
                
                if async_tasks:
                    # Execute tasks concurrently
                    task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(task_results):
                        task = ready_tasks[i]
                        if isinstance(result, Exception):
                            error_result = ExecutionResult(
                                task_id=task.task_id,
                                success=False,
                                error=result
                            )
                            results[task.task_id] = error_result
                        else:
                            results[task.task_id] = result
                        
                        completed.add(task.task_id)
                else:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            
            return results
        
        # Run async execution
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(_async_execute())
    
    def _execute_hybrid(self, tasks: List[TaskSpec]) -> Dict[str, ExecutionResult]:
        """Execute tasks using hybrid strategy"""
        # Separate tasks by type
        io_tasks = []
        cpu_tasks = []
        simple_tasks = []
        
        for task in tasks:
            if (ResourceType.DISK_IO in task.resource_requirements or 
                ResourceType.NETWORK_IO in task.resource_requirements):
                io_tasks.append(task)
            elif task.resource_requirements.get(ResourceType.CPU, 0) > 0.5:
                cpu_tasks.append(task)
            else:
                simple_tasks.append(task)
        
        results = {}
        
        # Execute I/O tasks with async
        if io_tasks:
            results.update(self._execute_async_io(io_tasks))
        
        # Execute CPU tasks with process pool
        if cpu_tasks:
            results.update(self._execute_process_pool(cpu_tasks))
        
        # Execute simple tasks with threads
        if simple_tasks:
            results.update(self._execute_threaded(simple_tasks))
        
        return results
    
    def _execute_adaptive(self, tasks: List[TaskSpec]) -> Dict[str, ExecutionResult]:
        """Execute tasks with adaptive strategy selection"""
        # Start with optimal strategy selection
        optimal_strategy = self._select_optimal_strategy(tasks)
        
        # If we have historical performance data, consider switching
        if optimal_strategy in self.strategy_performance:
            performance_history = self.strategy_performance[optimal_strategy]
            if len(performance_history) > 5:
                # If recent performance is poor, try alternative strategy
                recent_avg = sum(performance_history[-3:]) / 3
                overall_avg = sum(performance_history) / len(performance_history)
                
                if recent_avg > overall_avg * 1.5:  # Recent performance is 50% worse
                    # Try alternative strategy
                    alternatives = [
                        ExecutionStrategy.THREADED,
                        ExecutionStrategy.ASYNC_IO,
                        ExecutionStrategy.HYBRID
                    ]
                    
                    alternative = next((s for s in alternatives if s != optimal_strategy), optimal_strategy)
                    logger.info(f"Switching from {optimal_strategy.value} to {alternative.value} due to poor recent performance")
                    optimal_strategy = alternative
        
        # Execute with selected strategy
        return self.execute_batch(tasks, strategy_override=optimal_strategy)
    
    async def _make_async(self, task: TaskSpec) -> ExecutionResult:
        """Convert sync task to async execution"""
        loop = asyncio.get_event_loop()
        
        # Execute in thread pool to avoid blocking the event loop
        result = await loop.run_in_executor(None, self._execute_single_task, task)
        return result
    
    def _execute_single_task(self, task: TaskSpec) -> ExecutionResult:
        """Execute a single task and return result"""
        start_time = time.time()
        worker_id = self.load_balancer.select_worker(task)
        
        try:
            # Track active task
            self.active_tasks[task.task_id] = task
            
            # Execute task with timeout
            if task.timeout:
                # Use threading for timeout
                result_queue = queue.Queue()
                
                def target():
                    try:
                        result = task.function(*task.args, **task.kwargs)
                        result_queue.put(('success', result))
                    except Exception as e:
                        result_queue.put(('error', e))
                
                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout=task.timeout)
                
                if thread.is_alive():
                    # Task timed out
                    raise TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")
                
                try:
                    status, result = result_queue.get_nowait()
                    if status == 'error':
                        raise result
                except queue.Empty:
                    raise RuntimeError(f"Task {task.task_id} completed but no result available")
            else:
                # Execute directly
                result = task.function(*task.args, **task.kwargs)
            
            # Success
            execution_time = time.time() - start_time
            execution_result = ExecutionResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id,
                resource_usage=self.resource_monitor.get_current_usage()
            )
            
            # Update statistics
            self.load_balancer.update_worker_stats(worker_id, execution_result)
            
            return execution_result
            
        except Exception as e:
            # Handle failure
            execution_time = time.time() - start_time
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"Task {task.task_id} failed (attempt {task.retry_count}), retrying: {e}")
                time.sleep(0.5 * task.retry_count)  # Exponential backoff
                return self._execute_single_task(task)
            
            execution_result = ExecutionResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                worker_id=worker_id
            )
            
            self.load_balancer.update_worker_stats(worker_id, execution_result)
            return execution_result
            
        finally:
            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_tasks = self.total_tasks_executed
        avg_execution_time = self.total_execution_time / max(1, total_tasks)
        
        # Strategy performance comparison
        strategy_stats = {}
        for strategy, times in self.strategy_performance.items():
            if times:
                strategy_stats[strategy.value] = {
                    'executions': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        # Resource usage
        resource_usage = self.resource_monitor.get_average_usage()
        
        # Worker statistics
        worker_stats = {
            worker_id: {
                'tasks_completed': stats.tasks_completed,
                'tasks_failed': stats.tasks_failed,
                'success_rate': stats.tasks_completed / max(1, stats.tasks_completed + stats.tasks_failed),
                'avg_execution_time': stats.average_execution_time
            }
            for worker_id, stats in self.load_balancer.worker_stats.items()
        }
        
        return {
            'total_tasks_executed': total_tasks,
            'average_execution_time': avg_execution_time,
            'active_tasks': len(self.active_tasks),
            'strategy_performance': strategy_stats,
            'resource_usage': {rt.value: usage for rt, usage in resource_usage.items()},
            'worker_statistics': worker_stats,
            'current_strategy': self.strategy.value
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor and clean up resources"""
        logger.info("Shutting down concurrent executor...")
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=wait)
        
        # Shutdown process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=wait)
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        logger.info("Concurrent executor shutdown complete")

# Global concurrent executor instance
global_concurrent_executor = ConcurrentExecutor()