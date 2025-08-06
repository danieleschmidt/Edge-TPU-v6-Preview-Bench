"""
Performance Optimizer - Advanced scaling and caching for quantum task planning
Implements distributed computing, caching strategies, and performance monitoring
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue, cpu_count
import json
import pickle
import hashlib
from functools import lru_cache, wraps
from collections import defaultdict
import weakref
import gc
import psutil
import redis
from contextlib import contextmanager

from .quantum_task_engine import QuantumTask, Priority, TaskState

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    MEMORY_ONLY = "memory_only"
    REDIS_DISTRIBUTED = "redis_distributed"
    HYBRID_TIERED = "hybrid_tiered"
    ADAPTIVE = "adaptive"

class ScalingMode(Enum):
    SINGLE_THREADED = "single_threaded"
    MULTI_THREADED = "multi_threaded"
    MULTI_PROCESS = "multi_process"
    DISTRIBUTED_CLUSTER = "distributed_cluster"
    AUTO_SCALE = "auto_scale"

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_tasks_per_second: float = 0.0
    parallelism_efficiency: float = 0.0
    network_latency_ms: float = 0.0
    disk_io_mb_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_utilization_percent': self.cpu_utilization_percent,
            'cache_hit_rate': self.cache_hit_rate,
            'throughput_tasks_per_second': self.throughput_tasks_per_second,
            'parallelism_efficiency': self.parallelism_efficiency,
            'network_latency_ms': self.network_latency_ms,
            'disk_io_mb_per_second': self.disk_io_mb_per_second
        }

@dataclass
class ScalingConfig:
    """Configuration for performance scaling"""
    mode: ScalingMode = ScalingMode.AUTO_SCALE
    max_workers: int = field(default_factory=lambda: cpu_count())
    memory_limit_mb: int = 8192  # 8GB default
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    enable_monitoring: bool = True
    auto_gc_threshold: int = 1000  # Tasks processed before garbage collection
    prefetch_size: int = 100
    batch_size: int = 50
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    distributed_nodes: List[str] = field(default_factory=list)
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, hash_based
    
    def __post_init__(self):
        """Validate scaling configuration"""
        if self.max_workers <= 0:
            self.max_workers = cpu_count()
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

class QuantumCache:
    """
    High-performance caching system for quantum task planning
    
    Features:
    - Multi-tiered caching (memory -> Redis -> disk)
    - Automatic cache invalidation
    - LRU eviction policies
    - Cache warming and prefetching
    - Performance monitoring
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.strategy = config.cache_strategy
        
        # Memory cache
        self._memory_cache: Dict[str, Any] = {}
        self._memory_access_times: Dict[str, float] = {}
        self._memory_hit_count = 0
        self._memory_miss_count = 0
        self._memory_lock = threading.RLock()
        
        # Redis cache
        self._redis_client: Optional[redis.Redis] = None
        self._redis_hit_count = 0
        self._redis_miss_count = 0
        
        # Cache statistics
        self._cache_stats = {
            'total_gets': 0,
            'total_sets': 0,
            'memory_hits': 0,
            'redis_hits': 0,
            'misses': 0
        }
        
        self._initialize_cache()
        
        logger.info(f"QuantumCache initialized with strategy: {self.strategy.value}")
    
    def _initialize_cache(self):
        """Initialize cache backends"""
        try:
            if self.strategy in [CacheStrategy.REDIS_DISTRIBUTED, CacheStrategy.HYBRID_TIERED, CacheStrategy.ADAPTIVE]:
                self._redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=False,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0
                )
                
                # Test Redis connection
                try:
                    self._redis_client.ping()
                    logger.info("Redis cache backend connected successfully")
                except redis.ConnectionError:
                    logger.warning("Redis connection failed, falling back to memory-only cache")
                    self._redis_client = None
                    if self.strategy == CacheStrategy.REDIS_DISTRIBUTED:
                        self.strategy = CacheStrategy.MEMORY_ONLY
                    
        except Exception as e:
            logger.error(f"Failed to initialize cache backends: {e}")
            self.strategy = CacheStrategy.MEMORY_ONLY
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with multi-tier lookup"""
        self._cache_stats['total_gets'] += 1
        
        try:
            # Try memory cache first
            with self._memory_lock:
                if key in self._memory_cache:
                    self._memory_access_times[key] = time.time()
                    self._memory_hit_count += 1
                    self._cache_stats['memory_hits'] += 1
                    return self._memory_cache[key]
            
            # Try Redis cache if available
            if self._redis_client and self.strategy != CacheStrategy.MEMORY_ONLY:
                try:
                    cached_data = self._redis_client.get(key)
                    if cached_data:
                        value = pickle.loads(cached_data)
                        
                        # Promote to memory cache
                        self._set_memory_cache(key, value)
                        
                        self._redis_hit_count += 1
                        self._cache_stats['redis_hits'] += 1
                        return value
                        
                except (redis.ConnectionError, pickle.PickleError) as e:
                    logger.warning(f"Redis cache error: {e}")
            
            # Cache miss
            self._cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache with multi-tier storage"""
        self._cache_stats['total_sets'] += 1
        
        try:
            success = False
            
            # Set in memory cache
            if self.strategy in [CacheStrategy.MEMORY_ONLY, CacheStrategy.HYBRID_TIERED, CacheStrategy.ADAPTIVE]:
                success = self._set_memory_cache(key, value)
            
            # Set in Redis cache
            if self._redis_client and self.strategy != CacheStrategy.MEMORY_ONLY:
                try:
                    serialized_value = pickle.dumps(value)
                    if ttl:
                        self._redis_client.setex(key, ttl, serialized_value)
                    else:
                        self._redis_client.set(key, serialized_value)
                    success = True
                    
                except (redis.ConnectionError, pickle.PickleError) as e:
                    logger.warning(f"Redis cache set error: {e}")
            
            return success
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def _set_memory_cache(self, key: str, value: Any) -> bool:
        """Set item in memory cache with LRU eviction"""
        try:
            with self._memory_lock:
                # Check memory usage and evict if necessary
                if len(self._memory_cache) >= 1000:  # Max memory cache size
                    self._evict_lru_memory_items()
                
                self._memory_cache[key] = value
                self._memory_access_times[key] = time.time()
                
            return True
            
        except Exception as e:
            logger.warning(f"Memory cache set error: {e}")
            return False
    
    def _evict_lru_memory_items(self):
        """Evict least recently used items from memory cache"""
        try:
            # Sort by access time and remove oldest 20%
            sorted_items = sorted(
                self._memory_access_times.items(),
                key=lambda x: x[1]
            )
            
            items_to_remove = int(len(sorted_items) * 0.2)
            for key, _ in sorted_items[:items_to_remove]:
                self._memory_cache.pop(key, None)
                self._memory_access_times.pop(key, None)
                
        except Exception as e:
            logger.warning(f"Memory cache eviction error: {e}")
    
    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries by pattern"""
        invalidated = 0
        
        try:
            if pattern is None:
                # Clear all caches
                with self._memory_lock:
                    invalidated += len(self._memory_cache)
                    self._memory_cache.clear()
                    self._memory_access_times.clear()
                
                if self._redis_client:
                    try:
                        self._redis_client.flushdb()
                    except redis.ConnectionError:
                        pass
                        
            else:
                # Pattern-based invalidation
                keys_to_remove = []
                
                with self._memory_lock:
                    for key in self._memory_cache.keys():
                        if pattern in key:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        self._memory_cache.pop(key, None)
                        self._memory_access_times.pop(key, None)
                        invalidated += 1
                
                if self._redis_client:
                    try:
                        # Redis pattern deletion
                        keys = self._redis_client.keys(f"*{pattern}*")
                        if keys:
                            self._redis_client.delete(*keys)
                            invalidated += len(keys)
                    except redis.ConnectionError:
                        pass
            
            return invalidated
            
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_stats['total_gets']
        
        stats = self._cache_stats.copy()
        stats.update({
            'memory_size': len(self._memory_cache),
            'hit_rate': (stats['memory_hits'] + stats['redis_hits']) / max(1, total_requests),
            'memory_hit_rate': stats['memory_hits'] / max(1, total_requests),
            'redis_hit_rate': stats['redis_hits'] / max(1, total_requests),
            'miss_rate': stats['misses'] / max(1, total_requests)
        })
        
        return stats

def performance_monitor(func: Callable) -> Callable:
    """Decorator for performance monitoring"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_utilization_percent=end_cpu - start_cpu
            )
            
            # Store metrics if the function has a performance tracker
            if hasattr(func, '__self__') and hasattr(func.__self__, '_performance_metrics'):
                func.__self__._performance_metrics.append(metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Performance monitoring error in {func.__name__}: {e}")
            raise
    
    return wrapper

class ParallelExecutor:
    """
    High-performance parallel execution engine for quantum tasks
    
    Features:
    - Adaptive scaling based on workload
    - Load balancing across workers
    - Memory management and cleanup
    - Performance monitoring
    - Failure recovery
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.mode = config.mode
        self._performance_metrics: List[PerformanceMetrics] = []
        self._active_workers = 0
        self._task_queue = Queue() if config.mode == ScalingMode.MULTI_PROCESS else asyncio.Queue()
        self._results_cache = QuantumCache(config)
        self._executor_pool: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        
        # Load balancing
        self._worker_loads: Dict[int, int] = defaultdict(int)
        self._worker_performance: Dict[int, List[float]] = defaultdict(list)
        
        # Auto-scaling parameters
        self._last_scale_time = time.time()
        self._scale_check_interval = 10.0  # seconds
        self._load_threshold_high = 0.8
        self._load_threshold_low = 0.3
        
        self._initialize_executor()
        
        logger.info(f"ParallelExecutor initialized with mode: {self.mode.value}, max_workers: {config.max_workers}")
    
    def _initialize_executor(self):
        """Initialize the appropriate executor pool"""
        try:
            if self.mode == ScalingMode.MULTI_THREADED:
                self._executor_pool = ThreadPoolExecutor(
                    max_workers=self.config.max_workers,
                    thread_name_prefix="QuantumTask"
                )
            elif self.mode == ScalingMode.MULTI_PROCESS:
                self._executor_pool = ProcessPoolExecutor(
                    max_workers=self.config.max_workers
                )
            elif self.mode == ScalingMode.AUTO_SCALE:
                # Start with thread pool, can switch to process pool if needed
                self._executor_pool = ThreadPoolExecutor(
                    max_workers=min(4, self.config.max_workers),
                    thread_name_prefix="QuantumTask"
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize executor: {e}")
            self.mode = ScalingMode.SINGLE_THREADED
    
    @performance_monitor
    async def execute_tasks_parallel(self, 
                                   tasks: Dict[str, QuantumTask],
                                   task_functions: Dict[str, Callable],
                                   dependencies: Dict[str, Set[str]]) -> Dict[str, Any]:
        """
        Execute tasks in parallel with dependency resolution
        
        Args:
            tasks: Dictionary of tasks to execute
            task_functions: Dictionary mapping task IDs to functions
            dependencies: Task dependency graph
            
        Returns:
            Dictionary of task results
        """
        if not tasks:
            return {}
        
        try:
            results = {}
            completed_tasks = set()
            failed_tasks = set()
            task_futures = {}
            
            # Create execution batches based on dependencies
            execution_batches = self._create_execution_batches(tasks, dependencies)
            
            for batch_idx, batch in enumerate(execution_batches):
                logger.debug(f"Executing batch {batch_idx + 1}/{len(execution_batches)} with {len(batch)} tasks")
                
                if self.mode == ScalingMode.SINGLE_THREADED:
                    # Sequential execution
                    batch_results = await self._execute_batch_sequential(batch, tasks, task_functions)
                else:
                    # Parallel execution
                    batch_results = await self._execute_batch_parallel(batch, tasks, task_functions)
                
                # Process batch results
                for task_id, result in batch_results.items():
                    if isinstance(result, Exception):
                        failed_tasks.add(task_id)
                        logger.error(f"Task {task_id} failed: {result}")
                    else:
                        completed_tasks.add(task_id)
                        results[task_id] = result
                
                # Auto-scaling check
                if self.mode == ScalingMode.AUTO_SCALE:
                    await self._check_auto_scaling()
                
                # Memory cleanup
                if batch_idx % 10 == 0:  # Every 10 batches
                    self._cleanup_memory()
            
            # Final performance metrics
            final_metrics = PerformanceMetrics(
                throughput_tasks_per_second=len(completed_tasks) / max(0.1, sum(m.execution_time for m in self._performance_metrics[-10:])),
                parallelism_efficiency=len(completed_tasks) / max(1, len(tasks))
            )
            
            self._performance_metrics.append(final_metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise
    
    def _create_execution_batches(self, 
                                 tasks: Dict[str, QuantumTask], 
                                 dependencies: Dict[str, Set[str]]) -> List[List[str]]:
        """Create execution batches respecting dependencies"""
        try:
            batches = []
            remaining_tasks = set(tasks.keys())
            completed_tasks = set()
            
            while remaining_tasks:
                # Find tasks with no pending dependencies
                ready_tasks = []
                for task_id in remaining_tasks:
                    task_deps = dependencies.get(task_id, set())
                    if task_deps.issubset(completed_tasks):
                        ready_tasks.append(task_id)
                
                if not ready_tasks:
                    # Circular dependency or missing dependency
                    logger.warning(f"Circular dependency detected or missing dependencies. Remaining tasks: {remaining_tasks}")
                    ready_tasks = list(remaining_tasks)  # Force execution
                
                # Create batch with size limit
                batch_size = min(len(ready_tasks), self.config.batch_size)
                batch = ready_tasks[:batch_size]
                batches.append(batch)
                
                # Update tracking sets
                for task_id in batch:
                    remaining_tasks.remove(task_id)
                    completed_tasks.add(task_id)
            
            return batches
            
        except Exception as e:
            logger.error(f"Error creating execution batches: {e}")
            return [[task_id] for task_id in tasks.keys()]  # Fallback to single-task batches
    
    async def _execute_batch_sequential(self, 
                                       batch: List[str], 
                                       tasks: Dict[str, QuantumTask], 
                                       task_functions: Dict[str, Callable]) -> Dict[str, Any]:
        """Execute batch sequentially"""
        results = {}
        
        for task_id in batch:
            try:
                if task_id not in tasks or task_id not in task_functions:
                    results[task_id] = Exception(f"Task {task_id} not found")
                    continue
                
                # Check cache first
                cache_key = f"task_result_{task_id}_{hash(str(tasks[task_id]))}"
                cached_result = self._results_cache.get(cache_key)
                
                if cached_result is not None:
                    results[task_id] = cached_result
                    continue
                
                # Execute task
                task_function = task_functions[task_id]
                
                if asyncio.iscoroutinefunction(task_function):
                    result = await task_function()
                else:
                    result = task_function()
                
                results[task_id] = result
                
                # Cache result
                self._results_cache.set(cache_key, result, ttl=3600)  # 1 hour TTL
                
            except Exception as e:
                results[task_id] = e
        
        return results
    
    async def _execute_batch_parallel(self, 
                                     batch: List[str], 
                                     tasks: Dict[str, QuantumTask], 
                                     task_functions: Dict[str, Callable]) -> Dict[str, Any]:
        """Execute batch in parallel"""
        if not self._executor_pool:
            return await self._execute_batch_sequential(batch, tasks, task_functions)
        
        results = {}
        futures = {}
        
        try:
            # Submit tasks to executor pool
            for task_id in batch:
                if task_id not in tasks or task_id not in task_functions:
                    results[task_id] = Exception(f"Task {task_id} not found")
                    continue
                
                # Check cache first
                cache_key = f"task_result_{task_id}_{hash(str(tasks[task_id]))}"
                cached_result = self._results_cache.get(cache_key)
                
                if cached_result is not None:
                    results[task_id] = cached_result
                    continue
                
                # Submit to executor
                task_function = task_functions[task_id]
                future = self._executor_pool.submit(self._execute_single_task, task_id, task_function)
                futures[future] = task_id
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=300):  # 5 minute timeout
                task_id = futures[future]
                
                try:
                    result = future.result(timeout=60)  # 1 minute per task
                    results[task_id] = result
                    
                    # Cache result
                    cache_key = f"task_result_{task_id}_{hash(str(tasks[task_id]))}"
                    self._results_cache.set(cache_key, result, ttl=3600)
                    
                except Exception as e:
                    results[task_id] = e
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel batch execution error: {e}")
            
            # Cancel pending futures
            for future in futures:
                future.cancel()
            
            # Fallback to sequential execution
            return await self._execute_batch_sequential(batch, tasks, task_functions)
    
    def _execute_single_task(self, task_id: str, task_function: Callable) -> Any:
        """Execute a single task (for use in executor pool)"""
        try:
            if asyncio.iscoroutinefunction(task_function):
                # Handle async functions in executor
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(task_function())
                finally:
                    loop.close()
            else:
                result = task_function()
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task_id} execution failed: {e}")
            raise
    
    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed"""
        if self.mode != ScalingMode.AUTO_SCALE:
            return
        
        current_time = time.time()
        if current_time - self._last_scale_time < self._scale_check_interval:
            return
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Determine if scaling is needed
            scale_up = (cpu_percent > 80 or memory_percent > 80) and self._active_workers < self.config.max_workers
            scale_down = (cpu_percent < 30 and memory_percent < 30) and self._active_workers > 1
            
            if scale_up:
                await self._scale_up()
            elif scale_down:
                await self._scale_down()
            
            self._last_scale_time = current_time
            
        except Exception as e:
            logger.warning(f"Auto-scaling check failed: {e}")
    
    async def _scale_up(self):
        """Scale up the executor pool"""
        try:
            if isinstance(self._executor_pool, ThreadPoolExecutor):
                # Create new thread pool with more workers
                old_pool = self._executor_pool
                new_max_workers = min(self.config.max_workers, self._active_workers + 2)
                
                self._executor_pool = ThreadPoolExecutor(
                    max_workers=new_max_workers,
                    thread_name_prefix="QuantumTask"
                )
                
                # Shutdown old pool gracefully
                old_pool.shutdown(wait=False)
                
                logger.info(f"Scaled up to {new_max_workers} workers")
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
    
    async def _scale_down(self):
        """Scale down the executor pool"""
        try:
            if isinstance(self._executor_pool, ThreadPoolExecutor):
                # Create new thread pool with fewer workers
                old_pool = self._executor_pool
                new_max_workers = max(1, self._active_workers - 1)
                
                self._executor_pool = ThreadPoolExecutor(
                    max_workers=new_max_workers,
                    thread_name_prefix="QuantumTask"
                )
                
                # Shutdown old pool gracefully
                old_pool.shutdown(wait=False)
                
                logger.info(f"Scaled down to {new_max_workers} workers")
            
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
    
    def _cleanup_memory(self):
        """Perform memory cleanup and garbage collection"""
        try:
            # Clear old performance metrics
            if len(self._performance_metrics) > 1000:
                self._performance_metrics = self._performance_metrics[-500:]  # Keep last 500
            
            # Cache cleanup
            if hasattr(self._results_cache, 'cleanup'):
                self._results_cache.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("Memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            if not self._performance_metrics:
                return {'message': 'No performance data available'}
            
            recent_metrics = self._performance_metrics[-100:]  # Last 100 measurements
            
            avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
            avg_memory_usage = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput_tasks_per_second for m in recent_metrics) / len(recent_metrics)
            
            cache_stats = self._results_cache.get_stats()
            
            return {
                'mode': self.mode.value,
                'max_workers': self.config.max_workers,
                'active_workers': self._active_workers,
                'average_execution_time': avg_execution_time,
                'average_memory_usage_mb': avg_memory_usage,
                'average_throughput': avg_throughput,
                'cache_stats': cache_stats,
                'total_measurements': len(self._performance_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown the executor gracefully"""
        try:
            if self._executor_pool:
                self._executor_pool.shutdown(wait=True, timeout=30)
                logger.info("ParallelExecutor shutdown completed")
            
        except Exception as e:
            logger.error(f"Executor shutdown error: {e}")

class PerformanceOptimizer:
    """
    Main performance optimization coordinator
    
    Integrates caching, parallel execution, and monitoring
    for maximum quantum task planning performance
    """
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.cache = QuantumCache(self.config)
        self.executor = ParallelExecutor(self.config)
        
        # Global performance monitoring
        self._global_metrics: List[PerformanceMetrics] = []
        self._optimization_start_time = time.time()
        
        logger.info("PerformanceOptimizer initialized successfully")
    
    @performance_monitor
    async def optimize_task_execution(self,
                                    tasks: Dict[str, QuantumTask],
                                    task_functions: Dict[str, Callable],
                                    dependencies: Dict[str, Set[str]]) -> Dict[str, Any]:
        """
        Optimize and execute tasks with full performance enhancements
        
        Args:
            tasks: Dictionary of tasks to execute
            task_functions: Dictionary mapping task IDs to functions
            dependencies: Task dependency graph
            
        Returns:
            Dictionary of task execution results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Prefetch and warm cache
            await self._warm_cache(tasks)
            
            # Execute tasks with parallel optimization
            results = await self.executor.execute_tasks_parallel(tasks, task_functions, dependencies)
            
            # Calculate final performance metrics
            execution_time = time.time() - start_time
            
            performance_report = {
                'results': results,
                'execution_time': execution_time,
                'tasks_completed': len([r for r in results.values() if not isinstance(r, Exception)]),
                'tasks_failed': len([r for r in results.values() if isinstance(r, Exception)]),
                'cache_stats': self.cache.get_stats(),
                'executor_stats': self.executor.get_performance_report()
            }
            
            logger.info(f"Task execution optimized: {len(results)} tasks in {execution_time:.2f}s")
            
            return performance_report
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            raise
    
    async def _warm_cache(self, tasks: Dict[str, QuantumTask]):
        """Warm cache with frequently accessed task data"""
        try:
            # Pre-compute and cache common task metrics
            for task_id, task in tasks.items():
                cache_key = f"task_metadata_{task_id}"
                
                if self.cache.get(cache_key) is None:
                    metadata = {
                        'priority_score': task.priority.value * task.quantum_weight,
                        'dependency_count': len(task.dependencies),
                        'estimated_duration': task.estimated_duration
                    }
                    
                    self.cache.set(cache_key, metadata, ttl=1800)  # 30 minutes
            
            logger.debug(f"Cache warmed for {len(tasks)} tasks")
            
        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance optimization report"""
        try:
            uptime = time.time() - self._optimization_start_time
            
            return {
                'uptime_seconds': uptime,
                'configuration': {
                    'scaling_mode': self.config.mode.value,
                    'max_workers': self.config.max_workers,
                    'cache_strategy': self.config.cache_strategy.value,
                    'memory_limit_mb': self.config.memory_limit_mb
                },
                'cache_performance': self.cache.get_stats(),
                'executor_performance': self.executor.get_performance_report(),
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage_percent': psutil.disk_usage('/').percent
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown performance optimizer gracefully"""
        try:
            self.executor.shutdown()
            logger.info("PerformanceOptimizer shutdown completed")
            
        except Exception as e:
            logger.error(f"Performance optimizer shutdown error: {e}")