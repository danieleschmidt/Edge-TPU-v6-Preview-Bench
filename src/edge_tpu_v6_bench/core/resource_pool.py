"""
Advanced resource pooling and management for Edge TPU v6 benchmarking
Intelligent resource allocation, load balancing, and auto-scaling
"""

import time
import threading
import logging
import queue
import weakref
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import concurrent.futures
import uuid

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for pooled resources

class ResourceStatus(Enum):
    """Resource status states"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    RETIRED = "retired"

class PoolStrategy(Enum):
    """Pool management strategies"""
    FIXED_SIZE = "fixed_size"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    LOAD_BALANCED = "load_balanced"

@dataclass
class ResourceMetrics:
    """Resource performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    total_usage_time: float = 0.0
    peak_concurrent_usage: int = 0
    error_rate: float = 0.0

@dataclass
class PooledResource(Generic[T]):
    """Wrapper for pooled resources"""
    resource_id: str
    resource: T
    status: ResourceStatus
    metrics: ResourceMetrics
    created_at: float
    last_health_check: float
    health_check_function: Optional[Callable[[T], bool]] = None
    cleanup_function: Optional[Callable[[T], None]] = None
    priority: int = 0  # Higher priority resources are preferred
    tags: Set[str] = field(default_factory=set)

class ResourcePool(Generic[T]):
    """
    High-performance resource pool with intelligent management
    
    Features:
    - Automatic resource creation and cleanup
    - Health monitoring and recovery
    - Load balancing and intelligent allocation
    - Dynamic scaling based on demand
    - Resource prioritization and tagging
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self,
                 resource_factory: Callable[[], T],
                 min_size: int = 1,
                 max_size: int = 10,
                 strategy: PoolStrategy = PoolStrategy.ADAPTIVE,
                 health_check_function: Optional[Callable[[T], bool]] = None,
                 cleanup_function: Optional[Callable[[T], None]] = None,
                 health_check_interval: float = 60.0,
                 idle_timeout: float = 300.0):
        """
        Initialize resource pool
        
        Args:
            resource_factory: Function to create new resources
            min_size: Minimum pool size
            max_size: Maximum pool size
            strategy: Pool management strategy
            health_check_function: Function to check resource health
            cleanup_function: Function to cleanup resources
            health_check_interval: Health check interval in seconds
            idle_timeout: Timeout for idle resources in seconds
        """
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.strategy = strategy
        self.health_check_function = health_check_function
        self.cleanup_function = cleanup_function
        self.health_check_interval = health_check_interval
        self.idle_timeout = idle_timeout
        
        # Resource management
        self.resources: Dict[str, PooledResource[T]] = {}
        self.available_queue = queue.PriorityQueue()  # (priority, resource_id)
        self.in_use: Set[str] = set()
        
        # Threading
        self.pool_lock = threading.RLock()
        self.condition = threading.Condition(self.pool_lock)
        
        # Background threads
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.resource_manager_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Pool statistics
        self.pool_metrics = {
            'total_resources_created': 0,
            'total_resources_destroyed': 0,
            'current_pool_size': 0,
            'peak_pool_size': 0,
            'total_acquisitions': 0,
            'average_acquisition_time': 0.0,
            'failed_acquisitions': 0,
            'resources_recovered': 0
        }
        
        # Load balancing
        self.load_balancer = ResourceLoadBalancer()
        
        # Initialize pool
        self._initialize_pool()
        self._start_background_threads()
        
        logger.info(f"ResourcePool initialized: {strategy.value} strategy, "
                   f"size range [{min_size}-{max_size}]")
    
    def _initialize_pool(self):
        """Initialize pool with minimum resources"""
        with self.pool_lock:
            for _ in range(self.min_size):
                try:
                    self._create_resource()
                except Exception as e:
                    logger.error(f"Failed to initialize pool resource: {e}")
    
    def _start_background_threads(self):
        """Start background monitoring and management threads"""
        # Health monitor thread
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="ResourcePool-HealthMonitor"
        )
        self.health_monitor_thread.start()
        
        # Resource manager thread
        self.resource_manager_thread = threading.Thread(
            target=self._resource_manager_loop,
            daemon=True,
            name="ResourcePool-ResourceManager"
        )
        self.resource_manager_thread.start()
    
    def _create_resource(self) -> str:
        """Create new resource and add to pool"""
        resource_id = str(uuid.uuid4())
        
        try:
            resource = self.resource_factory()
            
            pooled_resource = PooledResource(
                resource_id=resource_id,
                resource=resource,
                status=ResourceStatus.AVAILABLE,
                metrics=ResourceMetrics(),
                created_at=time.time(),
                last_health_check=time.time(),
                health_check_function=self.health_check_function,
                cleanup_function=self.cleanup_function
            )
            
            self.resources[resource_id] = pooled_resource
            self.available_queue.put((0, resource_id))  # Priority 0 for new resources
            
            self.pool_metrics['total_resources_created'] += 1
            self.pool_metrics['current_pool_size'] += 1
            self.pool_metrics['peak_pool_size'] = max(
                self.pool_metrics['peak_pool_size'],
                self.pool_metrics['current_pool_size']
            )
            
            logger.debug(f"Created resource {resource_id}")
            return resource_id
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            raise
    
    def _destroy_resource(self, resource_id: str):
        """Destroy resource and remove from pool"""
        if resource_id not in self.resources:
            return
        
        pooled_resource = self.resources[resource_id]
        
        try:
            # Cleanup resource if cleanup function provided
            if pooled_resource.cleanup_function:
                pooled_resource.cleanup_function(pooled_resource.resource)
        except Exception as e:
            logger.error(f"Error cleaning up resource {resource_id}: {e}")
        
        # Remove from pool
        del self.resources[resource_id]
        self.in_use.discard(resource_id)
        
        self.pool_metrics['total_resources_destroyed'] += 1
        self.pool_metrics['current_pool_size'] -= 1
        
        logger.debug(f"Destroyed resource {resource_id}")
    
    def acquire(self, timeout: Optional[float] = None, tags: Optional[Set[str]] = None) -> Optional[T]:
        """
        Acquire resource from pool
        
        Args:
            timeout: Maximum wait time in seconds
            tags: Required resource tags
            
        Returns:
            Resource if available, None if timeout
        """
        start_time = time.time()
        
        with self.condition:
            while True:
                # Try to get available resource
                resource_id = self._get_best_available_resource(tags)
                
                if resource_id:
                    # Mark as in use
                    pooled_resource = self.resources[resource_id]
                    pooled_resource.status = ResourceStatus.IN_USE
                    pooled_resource.metrics.last_used = time.time()
                    pooled_resource.metrics.total_requests += 1
                    self.in_use.add(resource_id)
                    
                    # Update metrics
                    acquisition_time = time.time() - start_time
                    self.pool_metrics['total_acquisitions'] += 1
                    self._update_average_acquisition_time(acquisition_time)
                    
                    logger.debug(f"Acquired resource {resource_id} in {acquisition_time:.3f}s")
                    return pooled_resource.resource
                
                # Try to create new resource if pool can grow
                if self._can_create_resource():
                    try:
                        resource_id = self._create_resource()
                        continue  # Try again with new resource
                    except Exception as e:
                        logger.error(f"Failed to create resource on demand: {e}")
                
                # Wait for resource to become available
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    
                    if remaining <= 0:
                        self.pool_metrics['failed_acquisitions'] += 1
                        logger.warning("Resource acquisition timed out")
                        return None
                    
                    self.condition.wait(remaining)
                else:
                    self.condition.wait()
    
    def release(self, resource: T):
        """
        Release resource back to pool
        
        Args:
            resource: Resource to release
        """
        with self.pool_lock:
            # Find resource by object reference
            resource_id = None
            for rid, pooled_resource in self.resources.items():
                if pooled_resource.resource is resource:
                    resource_id = rid
                    break
            
            if resource_id is None:
                logger.error("Attempted to release unknown resource")
                return
            
            pooled_resource = self.resources[resource_id]
            
            # Update metrics
            usage_time = time.time() - pooled_resource.metrics.last_used
            pooled_resource.metrics.total_usage_time += usage_time
            pooled_resource.metrics.successful_requests += 1
            
            # Mark as available
            pooled_resource.status = ResourceStatus.AVAILABLE
            self.in_use.discard(resource_id)
            
            # Add back to available queue with updated priority
            priority = self.load_balancer.calculate_priority(pooled_resource)
            self.available_queue.put((-priority, resource_id))  # Negative for max-heap behavior
            
            # Notify waiting threads
            with self.condition:
                self.condition.notify()
            
            logger.debug(f"Released resource {resource_id}")
    
    def release_failed(self, resource: T, error: Exception):
        """
        Release resource that failed during use
        
        Args:
            resource: Failed resource
            error: Error that occurred
        """
        with self.pool_lock:
            # Find resource by object reference
            resource_id = None
            for rid, pooled_resource in self.resources.items():
                if pooled_resource.resource is resource:
                    resource_id = rid
                    break
            
            if resource_id is None:
                logger.error("Attempted to release unknown failed resource")
                return
            
            pooled_resource = self.resources[resource_id]
            
            # Update error metrics
            pooled_resource.metrics.failed_requests += 1
            pooled_resource.metrics.error_rate = (
                pooled_resource.metrics.failed_requests / 
                max(1, pooled_resource.metrics.total_requests)
            )
            
            # Mark as failed and remove from in_use
            pooled_resource.status = ResourceStatus.FAILED
            self.in_use.discard(resource_id)
            
            logger.warning(f"Resource {resource_id} failed: {error}")
            
            # Schedule for recovery or replacement
            self._schedule_resource_recovery(resource_id)
    
    def _get_best_available_resource(self, required_tags: Optional[Set[str]] = None) -> Optional[str]:
        """Get best available resource based on strategy and tags"""
        candidates = []
        
        # Collect available resources
        while not self.available_queue.empty():
            try:
                priority, resource_id = self.available_queue.get_nowait()
                
                if resource_id not in self.resources:
                    continue  # Resource was destroyed
                
                pooled_resource = self.resources[resource_id]
                
                if pooled_resource.status != ResourceStatus.AVAILABLE:
                    continue  # Resource is not available
                
                # Check tag requirements
                if required_tags and not required_tags.issubset(pooled_resource.tags):
                    candidates.append((priority, resource_id))  # Put back in queue later
                    continue
                
                # Found suitable resource
                # Put other candidates back in queue
                for p, rid in candidates:
                    self.available_queue.put((p, rid))
                
                return resource_id
                
            except queue.Empty:
                break
        
        # Put all candidates back
        for priority, resource_id in candidates:
            self.available_queue.put((priority, resource_id))
        
        return None
    
    def _can_create_resource(self) -> bool:
        """Check if new resource can be created"""
        if self.strategy == PoolStrategy.FIXED_SIZE:
            return False
        
        current_size = len(self.resources)
        return current_size < self.max_size
    
    def _should_create_resource(self) -> bool:
        """Determine if new resource should be created based on strategy"""
        if not self._can_create_resource():
            return False
        
        current_size = len(self.resources)
        available_count = len([r for r in self.resources.values() 
                              if r.status == ResourceStatus.AVAILABLE])
        
        if self.strategy == PoolStrategy.DYNAMIC:
            # Create if no resources available and demand exists
            return available_count == 0 and len(self.in_use) > 0
        
        elif self.strategy == PoolStrategy.ADAPTIVE:
            # Create based on utilization and demand patterns
            utilization = len(self.in_use) / max(1, current_size)
            return utilization > 0.8 and available_count < 2
        
        elif self.strategy == PoolStrategy.LOAD_BALANCED:
            # Create based on load balancer recommendations
            return self.load_balancer.should_scale_up(self.resources, self.in_use)
        
        return False
    
    def _schedule_resource_recovery(self, resource_id: str):
        """Schedule resource for recovery attempt"""
        # For now, just destroy failed resources
        # In a more sophisticated implementation, we might attempt recovery
        threading.Thread(
            target=self._destroy_resource,
            args=(resource_id,),
            daemon=True
        ).start()
    
    def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                with self.pool_lock:
                    for resource_id, pooled_resource in list(self.resources.items()):
                        # Skip resources currently in use
                        if resource_id in self.in_use:
                            continue
                        
                        # Check if health check is due
                        if (current_time - pooled_resource.last_health_check >= 
                            self.health_check_interval):
                            
                            self._perform_health_check(resource_id)
                
                # Sleep until next check
                self.shutdown_event.wait(min(30, self.health_check_interval))
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self.shutdown_event.wait(10)  # Wait before retrying
    
    def _perform_health_check(self, resource_id: str):
        """Perform health check on specific resource"""
        if resource_id not in self.resources:
            return
        
        pooled_resource = self.resources[resource_id]
        pooled_resource.last_health_check = time.time()
        
        # Skip if no health check function
        if not pooled_resource.health_check_function:
            return
        
        try:
            is_healthy = pooled_resource.health_check_function(pooled_resource.resource)
            
            if not is_healthy:
                logger.warning(f"Resource {resource_id} failed health check")
                pooled_resource.status = ResourceStatus.FAILED
                self._schedule_resource_recovery(resource_id)
            elif pooled_resource.status == ResourceStatus.FAILED:
                # Resource recovered
                pooled_resource.status = ResourceStatus.AVAILABLE
                self.available_queue.put((0, resource_id))
                self.pool_metrics['resources_recovered'] += 1
                logger.info(f"Resource {resource_id} recovered")
                
        except Exception as e:
            logger.error(f"Health check failed for resource {resource_id}: {e}")
            pooled_resource.status = ResourceStatus.FAILED
            self._schedule_resource_recovery(resource_id)
    
    def _resource_manager_loop(self):
        """Background resource management loop"""
        while not self.shutdown_event.is_set():
            try:
                self._manage_pool_size()
                self._cleanup_idle_resources()
                self.shutdown_event.wait(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Resource manager error: {e}")
                self.shutdown_event.wait(30)
    
    def _manage_pool_size(self):
        """Manage pool size based on strategy and demand"""
        with self.pool_lock:
            # Create resources if needed
            if self._should_create_resource():
                try:
                    self._create_resource()
                    logger.info("Created resource due to demand")
                except Exception as e:
                    logger.error(f"Failed to create resource: {e}")
            
            # Ensure minimum pool size
            current_size = len([r for r in self.resources.values() 
                              if r.status != ResourceStatus.FAILED])
            
            while current_size < self.min_size:
                try:
                    self._create_resource()
                    current_size += 1
                except Exception as e:
                    logger.error(f"Failed to maintain minimum pool size: {e}")
                    break
    
    def _cleanup_idle_resources(self):
        """Cleanup idle resources that exceed idle timeout"""
        if self.idle_timeout <= 0:
            return
        
        current_time = time.time()
        
        with self.pool_lock:
            idle_resources = []
            
            for resource_id, pooled_resource in self.resources.items():
                if (resource_id not in self.in_use and
                    pooled_resource.status == ResourceStatus.AVAILABLE and
                    current_time - pooled_resource.metrics.last_used > self.idle_timeout):
                    idle_resources.append(resource_id)
            
            # Don't cleanup below minimum size
            current_size = len(self.resources)
            max_cleanup = max(0, current_size - self.min_size)
            
            for resource_id in idle_resources[:max_cleanup]:
                logger.info(f"Cleaning up idle resource {resource_id}")
                self._destroy_resource(resource_id)
    
    def _update_average_acquisition_time(self, acquisition_time: float):
        """Update average acquisition time with exponential moving average"""
        alpha = 0.1  # Smoothing factor
        current_avg = self.pool_metrics['average_acquisition_time']
        
        if current_avg == 0:
            self.pool_metrics['average_acquisition_time'] = acquisition_time
        else:
            self.pool_metrics['average_acquisition_time'] = (
                alpha * acquisition_time + (1 - alpha) * current_avg
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self.pool_lock:
            available_count = len([r for r in self.resources.values() 
                                 if r.status == ResourceStatus.AVAILABLE])
            in_use_count = len(self.in_use)
            failed_count = len([r for r in self.resources.values() 
                              if r.status == ResourceStatus.FAILED])
            
            total_requests = sum(r.metrics.total_requests for r in self.resources.values())
            total_failures = sum(r.metrics.failed_requests for r in self.resources.values())
            
            return {
                **self.pool_metrics,
                'current_available': available_count,
                'current_in_use': in_use_count,
                'current_failed': failed_count,
                'pool_utilization': in_use_count / max(1, len(self.resources)),
                'total_requests_served': total_requests,
                'overall_error_rate': total_failures / max(1, total_requests),
                'strategy': self.strategy.value
            }
    
    def shutdown(self, timeout: float = 30.0):
        """Shutdown resource pool"""
        logger.info("Shutting down resource pool...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for background threads
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=timeout/2)
        
        if self.resource_manager_thread:
            self.resource_manager_thread.join(timeout=timeout/2)
        
        # Cleanup all resources
        with self.pool_lock:
            for resource_id in list(self.resources.keys()):
                self._destroy_resource(resource_id)
        
        logger.info("Resource pool shutdown complete")

class ResourceLoadBalancer:
    """Load balancer for resource selection and scaling decisions"""
    
    def calculate_priority(self, pooled_resource: PooledResource) -> float:
        """Calculate resource priority for selection"""
        metrics = pooled_resource.metrics
        
        # Factors: low error rate, recent success, fast response
        error_weight = 1.0 - min(metrics.error_rate, 0.5)  # Cap at 50% penalty
        success_weight = metrics.successful_requests / max(1, metrics.total_requests)
        recency_weight = max(0.1, 1.0 / (time.time() - metrics.last_used + 1))
        
        # Combined priority score
        priority = (error_weight * 0.4 + success_weight * 0.4 + recency_weight * 0.2) * 100
        
        return priority
    
    def should_scale_up(self, resources: Dict[str, PooledResource], in_use: Set[str]) -> bool:
        """Determine if pool should scale up"""
        if not resources:
            return True
        
        # Calculate current load
        total_resources = len(resources)
        active_resources = len(in_use)
        utilization = active_resources / total_resources
        
        # Check average error rate
        total_requests = sum(r.metrics.total_requests for r in resources.values())
        total_errors = sum(r.metrics.failed_requests for r in resources.values())
        error_rate = total_errors / max(1, total_requests)
        
        # Scale up if high utilization and low error rate
        return utilization > 0.7 and error_rate < 0.1

# Context manager for resource acquisition
class ResourceContext(Generic[T]):
    """Context manager for automatic resource acquisition and release"""
    
    def __init__(self, pool: ResourcePool[T], timeout: Optional[float] = None, 
                 tags: Optional[Set[str]] = None):
        self.pool = pool
        self.timeout = timeout
        self.tags = tags
        self.resource: Optional[T] = None
        self.acquired = False
    
    def __enter__(self) -> Optional[T]:
        self.resource = self.pool.acquire(timeout=self.timeout, tags=self.tags)
        self.acquired = self.resource is not None
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired and self.resource:
            if exc_type is None:
                self.pool.release(self.resource)
            else:
                self.pool.release_failed(self.resource, exc_val)
        return False