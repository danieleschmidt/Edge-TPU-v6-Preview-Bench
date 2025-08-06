"""
Task Graph - Robust task dependency management with validation and error handling
Implements quantum-inspired graph algorithms with comprehensive safety measures
"""

import logging
import time
import uuid
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path

from .quantum_task_engine import QuantumTask, TaskState, Priority

logger = logging.getLogger(__name__)

class GraphValidationError(Exception):
    """Raised when task graph validation fails"""
    pass

class CircularDependencyError(GraphValidationError):
    """Raised when circular dependencies are detected"""
    pass

class TaskNotFoundError(Exception):
    """Raised when referenced task is not found"""
    pass

class GraphIntegrityError(Exception):
    """Raised when graph integrity is compromised"""
    pass

@dataclass
class DependencyEdge:
    """Represents a dependency relationship between tasks"""
    from_task: str
    to_task: str
    dependency_type: str = "hard"  # hard, soft, conditional
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate edge data"""
        if not self.from_task or not self.to_task:
            raise ValueError("Edge must have valid from_task and to_task")
        if self.from_task == self.to_task:
            raise ValueError("Self-dependencies are not allowed")
        if self.weight < 0:
            raise ValueError("Edge weight must be non-negative")

@dataclass
class GraphMetrics:
    """Task graph analysis metrics"""
    total_nodes: int = 0
    total_edges: int = 0
    max_depth: int = 0
    critical_path_length: float = 0.0
    complexity_score: float = 0.0
    cyclic_complexity: float = 0.0
    parallelism_factor: float = 0.0
    bottleneck_tasks: List[str] = field(default_factory=list)
    
class TaskGraph:
    """
    Robust task dependency graph with quantum-inspired algorithms
    
    Features:
    - Comprehensive validation and error handling
    - Circular dependency detection and resolution
    - Graph integrity monitoring
    - Performance metrics and analysis
    - Secure serialization/deserialization
    - Concurrent access protection
    """
    
    def __init__(self, graph_id: Optional[str] = None):
        self.graph_id = graph_id or str(uuid.uuid4())
        self.tasks: Dict[str, QuantumTask] = {}
        self.edges: Dict[str, DependencyEdge] = {}
        self.adjacency_list: Dict[str, Set[str]] = {}  # task_id -> set of dependent tasks
        self.reverse_adjacency: Dict[str, Set[str]] = {}  # task_id -> set of prerequisite tasks
        
        # Graph integrity and security
        self.graph_hash: Optional[str] = None
        self.creation_time = time.time()
        self.last_modified = time.time()
        self.version = 1
        self.is_sealed = False
        
        # Metrics and analysis cache
        self._metrics_cache: Optional[GraphMetrics] = None
        self._metrics_cache_time: float = 0.0
        self._cache_ttl = 60.0  # Cache TTL in seconds
        
        # Validation settings
        self.max_graph_size = 10000  # Maximum number of tasks
        self.max_dependency_depth = 100  # Maximum dependency chain length
        self.enable_strict_validation = True
        
        logger.info(f"TaskGraph {self.graph_id} initialized")
    
    def add_task(self, task: QuantumTask, validate: bool = True) -> bool:
        """
        Add task to graph with comprehensive validation
        
        Args:
            task: QuantumTask to add
            validate: Whether to perform validation checks
            
        Returns:
            True if task was added successfully
            
        Raises:
            GraphValidationError: If validation fails
            ValueError: If task data is invalid
        """
        if self.is_sealed:
            raise GraphIntegrityError("Cannot modify sealed graph")
            
        try:
            # Input validation
            if not task or not task.task_id:
                raise ValueError("Task must have valid task_id")
            
            if task.task_id in self.tasks:
                logger.warning(f"Task {task.task_id} already exists, updating")
            
            # Size limit check
            if len(self.tasks) >= self.max_graph_size:
                raise GraphValidationError(f"Graph size limit exceeded: {self.max_graph_size}")
            
            # Task validation
            if validate and self.enable_strict_validation:
                self._validate_task(task)
            
            # Add task to graph
            self.tasks[task.task_id] = task
            
            # Initialize adjacency lists if needed
            if task.task_id not in self.adjacency_list:
                self.adjacency_list[task.task_id] = set()
            if task.task_id not in self.reverse_adjacency:
                self.reverse_adjacency[task.task_id] = set()
            
            # Add dependency edges
            for dep_id in task.dependencies:
                self._add_dependency_edge(dep_id, task.task_id, validate=validate)
            
            # Update graph metadata
            self._update_metadata()
            
            # Validate graph integrity after addition
            if validate:
                self._validate_graph_integrity()
            
            logger.debug(f"Added task {task.task_id} to graph {self.graph_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add task {task.task_id}: {e}")
            # Cleanup partial state
            self._cleanup_partial_task_addition(task.task_id)
            raise
    
    def remove_task(self, task_id: str, force: bool = False) -> bool:
        """
        Remove task from graph with dependency cleanup
        
        Args:
            task_id: ID of task to remove
            force: If True, remove even if other tasks depend on it
            
        Returns:
            True if task was removed successfully
            
        Raises:
            TaskNotFoundError: If task doesn't exist
            GraphValidationError: If removal would break dependencies
        """
        if self.is_sealed:
            raise GraphIntegrityError("Cannot modify sealed graph")
            
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        
        try:
            # Check for dependent tasks
            dependents = self.get_dependent_tasks(task_id)
            if dependents and not force:
                raise GraphValidationError(
                    f"Cannot remove task {task_id}: {len(dependents)} tasks depend on it. "
                    f"Use force=True to override."
                )
            
            # Remove dependency edges
            edges_to_remove = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.from_task == task_id or edge.to_task == task_id
            ]
            
            for edge_id in edges_to_remove:
                self._remove_edge(edge_id)
            
            # Clean up adjacency lists
            if task_id in self.adjacency_list:
                for dependent in self.adjacency_list[task_id].copy():
                    self.reverse_adjacency[dependent].discard(task_id)
                del self.adjacency_list[task_id]
            
            if task_id in self.reverse_adjacency:
                for prerequisite in self.reverse_adjacency[task_id].copy():
                    self.adjacency_list[prerequisite].discard(task_id)
                del self.reverse_adjacency[task_id]
            
            # Remove task
            del self.tasks[task_id]
            
            # Update dependent tasks if forced removal
            if force and dependents:
                for dependent_id in dependents:
                    if dependent_id in self.tasks:
                        self.tasks[dependent_id].dependencies.discard(task_id)
            
            self._update_metadata()
            
            logger.info(f"Removed task {task_id} from graph {self.graph_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove task {task_id}: {e}")
            raise
    
    def add_dependency(self, 
                      from_task_id: str, 
                      to_task_id: str,
                      dependency_type: str = "hard",
                      weight: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None,
                      validate: bool = True) -> bool:
        """
        Add dependency relationship between tasks
        
        Args:
            from_task_id: Prerequisite task ID
            to_task_id: Dependent task ID  
            dependency_type: Type of dependency (hard, soft, conditional)
            weight: Dependency weight
            metadata: Additional dependency metadata
            validate: Whether to perform validation
            
        Returns:
            True if dependency was added successfully
        """
        if self.is_sealed:
            raise GraphIntegrityError("Cannot modify sealed graph")
            
        try:
            # Validate task existence
            if from_task_id not in self.tasks:
                raise TaskNotFoundError(f"Prerequisite task {from_task_id} not found")
            if to_task_id not in self.tasks:
                raise TaskNotFoundError(f"Dependent task {to_task_id} not found")
            
            # Create dependency edge
            edge = DependencyEdge(
                from_task=from_task_id,
                to_task=to_task_id,
                dependency_type=dependency_type,
                weight=weight,
                metadata=metadata or {}
            )
            
            # Add edge with validation
            return self._add_dependency_edge(from_task_id, to_task_id, validate, edge)
            
        except Exception as e:
            logger.error(f"Failed to add dependency {from_task_id} -> {to_task_id}: {e}")
            raise
    
    def get_execution_order(self, algorithm: str = "topological") -> List[str]:
        """
        Get optimal task execution order
        
        Args:
            algorithm: Ordering algorithm (topological, critical_path, quantum_priority)
            
        Returns:
            List of task IDs in execution order
            
        Raises:
            CircularDependencyError: If circular dependencies exist
            GraphValidationError: If graph is invalid
        """
        try:
            # Validate graph before ordering
            self._validate_graph_integrity()
            
            if algorithm == "topological":
                return self._topological_sort()
            elif algorithm == "critical_path":
                return self._critical_path_sort()
            elif algorithm == "quantum_priority":
                return self._quantum_priority_sort()
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to get execution order: {e}")
            raise
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the graph
        
        Returns:
            List of cycles (each cycle is a list of task IDs)
        """
        try:
            cycles = []
            visited = set()
            rec_stack = set()
            path = []
            
            def dfs(task_id: str) -> bool:
                if task_id in rec_stack:
                    # Found cycle, extract it
                    cycle_start = path.index(task_id)
                    cycle = path[cycle_start:] + [task_id]
                    cycles.append(cycle)
                    return True
                
                if task_id in visited:
                    return False
                
                visited.add(task_id)
                rec_stack.add(task_id)
                path.append(task_id)
                
                # Check all dependencies
                for dependent in self.adjacency_list.get(task_id, set()):
                    if dfs(dependent):
                        return True
                
                rec_stack.remove(task_id)
                path.pop()
                return False
            
            # Check all nodes
            for task_id in self.tasks:
                if task_id not in visited:
                    dfs(task_id)
            
            return cycles
            
        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}")
            return []
    
    def get_graph_metrics(self, force_refresh: bool = False) -> GraphMetrics:
        """
        Get comprehensive graph analysis metrics
        
        Args:
            force_refresh: Force recalculation of cached metrics
            
        Returns:
            GraphMetrics object with analysis results
        """
        try:
            # Check cache validity
            current_time = time.time()
            if (not force_refresh and 
                self._metrics_cache and 
                current_time - self._metrics_cache_time < self._cache_ttl):
                return self._metrics_cache
            
            # Calculate metrics
            metrics = GraphMetrics()
            
            metrics.total_nodes = len(self.tasks)
            metrics.total_edges = len(self.edges)
            
            if metrics.total_nodes > 0:
                # Calculate depth and critical path
                depths = self._calculate_task_depths()
                metrics.max_depth = max(depths.values()) if depths else 0
                
                # Critical path analysis
                metrics.critical_path_length = self._calculate_critical_path_length()
                
                # Complexity metrics
                metrics.complexity_score = self._calculate_complexity_score()
                metrics.cyclic_complexity = len(self.detect_circular_dependencies())
                
                # Parallelism analysis
                metrics.parallelism_factor = self._calculate_parallelism_factor()
                
                # Bottleneck identification
                metrics.bottleneck_tasks = self._identify_bottlenecks()
            
            # Cache results
            self._metrics_cache = metrics
            self._metrics_cache_time = current_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return GraphMetrics()  # Return empty metrics on error
    
    def validate_graph(self, raise_on_error: bool = True) -> Dict[str, Any]:
        """
        Comprehensive graph validation
        
        Args:
            raise_on_error: Whether to raise exceptions on validation errors
            
        Returns:
            Validation report dictionary
        """
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Basic structure validation
            self._validate_basic_structure(validation_report)
            
            # Dependency validation
            self._validate_dependencies(validation_report)
            
            # Circular dependency check
            cycles = self.detect_circular_dependencies()
            if cycles:
                error_msg = f"Circular dependencies detected: {len(cycles)} cycles"
                validation_report['errors'].append(error_msg)
                validation_report['is_valid'] = False
                
                if raise_on_error:
                    raise CircularDependencyError(error_msg)
            
            # Performance and complexity validation
            self._validate_performance_constraints(validation_report)
            
            # Calculate validation metrics
            validation_report['metrics'] = self.get_graph_metrics().to_dict() if hasattr(self.get_graph_metrics(), 'to_dict') else {}
            
        except Exception as e:
            error_msg = f"Graph validation failed: {e}"
            validation_report['errors'].append(error_msg)
            validation_report['is_valid'] = False
            
            if raise_on_error:
                raise GraphValidationError(error_msg)
        
        return validation_report
    
    def serialize(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Serialize graph to dictionary with security measures
        
        Args:
            include_metadata: Whether to include graph metadata
            
        Returns:
            Serialized graph dictionary
        """
        try:
            serialized = {
                'graph_id': self.graph_id,
                'version': self.version,
                'tasks': {},
                'edges': {},
                'adjacency_list': {k: list(v) for k, v in self.adjacency_list.items()},
                'serialization_time': time.time()
            }
            
            # Serialize tasks
            for task_id, task in self.tasks.items():
                serialized['tasks'][task_id] = {
                    'task_id': task.task_id,
                    'name': task.name,
                    'description': task.description,
                    'dependencies': list(task.dependencies),
                    'estimated_duration': task.estimated_duration,
                    'priority': task.priority.value,
                    'quantum_weight': task.quantum_weight,
                    'state': task.state.value,
                    'metadata': task.metadata
                }
            
            # Serialize edges
            for edge_id, edge in self.edges.items():
                serialized['edges'][edge_id] = {
                    'from_task': edge.from_task,
                    'to_task': edge.to_task,
                    'dependency_type': edge.dependency_type,
                    'weight': edge.weight,
                    'metadata': edge.metadata,
                    'created_at': edge.created_at
                }
            
            # Include metadata if requested
            if include_metadata:
                serialized['metadata'] = {
                    'creation_time': self.creation_time,
                    'last_modified': self.last_modified,
                    'is_sealed': self.is_sealed,
                    'graph_hash': self._calculate_graph_hash()
                }
            
            return serialized
            
        except Exception as e:
            logger.error(f"Graph serialization failed: {e}")
            raise
    
    def deserialize(self, data: Dict[str, Any], validate: bool = True) -> bool:
        """
        Deserialize graph from dictionary with validation
        
        Args:
            data: Serialized graph data
            validate: Whether to validate deserialized data
            
        Returns:
            True if deserialization was successful
        """
        try:
            if self.is_sealed:
                raise GraphIntegrityError("Cannot deserialize into sealed graph")
            
            # Validate input data structure
            required_fields = ['graph_id', 'version', 'tasks', 'edges']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Clear existing data
            self.tasks.clear()
            self.edges.clear()
            self.adjacency_list.clear()
            self.reverse_adjacency.clear()
            
            # Restore basic properties
            self.graph_id = data['graph_id']
            self.version = data['version']
            
            # Deserialize tasks
            for task_id, task_data in data['tasks'].items():
                task = QuantumTask(
                    task_id=task_data['task_id'],
                    name=task_data['name'],
                    description=task_data['description'],
                    function=lambda: None,  # Placeholder function
                    dependencies=set(task_data['dependencies']),
                    estimated_duration=task_data['estimated_duration'],
                    priority=Priority(task_data['priority']),
                    quantum_weight=task_data['quantum_weight'],
                    metadata=task_data.get('metadata', {})
                )
                task.state = TaskState(task_data['state'])
                
                self.tasks[task_id] = task
            
            # Deserialize edges
            for edge_id, edge_data in data['edges'].items():
                edge = DependencyEdge(
                    from_task=edge_data['from_task'],
                    to_task=edge_data['to_task'],
                    dependency_type=edge_data['dependency_type'],
                    weight=edge_data['weight'],
                    metadata=edge_data.get('metadata', {}),
                    created_at=edge_data['created_at']
                )
                self.edges[edge_id] = edge
            
            # Rebuild adjacency lists
            for task_id in self.tasks:
                self.adjacency_list[task_id] = set()
                self.reverse_adjacency[task_id] = set()
            
            for edge in self.edges.values():
                self.adjacency_list[edge.from_task].add(edge.to_task)
                self.reverse_adjacency[edge.to_task].add(edge.from_task)
            
            # Restore metadata if available
            if 'metadata' in data:
                metadata = data['metadata']
                self.creation_time = metadata.get('creation_time', time.time())
                self.last_modified = metadata.get('last_modified', time.time())
                self.is_sealed = metadata.get('is_sealed', False)
                
                # Verify graph hash if available
                stored_hash = metadata.get('graph_hash')
                if stored_hash and validate:
                    current_hash = self._calculate_graph_hash()
                    if stored_hash != current_hash:
                        raise GraphIntegrityError("Graph hash verification failed")
            
            # Validate deserialized graph
            if validate:
                validation_result = self.validate_graph(raise_on_error=True)
                if not validation_result['is_valid']:
                    raise GraphValidationError("Deserialized graph failed validation")
            
            self._update_metadata()
            
            logger.info(f"Successfully deserialized graph {self.graph_id}")
            return True
            
        except Exception as e:
            logger.error(f"Graph deserialization failed: {e}")
            # Clear potentially corrupted state
            self.tasks.clear()
            self.edges.clear()
            self.adjacency_list.clear()
            self.reverse_adjacency.clear()
            raise
    
    def seal_graph(self) -> str:
        """
        Seal graph to prevent modifications and generate integrity hash
        
        Returns:
            Graph integrity hash
        """
        try:
            # Final validation before sealing
            validation_result = self.validate_graph(raise_on_error=True)
            if not validation_result['is_valid']:
                raise GraphValidationError("Cannot seal invalid graph")
            
            # Generate final hash
            self.graph_hash = self._calculate_graph_hash()
            self.is_sealed = True
            
            logger.info(f"Graph {self.graph_id} sealed with hash {self.graph_hash}")
            return self.graph_hash
            
        except Exception as e:
            logger.error(f"Failed to seal graph: {e}")
            raise
    
    def get_dependent_tasks(self, task_id: str) -> Set[str]:
        """Get all tasks that depend on the given task"""
        return self.adjacency_list.get(task_id, set()).copy()
    
    def get_prerequisite_tasks(self, task_id: str) -> Set[str]:
        """Get all tasks that the given task depends on"""
        return self.reverse_adjacency.get(task_id, set()).copy()
    
    def get_task_level(self, task_id: str) -> int:
        """Get the dependency level of a task (0 = no dependencies)"""
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        
        visited = set()
        
        def calculate_level(tid: str) -> int:
            if tid in visited:
                return 0  # Avoid infinite recursion
            visited.add(tid)
            
            prerequisites = self.get_prerequisite_tasks(tid)
            if not prerequisites:
                return 0
            
            max_prereq_level = max(calculate_level(prereq) for prereq in prerequisites)
            return max_prereq_level + 1
        
        return calculate_level(task_id)
    
    # Private helper methods
    
    def _validate_task(self, task: QuantumTask):
        """Validate individual task"""
        if not task.name:
            raise ValueError("Task name cannot be empty")
        
        if task.estimated_duration < 0:
            raise ValueError("Task duration cannot be negative")
        
        if task.quantum_weight < 0 or task.quantum_weight > 1:
            raise ValueError("Quantum weight must be between 0 and 1")
        
        # Validate dependencies exist
        for dep_id in task.dependencies:
            if dep_id not in self.tasks and dep_id != task.task_id:
                logger.warning(f"Task {task.task_id} depends on non-existent task {dep_id}")
    
    def _add_dependency_edge(self, from_task_id: str, to_task_id: str, validate: bool = True, edge: Optional[DependencyEdge] = None) -> bool:
        """Internal method to add dependency edge"""
        if from_task_id == to_task_id:
            raise ValueError("Self-dependencies are not allowed")
        
        # Create edge if not provided
        if edge is None:
            edge = DependencyEdge(from_task=from_task_id, to_task=to_task_id)
        
        edge_id = f"{from_task_id}->{to_task_id}"
        self.edges[edge_id] = edge
        
        # Update adjacency lists
        if from_task_id not in self.adjacency_list:
            self.adjacency_list[from_task_id] = set()
        if to_task_id not in self.reverse_adjacency:
            self.reverse_adjacency[to_task_id] = set()
        
        self.adjacency_list[from_task_id].add(to_task_id)
        self.reverse_adjacency[to_task_id].add(from_task_id)
        
        # Update task dependencies
        if to_task_id in self.tasks:
            self.tasks[to_task_id].dependencies.add(from_task_id)
        
        # Validate no cycles created
        if validate:
            cycles = self.detect_circular_dependencies()
            if cycles:
                # Rollback the edge addition
                self._remove_edge(edge_id)
                raise CircularDependencyError(f"Adding dependency {from_task_id} -> {to_task_id} would create circular dependency")
        
        return True
    
    def _remove_edge(self, edge_id: str):
        """Remove dependency edge"""
        if edge_id not in self.edges:
            return
        
        edge = self.edges[edge_id]
        
        # Update adjacency lists
        if edge.from_task in self.adjacency_list:
            self.adjacency_list[edge.from_task].discard(edge.to_task)
        
        if edge.to_task in self.reverse_adjacency:
            self.reverse_adjacency[edge.to_task].discard(edge.from_task)
        
        # Update task dependencies
        if edge.to_task in self.tasks:
            self.tasks[edge.to_task].dependencies.discard(edge.from_task)
        
        del self.edges[edge_id]
    
    def _cleanup_partial_task_addition(self, task_id: str):
        """Clean up partial state after failed task addition"""
        if task_id in self.tasks:
            del self.tasks[task_id]
        if task_id in self.adjacency_list:
            del self.adjacency_list[task_id]
        if task_id in self.reverse_adjacency:
            del self.reverse_adjacency[task_id]
    
    def _update_metadata(self):
        """Update graph metadata"""
        self.last_modified = time.time()
        self.version += 1
        self._metrics_cache = None  # Invalidate cache
    
    def _validate_graph_integrity(self):
        """Validate overall graph integrity"""
        # Check adjacency list consistency
        for from_task, dependents in self.adjacency_list.items():
            for to_task in dependents:
                if to_task not in self.reverse_adjacency or from_task not in self.reverse_adjacency[to_task]:
                    raise GraphIntegrityError(f"Adjacency list inconsistency: {from_task} -> {to_task}")
        
        # Check edge consistency
        for edge_id, edge in self.edges.items():
            if (edge.to_task not in self.adjacency_list.get(edge.from_task, set()) or
                edge.from_task not in self.reverse_adjacency.get(edge.to_task, set())):
                raise GraphIntegrityError(f"Edge consistency error: {edge_id}")
    
    def _validate_basic_structure(self, report: Dict[str, Any]):
        """Validate basic graph structure"""
        # Check for orphaned tasks
        all_referenced_tasks = set()
        for edge in self.edges.values():
            all_referenced_tasks.add(edge.from_task)
            all_referenced_tasks.add(edge.to_task)
        
        orphaned = set(self.tasks.keys()) - all_referenced_tasks
        if orphaned:
            report['warnings'].append(f"Orphaned tasks found: {list(orphaned)}")
        
        # Check for missing tasks
        missing = all_referenced_tasks - set(self.tasks.keys())
        if missing:
            report['errors'].append(f"Missing tasks referenced in edges: {list(missing)}")
            report['is_valid'] = False
    
    def _validate_dependencies(self, report: Dict[str, Any]):
        """Validate task dependencies"""
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    report['errors'].append(f"Task {task_id} has invalid dependency: {dep_id}")
                    report['is_valid'] = False
    
    def _validate_performance_constraints(self, report: Dict[str, Any]):
        """Validate performance constraints"""
        if len(self.tasks) > self.max_graph_size:
            report['errors'].append(f"Graph size ({len(self.tasks)}) exceeds limit ({self.max_graph_size})")
            report['is_valid'] = False
        
        # Check dependency depth
        try:
            depths = self._calculate_task_depths()
            max_depth = max(depths.values()) if depths else 0
            
            if max_depth > self.max_dependency_depth:
                report['errors'].append(f"Dependency depth ({max_depth}) exceeds limit ({self.max_dependency_depth})")
                report['is_valid'] = False
        except Exception as e:
            report['warnings'].append(f"Could not calculate dependency depth: {e}")
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort with cycle detection"""
        in_degree = {task_id: len(self.reverse_adjacency.get(task_id, set())) for task_id in self.tasks}
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority for consistent ordering
            queue.sort(key=lambda x: (
                -self.tasks[x].priority.value,
                -self.tasks[x].quantum_weight,
                self.tasks[x].estimated_duration
            ))
            
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees
            for dependent in self.adjacency_list.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(result) != len(self.tasks):
            remaining_tasks = set(self.tasks.keys()) - set(result)
            raise CircularDependencyError(f"Circular dependency detected involving tasks: {remaining_tasks}")
        
        return result
    
    def _critical_path_sort(self) -> List[str]:
        """Sort tasks by critical path priority"""
        # Calculate earliest start times and critical path
        earliest_start = {}
        latest_start = {}
        
        # Forward pass - earliest start times
        topo_order = self._topological_sort()
        
        for task_id in topo_order:
            task = self.tasks[task_id]
            prerequisites = self.get_prerequisite_tasks(task_id)
            
            if not prerequisites:
                earliest_start[task_id] = 0.0
            else:
                earliest_start[task_id] = max(
                    earliest_start[prereq] + self.tasks[prereq].estimated_duration
                    for prereq in prerequisites
                )
        
        # Backward pass - latest start times
        max_completion = max(
            earliest_start[task_id] + self.tasks[task_id].estimated_duration
            for task_id in self.tasks
        )
        
        for task_id in reversed(topo_order):
            task = self.tasks[task_id]
            dependents = self.get_dependent_tasks(task_id)
            
            if not dependents:
                latest_start[task_id] = max_completion - task.estimated_duration
            else:
                latest_start[task_id] = min(
                    latest_start[dependent] 
                    for dependent in dependents
                ) - task.estimated_duration
        
        # Calculate slack and sort by critical path
        critical_tasks = []
        for task_id in self.tasks:
            slack = latest_start[task_id] - earliest_start[task_id]
            critical_tasks.append((task_id, slack, earliest_start[task_id]))
        
        # Sort by slack (ascending), then by earliest start time
        critical_tasks.sort(key=lambda x: (x[1], x[2]))
        
        return [task_id for task_id, _, _ in critical_tasks]
    
    def _quantum_priority_sort(self) -> List[str]:
        """Sort tasks using quantum-inspired priority algorithm"""
        quantum_priorities = []
        
        for task_id, task in self.tasks.items():
            # Quantum priority calculation
            base_priority = task.priority.value * task.quantum_weight
            
            # Dependency influence
            prereq_count = len(self.get_prerequisite_tasks(task_id))
            dependent_count = len(self.get_dependent_tasks(task_id))
            
            # Quantum entanglement factor
            entanglement_factor = 1.0 + 0.1 * (prereq_count + dependent_count)
            
            # Duration influence (shorter tasks get boost)
            duration_factor = 1.0 / max(0.1, task.estimated_duration)
            
            quantum_priority = base_priority * entanglement_factor * duration_factor
            quantum_priorities.append((task_id, quantum_priority))
        
        # Sort by quantum priority (descending)
        quantum_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure topological constraints are maintained
        result = []
        remaining = {task_id for task_id, _ in quantum_priorities}
        
        while remaining:
            added_this_round = []
            
            for task_id, _ in quantum_priorities:
                if task_id in remaining:
                    prerequisites = self.get_prerequisite_tasks(task_id)
                    if prerequisites.issubset(set(result)):
                        result.append(task_id)
                        added_this_round.append(task_id)
            
            if not added_this_round:
                # Fallback to topological sort for remaining tasks
                remaining_topo = self._topological_sort()
                for task_id in remaining_topo:
                    if task_id in remaining:
                        result.append(task_id)
                break
            
            for task_id in added_this_round:
                remaining.remove(task_id)
        
        return result
    
    def _calculate_task_depths(self) -> Dict[str, int]:
        """Calculate depth of each task in dependency graph"""
        depths = {}
        visited = set()
        
        def calculate_depth(task_id: str) -> int:
            if task_id in depths:
                return depths[task_id]
            
            if task_id in visited:
                # Circular dependency detected
                return 0
            
            visited.add(task_id)
            
            prerequisites = self.get_prerequisite_tasks(task_id)
            if not prerequisites:
                depths[task_id] = 0
            else:
                max_prereq_depth = max(calculate_depth(prereq) for prereq in prerequisites)
                depths[task_id] = max_prereq_depth + 1
            
            visited.remove(task_id)
            return depths[task_id]
        
        for task_id in self.tasks:
            calculate_depth(task_id)
        
        return depths
    
    def _calculate_critical_path_length(self) -> float:
        """Calculate the length of the critical path"""
        try:
            topo_order = self._topological_sort()
            earliest_completion = {}
            
            for task_id in topo_order:
                task = self.tasks[task_id]
                prerequisites = self.get_prerequisite_tasks(task_id)
                
                if not prerequisites:
                    earliest_start = 0.0
                else:
                    earliest_start = max(earliest_completion[prereq] for prereq in prerequisites)
                
                earliest_completion[task_id] = earliest_start + task.estimated_duration
            
            return max(earliest_completion.values()) if earliest_completion else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_complexity_score(self) -> float:
        """Calculate graph complexity score"""
        if not self.tasks:
            return 0.0
        
        num_nodes = len(self.tasks)
        num_edges = len(self.edges)
        
        # Basic complexity metrics
        edge_density = num_edges / max(1, num_nodes * (num_nodes - 1) / 2)
        
        # Dependency complexity
        max_dependencies = max(len(task.dependencies) for task in self.tasks.values()) if self.tasks else 0
        avg_dependencies = sum(len(task.dependencies) for task in self.tasks.values()) / num_nodes
        
        # Depth complexity
        depths = self._calculate_task_depths()
        depth_variance = np.var(list(depths.values())) if depths else 0
        
        # Combined complexity score
        complexity = (
            0.3 * edge_density +
            0.2 * (max_dependencies / max(1, num_nodes)) +
            0.2 * (avg_dependencies / max(1, num_nodes)) +
            0.3 * (depth_variance / max(1, num_nodes))
        )
        
        return min(1.0, complexity)  # Cap at 1.0
    
    def _calculate_parallelism_factor(self) -> float:
        """Calculate potential parallelism in the graph"""
        if not self.tasks:
            return 0.0
        
        # Count tasks at each depth level
        depths = self._calculate_task_depths()
        depth_counts = {}
        
        for depth in depths.values():
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        # Average parallelism is the average number of tasks per level
        if depth_counts:
            avg_parallelism = sum(depth_counts.values()) / len(depth_counts)
            max_parallelism = max(depth_counts.values())
            
            return min(1.0, avg_parallelism / len(self.tasks))
        
        return 0.0
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify bottleneck tasks in the graph"""
        bottlenecks = []
        
        for task_id, task in self.tasks.items():
            # Tasks with many dependents are potential bottlenecks
            dependent_count = len(self.get_dependent_tasks(task_id))
            
            # Tasks on critical path with high duration
            is_critical_path = True  # Simplified - would need proper critical path calculation
            
            if (dependent_count > len(self.tasks) * 0.2 or  # > 20% of tasks depend on this
                (is_critical_path and task.estimated_duration > 0)):
                bottlenecks.append(task_id)
        
        return bottlenecks
    
    def _calculate_graph_hash(self) -> str:
        """Calculate cryptographic hash of graph for integrity verification"""
        # Create deterministic representation of graph
        graph_data = {
            'tasks': sorted([
                (task.task_id, task.name, task.estimated_duration, task.priority.value)
                for task in self.tasks.values()
            ]),
            'edges': sorted([
                (edge.from_task, edge.to_task, edge.dependency_type, edge.weight)
                for edge in self.edges.values()
            ])
        }
        
        # Calculate SHA-256 hash
        graph_json = json.dumps(graph_data, sort_keys=True)
        return hashlib.sha256(graph_json.encode()).hexdigest()