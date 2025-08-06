"""
Quantum Task Engine - Core orchestration engine using quantum-inspired algorithms
Implements superposition-based task exploration and entanglement-based dependencies
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)

class TaskState(Enum):
    SUPERPOSITION = "superposition"  # Task exists in multiple possible states
    COLLAPSED = "collapsed"          # Task state determined
    ENTANGLED = "entangled"         # Task depends on other tasks
    EXECUTED = "executed"           # Task completed
    FAILED = "failed"               # Task failed

class Priority(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    DEFERRED = 1

@dataclass
class QuantumTask:
    """
    Quantum-inspired task representation with superposition capabilities
    """
    task_id: str
    name: str
    description: str
    function: Callable
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 0.0
    priority: Priority = Priority.MEDIUM
    quantum_weight: float = 1.0  # Quantum probability amplitude
    state: TaskState = TaskState.SUPERPOSITION
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def collapse_state(self, new_state: TaskState):
        """Collapse quantum superposition to definite state"""
        logger.debug(f"Task {self.task_id} state collapsed: {self.state} -> {new_state}")
        self.state = new_state

class QuantumTaskEngine:
    """
    Quantum-inspired task orchestration engine
    
    Features:
    - Superposition-based task exploration
    - Entanglement-based dependency resolution
    - Quantum tunneling for deadline optimization
    - Interference pattern analysis for resource allocation
    """
    
    def __init__(self, max_workers: int = 4, quantum_coherence_time: float = 30.0):
        self.tasks: Dict[str, QuantumTask] = {}
        self.task_graph: Dict[str, Set[str]] = {}  # Dependency graph
        self.quantum_state_matrix: Optional[np.ndarray] = None
        self.max_workers = max_workers
        self.quantum_coherence_time = quantum_coherence_time
        self.execution_history: List[Dict[str, Any]] = []
        self.active_entanglements: Dict[str, Set[str]] = {}
        
        # Performance metrics
        self.total_tasks_executed = 0
        self.total_execution_time = 0.0
        self.quantum_efficiency_score = 0.0
        
        logger.info(f"QuantumTaskEngine initialized with {max_workers} workers, "
                   f"coherence_time={quantum_coherence_time}s")
    
    def add_task(self, 
                 task_id: str,
                 name: str, 
                 function: Callable,
                 description: str = "",
                 dependencies: Optional[Set[str]] = None,
                 priority: Priority = Priority.MEDIUM,
                 estimated_duration: float = 0.0,
                 quantum_weight: float = 1.0,
                 **metadata) -> QuantumTask:
        """
        Add a quantum task to the execution pipeline
        
        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            function: Callable to execute
            description: Task description
            dependencies: Set of task IDs this task depends on
            priority: Task priority level
            estimated_duration: Estimated execution time in seconds
            quantum_weight: Quantum probability amplitude (0.0 to 1.0)
            **metadata: Additional task metadata
            
        Returns:
            Created QuantumTask instance
        """
        if dependencies is None:
            dependencies = set()
            
        # Validate dependencies exist or will exist
        for dep in dependencies:
            if dep not in self.tasks and dep not in [t.task_id for t in self.tasks.values()]:
                logger.warning(f"Dependency {dep} for task {task_id} not found")
        
        task = QuantumTask(
            task_id=task_id,
            name=name,
            description=description,
            function=function,
            dependencies=dependencies,
            priority=priority,
            estimated_duration=estimated_duration,
            quantum_weight=quantum_weight,
            metadata=metadata
        )
        
        self.tasks[task_id] = task
        self.task_graph[task_id] = dependencies
        
        # Update quantum state matrix
        self._update_quantum_state_matrix()
        
        # Create entanglements based on dependencies
        self._create_entanglements(task_id, dependencies)
        
        logger.info(f"Added quantum task: {task_id} ({name}) with {len(dependencies)} dependencies")
        return task
    
    def _create_entanglements(self, task_id: str, dependencies: Set[str]):
        """Create quantum entanglements between dependent tasks"""
        for dep in dependencies:
            if dep in self.tasks:
                # Create bidirectional entanglement
                if task_id not in self.active_entanglements:
                    self.active_entanglements[task_id] = set()
                if dep not in self.active_entanglements:
                    self.active_entanglements[dep] = set()
                    
                self.active_entanglements[task_id].add(dep)
                self.active_entanglements[dep].add(task_id)
                
                # Mark tasks as entangled
                if self.tasks[task_id].state == TaskState.SUPERPOSITION:
                    self.tasks[task_id].collapse_state(TaskState.ENTANGLED)
                if self.tasks[dep].state == TaskState.SUPERPOSITION:
                    self.tasks[dep].collapse_state(TaskState.ENTANGLED)
    
    def _update_quantum_state_matrix(self):
        """Update quantum state probability matrix"""
        n_tasks = len(self.tasks)
        if n_tasks == 0:
            return
            
        # Create quantum state matrix representing task superpositions
        self.quantum_state_matrix = np.zeros((n_tasks, n_tasks), dtype=complex)
        
        task_ids = list(self.tasks.keys())
        for i, task_id in enumerate(task_ids):
            task = self.tasks[task_id]
            
            # Diagonal elements represent individual task amplitudes
            amplitude = task.quantum_weight * np.exp(1j * task.priority.value * np.pi / 10)
            self.quantum_state_matrix[i, i] = amplitude
            
            # Off-diagonal elements represent entanglements
            for dep in task.dependencies:
                if dep in task_ids:
                    j = task_ids.index(dep)
                    entanglement_strength = 0.5  # Base entanglement coupling
                    self.quantum_state_matrix[i, j] = entanglement_strength * amplitude
                    self.quantum_state_matrix[j, i] = entanglement_strength * np.conj(amplitude)
    
    async def execute_quantum_plan(self, 
                                   max_concurrent: Optional[int] = None,
                                   timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute all tasks using quantum-inspired scheduling
        
        Args:
            max_concurrent: Maximum concurrent executions (defaults to max_workers)
            timeout: Global execution timeout in seconds
            
        Returns:
            Execution results with quantum metrics
        """
        if max_concurrent is None:
            max_concurrent = self.max_workers
            
        start_time = time.time()
        executed_tasks = []
        failed_tasks = []
        
        logger.info(f"Starting quantum execution of {len(self.tasks)} tasks")
        
        try:
            # Phase 1: Quantum superposition analysis
            execution_order = self._calculate_quantum_execution_order()
            
            # Phase 2: Execute tasks with quantum tunneling optimization
            async with asyncio.Semaphore(max_concurrent):
                execution_tasks = []
                
                for task_batch in self._create_quantum_batches(execution_order):
                    batch_tasks = [
                        self._execute_single_task(task_id)
                        for task_id in task_batch
                        if self._can_execute_task(task_id)
                    ]
                    
                    if batch_tasks:
                        # Execute batch with quantum interference consideration
                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                        
                        for task_id, result in zip(task_batch, batch_results):
                            if isinstance(result, Exception):
                                failed_tasks.append(task_id)
                                self.tasks[task_id].error = str(result)
                                self.tasks[task_id].collapse_state(TaskState.FAILED)
                            else:
                                executed_tasks.append(task_id)
                                self.tasks[task_id].result = result
                                self.tasks[task_id].collapse_state(TaskState.EXECUTED)
        
        except asyncio.TimeoutError:
            logger.error(f"Quantum execution timed out after {timeout}s")
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate quantum efficiency metrics
        self.total_tasks_executed += len(executed_tasks)
        self.total_execution_time += total_time
        self.quantum_efficiency_score = self._calculate_quantum_efficiency(executed_tasks, total_time)
        
        # Generate execution report
        execution_report = {
            'total_tasks': len(self.tasks),
            'executed_tasks': len(executed_tasks),
            'failed_tasks': len(failed_tasks),
            'execution_time_seconds': total_time,
            'quantum_efficiency_score': self.quantum_efficiency_score,
            'average_task_duration': total_time / max(1, len(executed_tasks)),
            'task_results': {task_id: self.tasks[task_id].result for task_id in executed_tasks},
            'failed_task_errors': {task_id: self.tasks[task_id].error for task_id in failed_tasks},
            'quantum_coherence_maintained': total_time < self.quantum_coherence_time
        }
        
        # Store execution history
        self.execution_history.append({
            'timestamp': start_time,
            'report': execution_report
        })
        
        logger.info(f"Quantum execution completed: {len(executed_tasks)}/{len(self.tasks)} tasks "
                   f"in {total_time:.2f}s (efficiency: {self.quantum_efficiency_score:.3f})")
        
        return execution_report
    
    def _calculate_quantum_execution_order(self) -> List[str]:
        """Calculate optimal execution order using quantum superposition analysis"""
        if not self.quantum_state_matrix is not None:
            # Fallback to topological sort
            return self._topological_sort()
        
        # Quantum-inspired ordering based on eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.quantum_state_matrix.real)
        
        # Sort tasks by quantum probability amplitudes
        task_ids = list(self.tasks.keys())
        quantum_priorities = []
        
        for i, task_id in enumerate(task_ids):
            task = self.tasks[task_id]
            # Combine quantum amplitude with classical priority
            quantum_priority = (
                abs(eigenvalues[i]) * task.priority.value * 
                (1.0 / max(0.1, task.estimated_duration)) * task.quantum_weight
            )
            quantum_priorities.append((quantum_priority, task_id))
        
        # Sort by quantum priority (descending)
        quantum_priorities.sort(reverse=True)
        
        ordered_tasks = [task_id for _, task_id in quantum_priorities]
        
        # Ensure dependency constraints are maintained
        return self._enforce_dependency_constraints(ordered_tasks)
    
    def _topological_sort(self) -> List[str]:
        """Fallback topological sort for dependency ordering"""
        in_degree = {task_id: 0 for task_id in self.tasks}
        
        # Calculate in-degrees
        for task_id in self.tasks:
            for dep in self.tasks[task_id].dependencies:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        # Process nodes with zero in-degree
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority for consistent ordering
            queue.sort(key=lambda x: self.tasks[x].priority.value, reverse=True)
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent tasks
            for task_id in self.tasks:
                if current in self.tasks[task_id].dependencies:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        return result
    
    def _enforce_dependency_constraints(self, ordered_tasks: List[str]) -> List[str]:
        """Ensure dependency constraints are satisfied in execution order"""
        satisfied = []
        remaining = ordered_tasks.copy()
        
        while remaining:
            made_progress = False
            
            for i, task_id in enumerate(remaining):
                task = self.tasks[task_id]
                
                # Check if all dependencies are satisfied
                if all(dep in satisfied for dep in task.dependencies):
                    satisfied.append(task_id)
                    remaining.pop(i)
                    made_progress = True
                    break
            
            if not made_progress:
                # Circular dependency or missing dependency
                logger.warning(f"Dependency constraint violation detected. Remaining tasks: {remaining}")
                # Add remaining tasks in original order
                satisfied.extend(remaining)
                break
        
        return satisfied
    
    def _create_quantum_batches(self, execution_order: List[str]) -> List[List[str]]:
        """Create execution batches considering quantum interference"""
        batches = []
        current_batch = []
        
        for task_id in execution_order:
            task = self.tasks[task_id]
            
            # Check for quantum interference with current batch
            can_add_to_batch = True
            for batch_task_id in current_batch:
                if self._has_quantum_interference(task_id, batch_task_id):
                    can_add_to_batch = False
                    break
            
            if can_add_to_batch and len(current_batch) < self.max_workers:
                current_batch.append(task_id)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [task_id]
        
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def _has_quantum_interference(self, task1_id: str, task2_id: str) -> bool:
        """Check if two tasks have quantum interference (resource conflicts)"""
        task1 = self.tasks[task1_id]
        task2 = self.tasks[task2_id]
        
        # Check for resource conflicts in metadata
        if 'resources' in task1.metadata and 'resources' in task2.metadata:
            resources1 = set(task1.metadata['resources'])
            resources2 = set(task2.metadata['resources'])
            
            if resources1 & resources2:  # Intersection indicates conflict
                return True
        
        # Check for entanglement-based interference
        if task1_id in self.active_entanglements:
            if task2_id in self.active_entanglements[task1_id]:
                return True
        
        return False
    
    def _can_execute_task(self, task_id: str) -> bool:
        """Check if task can be executed (dependencies satisfied)"""
        task = self.tasks[task_id]
        
        # Check if task is in executable state
        if task.state in [TaskState.EXECUTED, TaskState.FAILED]:
            return False
        
        # Check dependency satisfaction
        for dep in task.dependencies:
            if dep not in self.tasks:
                logger.error(f"Dependency {dep} for task {task_id} not found")
                return False
            
            if self.tasks[dep].state != TaskState.EXECUTED:
                return False
        
        return True
    
    async def _execute_single_task(self, task_id: str) -> Any:
        """Execute a single quantum task"""
        task = self.tasks[task_id]
        task.start_time = time.time()
        task.collapse_state(TaskState.COLLAPSED)
        
        logger.info(f"Executing quantum task: {task_id} ({task.name})")
        
        try:
            # Execute task function
            if asyncio.iscoroutinefunction(task.function):
                result = await task.function()
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, task.function)
            
            task.end_time = time.time()
            logger.info(f"Task {task_id} completed successfully in "
                       f"{task.end_time - task.start_time:.3f}s")
            
            return result
            
        except Exception as e:
            task.end_time = time.time()
            logger.error(f"Task {task_id} failed: {e}")
            raise e
    
    def _calculate_quantum_efficiency(self, executed_tasks: List[str], total_time: float) -> float:
        """Calculate quantum efficiency score based on execution performance"""
        if not executed_tasks or total_time <= 0:
            return 0.0
        
        # Base efficiency: tasks per second
        base_efficiency = len(executed_tasks) / total_time
        
        # Quantum bonus factors
        entanglement_bonus = len(self.active_entanglements) * 0.1
        priority_bonus = sum(self.tasks[tid].priority.value for tid in executed_tasks) / (len(executed_tasks) * 5.0)
        coherence_bonus = 1.0 if total_time < self.quantum_coherence_time else 0.5
        
        quantum_efficiency = base_efficiency * (1.0 + entanglement_bonus + priority_bonus) * coherence_bonus
        
        return min(quantum_efficiency, 10.0)  # Cap at 10.0
    
    def get_quantum_state_report(self) -> Dict[str, Any]:
        """Get comprehensive quantum state report"""
        task_states = {}
        for task_id, task in self.tasks.items():
            task_states[task_id] = {
                'name': task.name,
                'state': task.state.value,
                'priority': task.priority.value,
                'quantum_weight': task.quantum_weight,
                'dependencies': list(task.dependencies),
                'duration': task.end_time - task.start_time if task.start_time and task.end_time else None
            }
        
        return {
            'total_tasks': len(self.tasks),
            'quantum_coherence_time': self.quantum_coherence_time,
            'active_entanglements': len(self.active_entanglements),
            'quantum_efficiency_score': self.quantum_efficiency_score,
            'task_states': task_states,
            'execution_history_length': len(self.execution_history)
        }
    
    def clear_completed_tasks(self):
        """Clear completed tasks to free memory"""
        completed_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.state == TaskState.EXECUTED
        ]
        
        for task_id in completed_tasks:
            del self.tasks[task_id]
            if task_id in self.task_graph:
                del self.task_graph[task_id]
            if task_id in self.active_entanglements:
                del self.active_entanglements[task_id]
        
        # Update quantum state matrix
        self._update_quantum_state_matrix()
        
        logger.info(f"Cleared {len(completed_tasks)} completed quantum tasks")