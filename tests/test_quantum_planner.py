"""
Comprehensive test suite for Quantum Task Planner
Tests all components with edge cases and performance validation
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Set, Any
import numpy as np

from edge_tpu_v6_bench.quantum_planner import (
    QuantumTaskEngine, QuantumTask, TaskState, Priority,
    QuantumScheduler, QuantumOptimizer, TaskGraph,
    QuantumHeuristics, PerformanceOptimizer
)
from edge_tpu_v6_bench.quantum_planner.quantum_scheduler import SchedulingStrategy
from edge_tpu_v6_bench.quantum_planner.quantum_optimizer import OptimizationObjective, QuantumAlgorithm
from edge_tpu_v6_bench.quantum_planner.quantum_heuristics import HeuristicType, HeuristicConfig
from edge_tpu_v6_bench.quantum_planner.performance_optimizer import ScalingConfig, ScalingMode, CacheStrategy
from edge_tpu_v6_bench.quantum_planner.task_graph import GraphValidationError, CircularDependencyError

class TestQuantumTask:
    """Test QuantumTask data structure and methods"""
    
    def test_quantum_task_creation(self):
        """Test basic QuantumTask creation"""
        def dummy_func():
            return "test"
        
        task = QuantumTask(
            task_id="test_task",
            name="Test Task",
            description="A test task",
            function=dummy_func,
            dependencies={"dep1", "dep2"},
            estimated_duration=10.0,
            priority=Priority.HIGH,
            quantum_weight=0.8
        )
        
        assert task.task_id == "test_task"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.dependencies == {"dep1", "dep2"}
        assert task.estimated_duration == 10.0
        assert task.priority == Priority.HIGH
        assert task.quantum_weight == 0.8
        assert task.state == TaskState.SUPERPOSITION
        assert task.result is None
        assert task.error is None
    
    def test_quantum_task_state_collapse(self):
        """Test quantum state collapse functionality"""
        def dummy_func():
            return "test"
        
        task = QuantumTask(
            task_id="test_task",
            name="Test Task",
            description="A test task",
            function=dummy_func
        )
        
        assert task.state == TaskState.SUPERPOSITION
        
        task.collapse_state(TaskState.ENTANGLED)
        assert task.state == TaskState.ENTANGLED
        
        task.collapse_state(TaskState.EXECUTED)
        assert task.state == TaskState.EXECUTED
    
    def test_quantum_task_validation(self):
        """Test QuantumTask input validation"""
        def dummy_func():
            return "test"
        
        # Valid task
        task = QuantumTask(
            task_id="valid_task",
            name="Valid Task",
            description="Valid",
            function=dummy_func
        )
        assert task.task_id == "valid_task"
        
        # Test edge cases
        task_minimal = QuantumTask(
            task_id="minimal",
            name="Minimal",
            description="",
            function=dummy_func
        )
        assert task_minimal.description == ""
        assert task_minimal.dependencies == set()

class TestQuantumTaskEngine:
    """Test QuantumTaskEngine core functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create a test QuantumTaskEngine"""
        return QuantumTaskEngine(max_workers=2, quantum_coherence_time=10.0)
    
    def test_engine_initialization(self, engine):
        """Test QuantumTaskEngine initialization"""
        assert engine.max_workers == 2
        assert engine.quantum_coherence_time == 10.0
        assert len(engine.tasks) == 0
        assert len(engine.task_graph) == 0
        assert engine.total_tasks_executed == 0
    
    def test_add_task_basic(self, engine):
        """Test basic task addition"""
        def test_func():
            return "test_result"
        
        task = engine.add_task(
            task_id="task1",
            name="Task 1",
            function=test_func,
            description="Test task",
            priority=Priority.HIGH,
            estimated_duration=5.0,
            quantum_weight=0.9
        )
        
        assert task.task_id == "task1"
        assert task.name == "Task 1"
        assert task.priority == Priority.HIGH
        assert len(engine.tasks) == 1
        assert "task1" in engine.tasks
    
    def test_add_task_with_dependencies(self, engine):
        """Test task addition with dependencies"""
        def dummy_func():
            return "result"
        
        # Add parent task first
        engine.add_task(
            task_id="parent",
            name="Parent Task",
            function=dummy_func
        )
        
        # Add dependent task
        task = engine.add_task(
            task_id="child",
            name="Child Task", 
            function=dummy_func,
            dependencies={"parent"}
        )
        
        assert "child" in engine.tasks
        assert "parent" in engine.tasks["child"].dependencies
        assert "child" in engine.task_graph
        assert engine.task_graph["child"] == {"parent"}
    
    @pytest.mark.asyncio
    async def test_execute_quantum_plan_simple(self, engine):
        """Test simple quantum plan execution"""
        results = []
        
        def task1_func():
            results.append("task1")
            return "task1_result"
        
        def task2_func():
            results.append("task2")
            return "task2_result"
        
        engine.add_task("task1", "Task 1", task1_func, estimated_duration=0.1)
        engine.add_task("task2", "Task 2", task2_func, estimated_duration=0.1)
        
        execution_result = await engine.execute_quantum_plan()
        
        assert execution_result['total_tasks'] == 2
        assert execution_result['executed_tasks'] == 2
        assert execution_result['failed_tasks'] == 0
        assert execution_result['task_results']['task1'] == "task1_result"
        assert execution_result['task_results']['task2'] == "task2_result"
        assert len(results) == 2
        assert "task1" in results
        assert "task2" in results
    
    @pytest.mark.asyncio
    async def test_execute_quantum_plan_with_dependencies(self, engine):
        """Test execution with dependencies"""
        execution_order = []
        
        def parent_func():
            execution_order.append("parent")
            time.sleep(0.01)  # Small delay
            return "parent_result"
        
        def child_func():
            execution_order.append("child")
            return "child_result"
        
        engine.add_task("parent", "Parent", parent_func, estimated_duration=0.1)
        engine.add_task("child", "Child", child_func, dependencies={"parent"}, estimated_duration=0.1)
        
        result = await engine.execute_quantum_plan()
        
        assert result['executed_tasks'] == 2
        assert len(execution_order) == 2
        assert execution_order.index("parent") < execution_order.index("child")
    
    @pytest.mark.asyncio
    async def test_execute_quantum_plan_with_failure(self, engine):
        """Test execution with task failures"""
        def success_func():
            return "success"
        
        def failure_func():
            raise ValueError("Task failed")
        
        engine.add_task("success", "Success Task", success_func)
        engine.add_task("failure", "Failure Task", failure_func)
        
        result = await engine.execute_quantum_plan()
        
        assert result['total_tasks'] == 2
        assert result['executed_tasks'] == 1
        assert result['failed_tasks'] == 1
        assert 'success' in result['task_results']
        assert 'failure' in result['failed_task_errors']
    
    def test_quantum_state_matrix_update(self, engine):
        """Test quantum state matrix updates"""
        def dummy_func():
            return "result"
        
        # Initially empty
        assert engine.quantum_state_matrix is None
        
        # Add tasks
        engine.add_task("task1", "Task 1", dummy_func, quantum_weight=0.8)
        engine.add_task("task2", "Task 2", dummy_func, quantum_weight=0.6)
        
        # Matrix should be created and have correct dimensions
        assert engine.quantum_state_matrix is not None
        assert engine.quantum_state_matrix.shape == (2, 2)
        
        # Diagonal elements should reflect quantum weights
        assert engine.quantum_state_matrix[0, 0] != 0
        assert engine.quantum_state_matrix[1, 1] != 0

class TestTaskGraph:
    """Test TaskGraph functionality"""
    
    @pytest.fixture
    def task_graph(self):
        """Create a test TaskGraph"""
        return TaskGraph()
    
    def test_task_graph_initialization(self, task_graph):
        """Test TaskGraph initialization"""
        assert len(task_graph.tasks) == 0
        assert len(task_graph.edges) == 0
        assert len(task_graph.adjacency_list) == 0
        assert len(task_graph.reverse_adjacency) == 0
        assert task_graph.is_sealed == False
    
    def test_add_task_to_graph(self, task_graph):
        """Test adding tasks to graph"""
        def dummy_func():
            return "result"
        
        task = QuantumTask(
            task_id="test_task",
            name="Test Task",
            description="Test",
            function=dummy_func
        )
        
        success = task_graph.add_task(task)
        
        assert success == True
        assert "test_task" in task_graph.tasks
        assert "test_task" in task_graph.adjacency_list
        assert "test_task" in task_graph.reverse_adjacency
    
    def test_add_dependency(self, task_graph):
        """Test adding dependencies between tasks"""
        def dummy_func():
            return "result"
        
        task1 = QuantumTask("task1", "Task 1", "First task", dummy_func)
        task2 = QuantumTask("task2", "Task 2", "Second task", dummy_func)
        
        task_graph.add_task(task1)
        task_graph.add_task(task2)
        
        # Add dependency: task2 depends on task1
        success = task_graph.add_dependency("task1", "task2")
        
        assert success == True
        assert "task1" in task_graph.adjacency_list
        assert "task2" in task_graph.adjacency_list["task1"]
        assert "task1" in task_graph.reverse_adjacency["task2"]
    
    def test_circular_dependency_detection(self, task_graph):
        """Test circular dependency detection"""
        def dummy_func():
            return "result"
        
        # Create tasks
        for i in range(3):
            task = QuantumTask(f"task{i}", f"Task {i}", f"Task {i}", dummy_func)
            task_graph.add_task(task)
        
        # Create circular dependency: task0 -> task1 -> task2 -> task0
        task_graph.add_dependency("task0", "task1")
        task_graph.add_dependency("task1", "task2")
        
        # This should detect the circular dependency
        with pytest.raises(CircularDependencyError):
            task_graph.add_dependency("task2", "task0")
    
    def test_topological_sort(self, task_graph):
        """Test topological sorting"""
        def dummy_func():
            return "result"
        
        # Create tasks with dependencies
        tasks_data = [
            ("A", []),
            ("B", ["A"]),
            ("C", ["A"]),
            ("D", ["B", "C"])
        ]
        
        for task_id, deps in tasks_data:
            task = QuantumTask(task_id, f"Task {task_id}", "", dummy_func, dependencies=set(deps))
            task_graph.add_task(task)
        
        execution_order = task_graph.get_execution_order("topological")
        
        # A should come before B and C
        assert execution_order.index("A") < execution_order.index("B")
        assert execution_order.index("A") < execution_order.index("C")
        # B and C should come before D
        assert execution_order.index("B") < execution_order.index("D")
        assert execution_order.index("C") < execution_order.index("D")
    
    def test_graph_serialization(self, task_graph):
        """Test graph serialization and deserialization"""
        def dummy_func():
            return "result"
        
        # Create a simple graph
        task1 = QuantumTask("task1", "Task 1", "First", dummy_func, estimated_duration=5.0)
        task2 = QuantumTask("task2", "Task 2", "Second", dummy_func, dependencies={"task1"})
        
        task_graph.add_task(task1)
        task_graph.add_task(task2)
        
        # Serialize
        serialized = task_graph.serialize()
        
        assert 'graph_id' in serialized
        assert 'tasks' in serialized
        assert 'edges' in serialized
        assert len(serialized['tasks']) == 2
        assert 'task1' in serialized['tasks']
        assert 'task2' in serialized['tasks']
        
        # Create new graph and deserialize
        new_graph = TaskGraph()
        success = new_graph.deserialize(serialized)
        
        assert success == True
        assert len(new_graph.tasks) == 2
        assert "task1" in new_graph.tasks
        assert "task2" in new_graph.tasks
        assert new_graph.tasks["task2"].dependencies == {"task1"}
    
    def test_graph_validation(self, task_graph):
        """Test comprehensive graph validation"""
        def dummy_func():
            return "result"
        
        # Add valid tasks
        task1 = QuantumTask("task1", "Task 1", "Valid task", dummy_func, estimated_duration=1.0)
        task_graph.add_task(task1)
        
        # Validate should pass
        validation_result = task_graph.validate_graph(raise_on_error=False)
        assert validation_result['is_valid'] == True
        assert len(validation_result['errors']) == 0

class TestQuantumScheduler:
    """Test QuantumScheduler functionality"""
    
    @pytest.fixture
    def scheduler(self):
        """Create a test QuantumScheduler"""
        return QuantumScheduler(
            strategy=SchedulingStrategy.QUANTUM_ANNEALING,
            max_iterations=100,
            temperature=10.0
        )
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler.strategy == SchedulingStrategy.QUANTUM_ANNEALING
        assert scheduler.max_iterations == 100
        assert scheduler.initial_temperature == 10.0
        assert len(scheduler.resource_constraints) == 0
        assert len(scheduler.scheduling_windows) == 0
    
    def test_add_resource_constraint(self, scheduler):
        """Test adding resource constraints"""
        scheduler.add_resource_constraint("cpu", 4.0)
        scheduler.add_resource_constraint("memory", 8.0)
        
        assert len(scheduler.resource_constraints) == 2
        assert "cpu" in scheduler.resource_constraints
        assert scheduler.resource_constraints["cpu"].max_capacity == 4.0
        assert scheduler.resource_constraints["memory"].max_capacity == 8.0
    
    def test_create_scheduling_windows(self, scheduler):
        """Test creating scheduling windows"""
        start_time = 0.0
        end_time = 100.0
        window_duration = 10.0
        capacity = 2
        
        windows = scheduler.create_scheduling_windows(
            start_time, end_time, window_duration, capacity
        )
        
        assert len(windows) == 10  # 100 / 10 = 10 windows
        assert windows[0].start_time == 0.0
        assert windows[0].end_time == 10.0
        assert windows[0].capacity == 2
        assert windows[-1].start_time == 90.0
        assert windows[-1].end_time == 100.0
    
    def test_optimize_schedule_basic(self, scheduler):
        """Test basic schedule optimization"""
        def dummy_func():
            return "result"
        
        # Create tasks
        tasks = {}
        for i in range(3):
            task = QuantumTask(
                task_id=f"task{i}",
                name=f"Task {i}",
                description=f"Task {i}",
                function=dummy_func,
                estimated_duration=5.0,
                priority=Priority.MEDIUM
            )
            tasks[f"task{i}"] = task
        
        # Create scheduling windows
        scheduler.create_scheduling_windows(0.0, 50.0, 10.0, 2)
        
        # Optimize
        result = scheduler.optimize_schedule(tasks)
        
        assert 'strategy' in result
        assert 'iterations' in result
        assert 'schedule' in result
        assert result['strategy'] == 'quantum_annealing'
        assert len(result['schedule']) == 3

class TestQuantumOptimizer:
    """Test QuantumOptimizer functionality"""
    
    @pytest.fixture
    def optimizer(self):
        """Create a test QuantumOptimizer"""
        from edge_tpu_v6_bench.quantum_planner.quantum_optimizer import OptimizationConfig
        
        config = OptimizationConfig(
            algorithm=QuantumAlgorithm.QAOA,
            objective=OptimizationObjective.MINIMIZE_MAKESPAN,
            max_iterations=100
        )
        return QuantumOptimizer(config)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.config.algorithm == QuantumAlgorithm.QAOA
        assert optimizer.config.objective == OptimizationObjective.MINIMIZE_MAKESPAN
        assert optimizer.config.max_iterations == 100
        assert len(optimizer.optimization_history) == 0
    
    def test_optimize_basic(self, optimizer):
        """Test basic optimization"""
        def dummy_func():
            return "result"
        
        # Create simple tasks
        tasks = {}
        for i in range(2):
            task = QuantumTask(
                task_id=f"task{i}",
                name=f"Task {i}",
                description=f"Task {i}",
                function=dummy_func,
                estimated_duration=1.0 + i,
                priority=Priority.MEDIUM
            )
            tasks[f"task{i}"] = task
        
        # Run optimization
        result = optimizer.optimize(tasks)
        
        assert result.success == True
        assert result.best_fitness < float('inf')
        assert len(result.best_solution) == 2
        assert result.iterations_completed > 0

class TestQuantumHeuristics:
    """Test QuantumHeuristics functionality"""
    
    @pytest.fixture
    def heuristics(self):
        """Create QuantumHeuristics instance"""
        return QuantumHeuristics()
    
    def test_heuristics_initialization(self, heuristics):
        """Test heuristics initialization"""
        assert len(heuristics.available_algorithms) > 0
        assert HeuristicType.QUANTUM_GENETIC in heuristics.available_algorithms
        assert len(heuristics.optimization_history) == 0
    
    def test_optimize_genetic(self, heuristics):
        """Test quantum genetic optimization"""
        def dummy_func():
            return "result"
        
        # Create test tasks
        tasks = {}
        for i in range(3):
            task = QuantumTask(
                task_id=f"task{i}",
                name=f"Task {i}",
                description=f"Task {i}",
                function=dummy_func,
                estimated_duration=2.0,
                priority=Priority.MEDIUM
            )
            tasks[f"task{i}"] = task
        
        # Configure optimization
        config = HeuristicConfig(
            algorithm_type=HeuristicType.QUANTUM_GENETIC,
            max_iterations=50,
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        # Run optimization
        result = heuristics.optimize(tasks, HeuristicType.QUANTUM_GENETIC, config)
        
        assert result.success == True
        assert result.best_fitness < float('inf')
        assert result.iterations_completed > 0
        assert len(result.best_solution) == 3

class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality"""
    
    @pytest.fixture
    def performance_config(self):
        """Create performance configuration"""
        return ScalingConfig(
            mode=ScalingMode.MULTI_THREADED,
            max_workers=2,
            memory_limit_mb=1024,
            cache_strategy=CacheStrategy.MEMORY_ONLY
        )
    
    @pytest.fixture
    def perf_optimizer(self, performance_config):
        """Create PerformanceOptimizer instance"""
        return PerformanceOptimizer(performance_config)
    
    def test_performance_optimizer_init(self, perf_optimizer):
        """Test performance optimizer initialization"""
        assert perf_optimizer.config.mode == ScalingMode.MULTI_THREADED
        assert perf_optimizer.config.max_workers == 2
        assert perf_optimizer.cache is not None
        assert perf_optimizer.executor is not None
    
    def test_cache_functionality(self, perf_optimizer):
        """Test caching functionality"""
        cache = perf_optimizer.cache
        
        # Test set and get
        key = "test_key"
        value = {"data": "test_value"}
        
        success = cache.set(key, value)
        assert success == True
        
        retrieved = cache.get(key)
        assert retrieved == value
        
        # Test cache miss
        missing = cache.get("nonexistent_key")
        assert missing is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert 'total_gets' in stats
        assert 'total_sets' in stats
        assert stats['total_sets'] >= 1
        assert stats['total_gets'] >= 2  # One hit, one miss
    
    @pytest.mark.asyncio
    async def test_optimize_task_execution(self, perf_optimizer):
        """Test optimized task execution"""
        # Create test tasks
        results = []
        
        def task1_func():
            results.append("task1")
            return "task1_result"
        
        def task2_func():
            results.append("task2") 
            return "task2_result"
        
        tasks = {
            "task1": QuantumTask("task1", "Task 1", "First task", task1_func, estimated_duration=0.1),
            "task2": QuantumTask("task2", "Task 2", "Second task", task2_func, estimated_duration=0.1)
        }
        
        task_functions = {
            "task1": task1_func,
            "task2": task2_func
        }
        
        dependencies = {
            "task1": set(),
            "task2": set()
        }
        
        # Execute with performance optimization
        performance_result = await perf_optimizer.optimize_task_execution(
            tasks, task_functions, dependencies
        )
        
        assert 'results' in performance_result
        assert 'execution_time' in performance_result
        assert 'cache_stats' in performance_result
        assert performance_result['tasks_completed'] == 2
        assert performance_result['tasks_failed'] == 0
        assert len(results) == 2

class TestIntegration:
    """Integration tests combining multiple components"""
    
    @pytest.mark.asyncio
    async def test_full_quantum_planning_pipeline(self):
        """Test complete quantum planning pipeline"""
        # Initialize components
        engine = QuantumTaskEngine(max_workers=2)
        
        execution_order = []
        
        def task_a_func():
            execution_order.append("A")
            time.sleep(0.01)
            return "A_result"
        
        def task_b_func():
            execution_order.append("B")
            time.sleep(0.01) 
            return "B_result"
        
        def task_c_func():
            execution_order.append("C")
            time.sleep(0.01)
            return "C_result"
        
        # Create tasks with dependencies: A -> B -> C
        engine.add_task("A", "Task A", task_a_func, estimated_duration=0.1)
        engine.add_task("B", "Task B", task_b_func, dependencies={"A"}, estimated_duration=0.1)
        engine.add_task("C", "Task C", task_c_func, dependencies={"B"}, estimated_duration=0.1)
        
        # Execute quantum plan
        result = await engine.execute_quantum_plan()
        
        # Verify execution
        assert result['executed_tasks'] == 3
        assert result['failed_tasks'] == 0
        assert len(execution_order) == 3
        
        # Verify dependency order
        assert execution_order.index("A") < execution_order.index("B")
        assert execution_order.index("B") < execution_order.index("C")
        
        # Verify results
        assert result['task_results']['A'] == "A_result"
        assert result['task_results']['B'] == "B_result" 
        assert result['task_results']['C'] == "C_result"
    
    def test_quantum_planning_with_optimization(self):
        """Test quantum planning with scheduling optimization"""
        # Create scheduler
        scheduler = QuantumScheduler(
            strategy=SchedulingStrategy.QUANTUM_ANNEALING,
            max_iterations=50
        )
        
        # Create tasks
        def dummy_func():
            return "result"
        
        tasks = {}
        for i in range(4):
            task = QuantumTask(
                task_id=f"task{i}",
                name=f"Task {i}",
                description=f"Task {i}",
                function=dummy_func,
                estimated_duration=float(i + 1),
                priority=Priority(min(5, i + 1))
            )
            tasks[f"task{i}"] = task
        
        # Create windows
        scheduler.create_scheduling_windows(0.0, 50.0, 10.0, 2)
        
        # Optimize schedule
        result = scheduler.optimize_schedule(tasks)
        
        assert result['strategy'] == 'quantum_annealing'
        assert len(result['schedule']) == 4
        assert result['quantum_efficiency'] > 0
        assert result['final_energy'] <= result['initial_energy']

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_task_execution(self):
        """Test execution with no tasks"""
        engine = QuantumTaskEngine()
        
        # Should handle empty execution gracefully
        result = asyncio.run(engine.execute_quantum_plan())
        
        assert result['total_tasks'] == 0
        assert result['executed_tasks'] == 0
        assert result['failed_tasks'] == 0
    
    def test_task_with_missing_dependencies(self):
        """Test task with non-existent dependencies"""
        engine = QuantumTaskEngine()
        
        def dummy_func():
            return "result"
        
        # Add task with non-existent dependency
        with pytest.raises(Exception):
            engine.add_task(
                "task1",
                "Task 1", 
                dummy_func,
                dependencies={"nonexistent_task"}
            )
    
    def test_scheduler_no_windows(self):
        """Test scheduler with no scheduling windows"""
        scheduler = QuantumScheduler()
        
        def dummy_func():
            return "result"
        
        task = QuantumTask("task1", "Task 1", "Test", dummy_func)
        tasks = {"task1": task}
        
        # Should auto-create windows
        result = scheduler.optimize_schedule(tasks)
        
        assert 'schedule' in result
        assert len(scheduler.scheduling_windows) > 0
    
    def test_graph_validation_errors(self):
        """Test graph validation with errors"""
        graph = TaskGraph()
        
        def dummy_func():
            return "result"
        
        # Add task with invalid dependency reference
        task1 = QuantumTask("task1", "Task 1", "Test", dummy_func, dependencies={"missing_task"})
        graph.add_task(task1, validate=False)  # Skip validation during add
        
        # Validation should catch the error
        validation_result = graph.validate_graph(raise_on_error=False)
        assert validation_result['is_valid'] == False
        assert len(validation_result['errors']) > 0

class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.asyncio
    async def test_large_task_set_execution(self):
        """Test execution with large number of tasks"""
        engine = QuantumTaskEngine(max_workers=4)
        
        # Create 50 tasks
        def make_task_func(task_id):
            def task_func():
                time.sleep(0.001)  # Minimal delay
                return f"{task_id}_result"
            return task_func
        
        for i in range(50):
            engine.add_task(
                f"task_{i}",
                f"Task {i}",
                make_task_func(f"task_{i}"),
                estimated_duration=0.01
            )
        
        start_time = time.time()
        result = await engine.execute_quantum_plan(timeout=10.0)
        execution_time = time.time() - start_time
        
        assert result['executed_tasks'] == 50
        assert execution_time < 5.0  # Should complete in reasonable time
        assert result['quantum_efficiency_score'] > 0
    
    def test_optimization_performance(self):
        """Test optimization algorithm performance"""
        from edge_tpu_v6_bench.quantum_planner.quantum_optimizer import OptimizationConfig
        
        config = OptimizationConfig(
            algorithm=QuantumAlgorithm.QAOA,
            max_iterations=50,  # Reduced for performance test
            convergence_threshold=1e-3
        )
        
        optimizer = QuantumOptimizer(config)
        
        def dummy_func():
            return "result"
        
        # Create 10 tasks
        tasks = {}
        for i in range(10):
            task = QuantumTask(
                task_id=f"task{i}",
                name=f"Task {i}",
                description="Test task",
                function=dummy_func,
                estimated_duration=float(i + 1),
                priority=Priority.MEDIUM
            )
            tasks[f"task{i}"] = task
        
        start_time = time.time()
        result = optimizer.optimize(tasks)
        optimization_time = time.time() - start_time
        
        assert result.success == True
        assert optimization_time < 10.0  # Should complete reasonably fast
        assert result.iterations_completed <= config.max_iterations

# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

def test_module_imports():
    """Test that all modules can be imported successfully"""
    from edge_tpu_v6_bench.quantum_planner import (
        QuantumTaskEngine, QuantumTask, TaskState, Priority,
        QuantumScheduler, QuantumOptimizer, TaskGraph,
        QuantumHeuristics, PerformanceOptimizer
    )
    
    # Verify all classes are importable and can be instantiated
    engine = QuantumTaskEngine()
    assert engine is not None
    
    def dummy_func():
        return "test"
    
    task = QuantumTask("test", "Test", "Test task", dummy_func)
    assert task is not None
    
    graph = TaskGraph()
    assert graph is not None
    
    heuristics = QuantumHeuristics()
    assert heuristics is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])