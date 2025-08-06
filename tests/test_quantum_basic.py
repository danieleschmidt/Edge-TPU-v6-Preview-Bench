"""
Basic functionality tests for Quantum Task Planner without heavy dependencies
Tests core logic and algorithms independently
"""

import sys
import os
import time
import asyncio
from typing import Dict, List, Set, Any

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_quantum_task_basic():
    """Test basic QuantumTask functionality without external dependencies"""
    try:
        from edge_tpu_v6_bench.quantum_planner.quantum_task_engine import QuantumTask, Priority, TaskState
        
        def dummy_func():
            return "test_result"
        
        # Test task creation
        task = QuantumTask(
            task_id="test_task",
            name="Test Task", 
            description="A test task",
            function=dummy_func,
            dependencies={"dep1"},
            estimated_duration=5.0,
            priority=Priority.HIGH,
            quantum_weight=0.8
        )
        
        assert task.task_id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == Priority.HIGH
        assert task.quantum_weight == 0.8
        assert task.state == TaskState.SUPERPOSITION
        
        # Test state collapse
        task.collapse_state(TaskState.EXECUTED)
        assert task.state == TaskState.EXECUTED
        
        print("✓ QuantumTask basic functionality works")
        return True
        
    except Exception as e:
        print(f"✗ QuantumTask test failed: {e}")
        return False

def test_task_graph_basic():
    """Test basic TaskGraph functionality"""
    try:
        from edge_tpu_v6_bench.quantum_planner.task_graph import TaskGraph
        from edge_tpu_v6_bench.quantum_planner.quantum_task_engine import QuantumTask, Priority
        
        def dummy_func():
            return "result"
        
        # Create graph
        graph = TaskGraph()
        
        # Create tasks
        task1 = QuantumTask("task1", "Task 1", "First task", dummy_func)
        task2 = QuantumTask("task2", "Task 2", "Second task", dummy_func, dependencies={"task1"})
        
        # Add tasks
        success1 = graph.add_task(task1)
        success2 = graph.add_task(task2)
        
        assert success1 == True
        assert success2 == True
        assert len(graph.tasks) == 2
        assert "task1" in graph.tasks
        assert "task2" in graph.tasks
        
        # Test execution order
        execution_order = graph.get_execution_order()
        assert len(execution_order) == 2
        assert execution_order.index("task1") < execution_order.index("task2")
        
        print("✓ TaskGraph basic functionality works")
        return True
        
    except Exception as e:
        print(f"✗ TaskGraph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantum_engine_basic():
    """Test basic QuantumTaskEngine functionality"""  
    try:
        from edge_tpu_v6_bench.quantum_planner.quantum_task_engine import QuantumTaskEngine, QuantumTask, Priority
        
        # Create engine
        engine = QuantumTaskEngine(max_workers=2, quantum_coherence_time=10.0)
        
        assert engine.max_workers == 2
        assert engine.quantum_coherence_time == 10.0
        assert len(engine.tasks) == 0
        
        def test_func():
            return "test_result"
        
        # Add task
        task = engine.add_task(
            task_id="task1",
            name="Task 1",
            function=test_func,
            description="Test task",
            priority=Priority.MEDIUM,
            estimated_duration=1.0
        )
        
        assert task.task_id == "task1"
        assert len(engine.tasks) == 1
        
        print("✓ QuantumTaskEngine basic functionality works")
        return True
        
    except Exception as e:
        print(f"✗ QuantumTaskEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_quantum_execution_basic():
    """Test basic quantum task execution"""
    try:
        from edge_tpu_v6_bench.quantum_planner.quantum_task_engine import QuantumTaskEngine
        
        engine = QuantumTaskEngine(max_workers=1)
        
        results = []
        
        def task1_func():
            results.append("task1")
            return "task1_result"
        
        def task2_func():
            results.append("task2")
            return "task2_result"
        
        # Add tasks
        engine.add_task("task1", "Task 1", task1_func, estimated_duration=0.01)
        engine.add_task("task2", "Task 2", task2_func, estimated_duration=0.01)
        
        # Execute
        result = await engine.execute_quantum_plan()
        
        assert result['total_tasks'] == 2
        assert result['executed_tasks'] == 2
        assert result['failed_tasks'] == 0
        assert len(results) == 2
        assert "task1" in results
        assert "task2" in results
        
        print("✓ Quantum task execution works")
        return True
        
    except Exception as e:
        print(f"✗ Quantum execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependency_execution():
    """Test execution with dependencies"""
    async def run_dependency_test():
        try:
            from edge_tpu_v6_bench.quantum_planner.quantum_task_engine import QuantumTaskEngine
            
            engine = QuantumTaskEngine(max_workers=2)
            
            execution_order = []
            
            def parent_func():
                execution_order.append("parent")
                time.sleep(0.001)  # Small delay
                return "parent_result"
            
            def child_func():
                execution_order.append("child")
                return "child_result"
            
            # Add tasks with dependency
            engine.add_task("parent", "Parent", parent_func, estimated_duration=0.01)
            engine.add_task("child", "Child", child_func, dependencies={"parent"}, estimated_duration=0.01)
            
            result = await engine.execute_quantum_plan()
            
            assert result['executed_tasks'] == 2
            assert len(execution_order) == 2
            assert execution_order.index("parent") < execution_order.index("child")
            
            print("✓ Dependency execution works")
            return True
            
        except Exception as e:
            print(f"✗ Dependency execution test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return asyncio.run(run_dependency_test())

def test_serialization_basic():
    """Test basic serialization functionality"""
    try:
        from edge_tpu_v6_bench.quantum_planner.task_graph import TaskGraph
        from edge_tpu_v6_bench.quantum_planner.quantum_task_engine import QuantumTask
        
        def dummy_func():
            return "result"
        
        # Create graph with tasks
        graph = TaskGraph()
        task = QuantumTask("task1", "Task 1", "Test task", dummy_func, estimated_duration=2.0)
        graph.add_task(task)
        
        # Serialize
        serialized = graph.serialize()
        
        assert 'graph_id' in serialized
        assert 'tasks' in serialized
        assert 'task1' in serialized['tasks']
        assert serialized['tasks']['task1']['name'] == "Task 1"
        assert serialized['tasks']['task1']['estimated_duration'] == 2.0
        
        print("✓ Serialization works")
        return True
        
    except Exception as e:
        print(f"✗ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all basic tests"""
    print("Running Quantum Task Planner Basic Tests")
    print("=" * 50)
    
    tests = [
        test_quantum_task_basic,
        test_task_graph_basic,
        test_quantum_engine_basic,
        test_dependency_execution,
        test_serialization_basic
    ]
    
    # Run async test separately
    async def async_test():
        return await test_quantum_execution_basic()
    
    passed = 0
    total = len(tests) + 1
    
    # Run sync tests
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    # Run async test
    try:
        if asyncio.run(async_test()):
            passed += 1
    except Exception as e:
        print(f"✗ Async test crashed: {e}")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)