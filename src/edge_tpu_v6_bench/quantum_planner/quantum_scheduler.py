"""
Quantum Scheduler - Advanced scheduling algorithms using quantum-inspired optimization
Implements quantum annealing for resource allocation and temporal optimization
"""

import math
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import Future

from .quantum_task_engine import QuantumTask, TaskState, Priority

logger = logging.getLogger(__name__)

class SchedulingStrategy(Enum):
    QUANTUM_ANNEALING = "quantum_annealing"
    QUANTUM_TUNNELING = "quantum_tunneling"
    ADIABATIC_EVOLUTION = "adiabatic_evolution"
    VARIATIONAL_QUANTUM = "variational_quantum"

@dataclass
class ResourceConstraint:
    """Resource constraint definition"""
    resource_type: str
    max_capacity: float
    current_usage: float = 0.0
    
    def can_allocate(self, requested: float) -> bool:
        return self.current_usage + requested <= self.max_capacity
    
    def allocate(self, amount: float) -> bool:
        if self.can_allocate(amount):
            self.current_usage += amount
            return True
        return False
    
    def deallocate(self, amount: float):
        self.current_usage = max(0.0, self.current_usage - amount)

@dataclass
class SchedulingWindow:
    """Time window for task scheduling"""
    start_time: float
    end_time: float
    capacity: int
    allocated_tasks: List[str]
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def remaining_capacity(self) -> int:
        return max(0, self.capacity - len(self.allocated_tasks))
    
    def can_schedule(self, task_duration: float) -> bool:
        return (self.remaining_capacity > 0 and 
                task_duration <= self.duration)

class QuantumScheduler:
    """
    Quantum-inspired task scheduler with advanced optimization algorithms
    
    Features:
    - Quantum annealing for global optimization
    - Temporal quantum tunneling for deadline satisfaction  
    - Adiabatic evolution for gradual load balancing
    - Variational quantum optimization for complex constraints
    """
    
    def __init__(self,
                 strategy: SchedulingStrategy = SchedulingStrategy.QUANTUM_ANNEALING,
                 max_iterations: int = 1000,
                 temperature: float = 10.0,
                 cooling_rate: float = 0.95,
                 quantum_fluctuation_strength: float = 0.1):
        
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.initial_temperature = temperature
        self.current_temperature = temperature
        self.cooling_rate = cooling_rate
        self.quantum_fluctuation_strength = quantum_fluctuation_strength
        
        # Resource management
        self.resource_constraints: Dict[str, ResourceConstraint] = {}
        self.scheduling_windows: List[SchedulingWindow] = []
        
        # Optimization state
        self.current_schedule: Dict[str, Tuple[float, int]] = {}  # task_id -> (start_time, window_index)
        self.best_schedule: Dict[str, Tuple[float, int]] = {}
        self.best_energy = float('inf')
        
        # Performance metrics
        self.optimization_iterations = 0
        self.convergence_history: List[float] = []
        
        logger.info(f"QuantumScheduler initialized with strategy: {strategy.value}")
    
    def add_resource_constraint(self, 
                              resource_type: str, 
                              max_capacity: float):
        """Add resource constraint for scheduling optimization"""
        self.resource_constraints[resource_type] = ResourceConstraint(
            resource_type=resource_type,
            max_capacity=max_capacity
        )
        logger.info(f"Added resource constraint: {resource_type} (capacity: {max_capacity})")
    
    def create_scheduling_windows(self,
                                start_time: float,
                                end_time: float,
                                window_duration: float,
                                window_capacity: int) -> List[SchedulingWindow]:
        """Create time windows for task scheduling"""
        windows = []
        current_time = start_time
        
        while current_time < end_time:
            window_end = min(current_time + window_duration, end_time)
            windows.append(SchedulingWindow(
                start_time=current_time,
                end_time=window_end,
                capacity=window_capacity,
                allocated_tasks=[]
            ))
            current_time = window_end
        
        self.scheduling_windows = windows
        logger.info(f"Created {len(windows)} scheduling windows of {window_duration}s each")
        return windows
    
    def optimize_schedule(self, 
                         tasks: Dict[str, QuantumTask],
                         deadline: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize task schedule using quantum-inspired algorithms
        
        Args:
            tasks: Dictionary of tasks to schedule
            deadline: Global deadline constraint
            
        Returns:
            Optimized schedule with performance metrics
        """
        logger.info(f"Starting quantum schedule optimization for {len(tasks)} tasks")
        
        if not self.scheduling_windows:
            # Auto-create windows if not specified
            max_duration = max(t.estimated_duration for t in tasks.values()) if tasks else 60.0
            total_duration = sum(t.estimated_duration for t in tasks.values()) if tasks else 300.0
            
            self.create_scheduling_windows(
                start_time=time.time(),
                end_time=time.time() + total_duration + 300,  # Add buffer
                window_duration=max_duration * 2,
                window_capacity=4
            )
        
        # Initialize optimization
        self.optimization_iterations = 0
        self.convergence_history = []
        self.current_temperature = self.initial_temperature
        
        # Generate initial schedule
        self.current_schedule = self._generate_initial_schedule(tasks)
        current_energy = self._calculate_schedule_energy(tasks, self.current_schedule, deadline)
        
        self.best_schedule = self.current_schedule.copy()
        self.best_energy = current_energy
        
        # Run optimization based on selected strategy
        if self.strategy == SchedulingStrategy.QUANTUM_ANNEALING:
            final_schedule = self._quantum_annealing_optimization(tasks, deadline)
        elif self.strategy == SchedulingStrategy.QUANTUM_TUNNELING:
            final_schedule = self._quantum_tunneling_optimization(tasks, deadline)
        elif self.strategy == SchedulingStrategy.ADIABATIC_EVOLUTION:
            final_schedule = self._adiabatic_evolution_optimization(tasks, deadline)
        else:  # VARIATIONAL_QUANTUM
            final_schedule = self._variational_quantum_optimization(tasks, deadline)
        
        # Generate optimization report
        optimization_report = {
            'strategy': self.strategy.value,
            'iterations': self.optimization_iterations,
            'initial_energy': current_energy,
            'final_energy': self.best_energy,
            'improvement_ratio': (current_energy - self.best_energy) / max(1e-9, current_energy),
            'convergence_history': self.convergence_history,
            'schedule': final_schedule,
            'resource_utilization': self._calculate_resource_utilization(tasks, final_schedule),
            'quantum_efficiency': self._calculate_quantum_efficiency(tasks, final_schedule, deadline)
        }
        
        logger.info(f"Quantum optimization completed: {self.optimization_iterations} iterations, "
                   f"energy improvement: {optimization_report['improvement_ratio']:.3f}")
        
        return optimization_report
    
    def _generate_initial_schedule(self, tasks: Dict[str, QuantumTask]) -> Dict[str, Tuple[float, int]]:
        """Generate initial schedule using priority-based heuristic"""
        schedule = {}
        
        # Sort tasks by quantum priority
        sorted_tasks = sorted(
            tasks.items(),
            key=lambda x: (x[1].priority.value, -x[1].quantum_weight, x[1].estimated_duration),
            reverse=True
        )
        
        # Assign tasks to windows greedily
        for task_id, task in sorted_tasks:
            assigned = False
            
            for window_idx, window in enumerate(self.scheduling_windows):
                if window.can_schedule(task.estimated_duration):
                    # Check resource constraints
                    if self._check_resource_feasibility(task, window_idx):
                        schedule[task_id] = (window.start_time, window_idx)
                        window.allocated_tasks.append(task_id)
                        assigned = True
                        break
            
            if not assigned:
                # Force assignment to last window as fallback
                last_window = self.scheduling_windows[-1]
                schedule[task_id] = (last_window.start_time, len(self.scheduling_windows) - 1)
                last_window.allocated_tasks.append(task_id)
        
        return schedule
    
    def _quantum_annealing_optimization(self, 
                                      tasks: Dict[str, QuantumTask],
                                      deadline: Optional[float]) -> Dict[str, Tuple[float, int]]:
        """Optimize schedule using quantum annealing algorithm"""
        current_schedule = self.current_schedule.copy()
        current_energy = self.best_energy
        
        for iteration in range(self.max_iterations):
            # Generate neighboring solution with quantum fluctuations
            neighbor_schedule = self._generate_quantum_neighbor(current_schedule, tasks)
            neighbor_energy = self._calculate_schedule_energy(tasks, neighbor_schedule, deadline)
            
            # Apply quantum annealing acceptance criteria
            energy_delta = neighbor_energy - current_energy
            
            if energy_delta < 0:
                # Always accept improvements
                current_schedule = neighbor_schedule
                current_energy = neighbor_energy
                
                if neighbor_energy < self.best_energy:
                    self.best_schedule = neighbor_schedule.copy()
                    self.best_energy = neighbor_energy
                    
            else:
                # Probabilistically accept worse solutions (quantum tunneling)
                acceptance_probability = math.exp(-energy_delta / max(1e-9, self.current_temperature))
                quantum_acceptance = acceptance_probability * (
                    1.0 + self.quantum_fluctuation_strength * np.random.normal(0, 1)
                )
                
                if np.random.random() < quantum_acceptance:
                    current_schedule = neighbor_schedule
                    current_energy = neighbor_energy
            
            # Cool down temperature
            self.current_temperature *= self.cooling_rate
            self.convergence_history.append(current_energy)
            self.optimization_iterations += 1
            
            # Check convergence
            if iteration > 50 and len(self.convergence_history) > 50:
                recent_improvement = (
                    self.convergence_history[-50] - self.convergence_history[-1]
                ) / max(1e-9, self.convergence_history[-50])
                
                if recent_improvement < 0.001:  # Less than 0.1% improvement
                    logger.info(f"Quantum annealing converged at iteration {iteration}")
                    break
        
        return self.best_schedule
    
    def _quantum_tunneling_optimization(self,
                                      tasks: Dict[str, QuantumTask],
                                      deadline: Optional[float]) -> Dict[str, Tuple[float, int]]:
        """Optimize using quantum tunneling for barrier penetration"""
        current_schedule = self.current_schedule.copy()
        
        for iteration in range(self.max_iterations):
            # Identify energy barriers in schedule
            barrier_tasks = self._identify_schedule_barriers(tasks, current_schedule)
            
            # Apply quantum tunneling to overcome barriers
            if barrier_tasks:
                tunnel_schedule = self._apply_quantum_tunneling(
                    current_schedule, barrier_tasks, tasks
                )
                tunnel_energy = self._calculate_schedule_energy(tasks, tunnel_schedule, deadline)
                
                # Accept if energy is reduced or with quantum probability
                current_energy = self._calculate_schedule_energy(tasks, current_schedule, deadline)
                
                if (tunnel_energy < current_energy or 
                    np.random.random() < self.quantum_fluctuation_strength):
                    current_schedule = tunnel_schedule
                    
                    if tunnel_energy < self.best_energy:
                        self.best_schedule = tunnel_schedule.copy()
                        self.best_energy = tunnel_energy
            
            self.convergence_history.append(
                self._calculate_schedule_energy(tasks, current_schedule, deadline)
            )
            self.optimization_iterations += 1
        
        return self.best_schedule
    
    def _adiabatic_evolution_optimization(self,
                                        tasks: Dict[str, QuantumTask],
                                        deadline: Optional[float]) -> Dict[str, Tuple[float, int]]:
        """Optimize using adiabatic quantum evolution"""
        # Implement gradual transformation from initial to optimal Hamiltonian
        evolution_steps = self.max_iterations
        
        for step in range(evolution_steps):
            # Adiabatic parameter (0 to 1)
            s = step / evolution_steps
            
            # Interpolate between initial and problem Hamiltonian
            mixed_schedule = self._adiabatic_interpolation(
                self.current_schedule, tasks, s
            )
            
            mixed_energy = self._calculate_schedule_energy(tasks, mixed_schedule, deadline)
            
            if mixed_energy < self.best_energy:
                self.best_schedule = mixed_schedule.copy()
                self.best_energy = mixed_energy
            
            self.convergence_history.append(mixed_energy)
            self.optimization_iterations += 1
        
        return self.best_schedule
    
    def _variational_quantum_optimization(self,
                                        tasks: Dict[str, QuantumTask],
                                        deadline: Optional[float]) -> Dict[str, Tuple[float, int]]:
        """Optimize using variational quantum optimization"""
        # Parametric optimization with quantum-inspired updates
        parameters = self._initialize_variational_parameters(len(tasks))
        
        for iteration in range(self.max_iterations):
            # Generate schedule from current parameters
            param_schedule = self._generate_schedule_from_parameters(
                parameters, tasks
            )
            
            param_energy = self._calculate_schedule_energy(tasks, param_schedule, deadline)
            
            # Update parameters using quantum gradient descent
            gradient = self._calculate_quantum_gradient(
                parameters, tasks, deadline
            )
            
            learning_rate = 0.1 / math.sqrt(iteration + 1)
            parameters = parameters - learning_rate * gradient
            
            if param_energy < self.best_energy:
                self.best_schedule = param_schedule.copy()
                self.best_energy = param_energy
            
            self.convergence_history.append(param_energy)
            self.optimization_iterations += 1
        
        return self.best_schedule
    
    def _generate_quantum_neighbor(self, 
                                 current_schedule: Dict[str, Tuple[float, int]],
                                 tasks: Dict[str, QuantumTask]) -> Dict[str, Tuple[float, int]]:
        """Generate neighboring solution with quantum fluctuations"""
        neighbor = current_schedule.copy()
        
        # Select random task for modification
        if not neighbor:
            return neighbor
            
        task_id = np.random.choice(list(neighbor.keys()))
        current_time, current_window = neighbor[task_id]
        
        # Apply quantum perturbation
        quantum_shift = np.random.normal(0, self.quantum_fluctuation_strength)
        
        # Try different window assignments
        available_windows = [
            i for i, window in enumerate(self.scheduling_windows)
            if window.can_schedule(tasks[task_id].estimated_duration)
        ]
        
        if available_windows:
            new_window_idx = np.random.choice(available_windows)
            new_window = self.scheduling_windows[new_window_idx]
            
            # Apply quantum time shift within window
            time_range = new_window.duration - tasks[task_id].estimated_duration
            if time_range > 0:
                time_offset = quantum_shift * time_range
                new_start_time = new_window.start_time + max(0, min(time_range, time_offset))
            else:
                new_start_time = new_window.start_time
            
            neighbor[task_id] = (new_start_time, new_window_idx)
        
        return neighbor
    
    def _calculate_schedule_energy(self,
                                 tasks: Dict[str, QuantumTask],
                                 schedule: Dict[str, Tuple[float, int]],
                                 deadline: Optional[float]) -> float:
        """Calculate energy (cost) of current schedule"""
        energy = 0.0
        
        # Penalty for deadline violations
        if deadline:
            for task_id, (start_time, window_idx) in schedule.items():
                task = tasks[task_id]
                completion_time = start_time + task.estimated_duration
                
                if completion_time > deadline:
                    delay_penalty = (completion_time - deadline) ** 2
                    priority_multiplier = task.priority.value
                    energy += delay_penalty * priority_multiplier
        
        # Resource constraint violations
        window_usage = {}
        for task_id, (start_time, window_idx) in schedule.items():
            if window_idx not in window_usage:
                window_usage[window_idx] = 0
            window_usage[window_idx] += 1
            
            # Penalty for window over-capacity
            window = self.scheduling_windows[window_idx]
            if window_usage[window_idx] > window.capacity:
                energy += (window_usage[window_idx] - window.capacity) ** 2 * 10.0
        
        # Load balancing penalty
        if window_usage:
            usage_variance = np.var(list(window_usage.values()))
            energy += usage_variance * 0.1
        
        # Priority satisfaction bonus (negative energy)
        for task_id, (start_time, window_idx) in schedule.items():
            task = tasks[task_id]
            priority_bonus = -task.priority.value * task.quantum_weight * 0.1
            energy += priority_bonus
        
        return energy
    
    def _identify_schedule_barriers(self,
                                   tasks: Dict[str, QuantumTask],
                                   schedule: Dict[str, Tuple[float, int]]) -> List[str]:
        """Identify tasks creating energy barriers"""
        barriers = []
        
        for task_id, (start_time, window_idx) in schedule.items():
            task = tasks[task_id]
            
            # High-priority tasks scheduled late are barriers
            if (task.priority.value >= 4 and 
                window_idx > len(self.scheduling_windows) // 2):
                barriers.append(task_id)
            
            # Tasks exceeding window capacity are barriers
            window = self.scheduling_windows[window_idx]
            if len(window.allocated_tasks) > window.capacity:
                barriers.append(task_id)
        
        return list(set(barriers))
    
    def _apply_quantum_tunneling(self,
                               schedule: Dict[str, Tuple[float, int]],
                               barrier_tasks: List[str],
                               tasks: Dict[str, QuantumTask]) -> Dict[str, Tuple[float, int]]:
        """Apply quantum tunneling to overcome scheduling barriers"""
        tunneled_schedule = schedule.copy()
        
        for task_id in barrier_tasks:
            # Find optimal window through tunneling
            best_window = None
            best_energy_reduction = 0
            
            for window_idx, window in enumerate(self.scheduling_windows):
                if window.can_schedule(tasks[task_id].estimated_duration):
                    # Calculate energy reduction from moving to this window
                    temp_schedule = tunneled_schedule.copy()
                    temp_schedule[task_id] = (window.start_time, window_idx)
                    
                    current_energy = self._calculate_schedule_energy(tasks, tunneled_schedule, None)
                    new_energy = self._calculate_schedule_energy(tasks, temp_schedule, None)
                    energy_reduction = current_energy - new_energy
                    
                    if energy_reduction > best_energy_reduction:
                        best_energy_reduction = energy_reduction
                        best_window = (window.start_time, window_idx)
            
            if best_window:
                tunneled_schedule[task_id] = best_window
        
        return tunneled_schedule
    
    def _adiabatic_interpolation(self,
                               initial_schedule: Dict[str, Tuple[float, int]],
                               tasks: Dict[str, QuantumTask],
                               s: float) -> Dict[str, Tuple[float, int]]:
        """Perform adiabatic interpolation between initial and optimal schedules"""
        # Simple linear interpolation for demonstration
        # In practice, this would implement proper quantum adiabatic evolution
        
        interpolated_schedule = {}
        
        for task_id, (initial_time, initial_window) in initial_schedule.items():
            # Target: optimal time based on priority
            task = tasks[task_id]
            optimal_window_idx = min(
                task.priority.value - 1,
                len(self.scheduling_windows) - 1
            )
            optimal_time = self.scheduling_windows[optimal_window_idx].start_time
            
            # Interpolate
            current_time = (1 - s) * initial_time + s * optimal_time
            current_window = int((1 - s) * initial_window + s * optimal_window_idx)
            current_window = max(0, min(len(self.scheduling_windows) - 1, current_window))
            
            interpolated_schedule[task_id] = (current_time, current_window)
        
        return interpolated_schedule
    
    def _initialize_variational_parameters(self, num_tasks: int) -> np.ndarray:
        """Initialize parameters for variational optimization"""
        # Initialize parameters for quantum circuit representation
        num_params = num_tasks * 2  # Time and window parameters
        return np.random.normal(0, 0.1, num_params)
    
    def _generate_schedule_from_parameters(self,
                                          parameters: np.ndarray,
                                          tasks: Dict[str, QuantumTask]) -> Dict[str, Tuple[float, int]]:
        """Generate schedule from variational parameters"""
        schedule = {}
        task_list = list(tasks.keys())
        
        for i, task_id in enumerate(task_list):
            if 2 * i + 1 < len(parameters):
                time_param = parameters[2 * i]
                window_param = parameters[2 * i + 1]
                
                # Map parameters to actual schedule
                window_idx = int(abs(window_param) % len(self.scheduling_windows))
                window = self.scheduling_windows[window_idx]
                
                # Map time parameter to window time range
                time_offset = (time_param % 1.0) * window.duration
                start_time = window.start_time + time_offset
                
                schedule[task_id] = (start_time, window_idx)
        
        return schedule
    
    def _calculate_quantum_gradient(self,
                                   parameters: np.ndarray,
                                   tasks: Dict[str, QuantumTask],
                                   deadline: Optional[float]) -> np.ndarray:
        """Calculate quantum gradient for parameter updates"""
        gradient = np.zeros_like(parameters)
        epsilon = 1e-4
        
        base_schedule = self._generate_schedule_from_parameters(parameters, tasks)
        base_energy = self._calculate_schedule_energy(tasks, base_schedule, deadline)
        
        for i in range(len(parameters)):
            # Forward difference
            params_plus = parameters.copy()
            params_plus[i] += epsilon
            
            schedule_plus = self._generate_schedule_from_parameters(params_plus, tasks)
            energy_plus = self._calculate_schedule_energy(tasks, schedule_plus, deadline)
            
            gradient[i] = (energy_plus - base_energy) / epsilon
        
        return gradient
    
    def _check_resource_feasibility(self, 
                                   task: QuantumTask, 
                                   window_idx: int) -> bool:
        """Check if task can be scheduled in window considering resource constraints"""
        if not self.resource_constraints:
            return True
        
        task_resources = task.metadata.get('resources', [])
        if not task_resources:
            return True
        
        # Check if window has sufficient resources
        window = self.scheduling_windows[window_idx]
        
        # Simple check: assume each task in window uses equal share of resources
        if len(window.allocated_tasks) >= window.capacity:
            return False
        
        return True
    
    def _calculate_resource_utilization(self,
                                       tasks: Dict[str, QuantumTask],
                                       schedule: Dict[str, Tuple[float, int]]) -> Dict[str, float]:
        """Calculate resource utilization across schedule"""
        window_utilization = {}
        
        for window_idx, window in enumerate(self.scheduling_windows):
            scheduled_tasks = [
                task_id for task_id, (_, w_idx) in schedule.items()
                if w_idx == window_idx
            ]
            
            utilization = len(scheduled_tasks) / max(1, window.capacity)
            window_utilization[f"window_{window_idx}"] = min(1.0, utilization)
        
        return window_utilization
    
    def _calculate_quantum_efficiency(self,
                                     tasks: Dict[str, QuantumTask],
                                     schedule: Dict[str, Tuple[float, int]],
                                     deadline: Optional[float]) -> float:
        """Calculate quantum scheduling efficiency"""
        if not schedule:
            return 0.0
        
        # Base efficiency: scheduled tasks ratio
        base_efficiency = len(schedule) / len(tasks) if tasks else 0.0
        
        # Priority satisfaction
        priority_satisfaction = 0.0
        for task_id, (start_time, window_idx) in schedule.items():
            task = tasks[task_id]
            
            # Earlier windows are better for high-priority tasks
            window_score = 1.0 - (window_idx / len(self.scheduling_windows))
            priority_weight = task.priority.value / 5.0
            
            priority_satisfaction += window_score * priority_weight
        
        priority_satisfaction /= max(1, len(schedule))
        
        # Deadline satisfaction
        deadline_satisfaction = 1.0
        if deadline:
            violations = 0
            for task_id, (start_time, window_idx) in schedule.items():
                task = tasks[task_id]
                completion_time = start_time + task.estimated_duration
                if completion_time > deadline:
                    violations += 1
            
            deadline_satisfaction = 1.0 - (violations / max(1, len(schedule)))
        
        # Combined quantum efficiency
        quantum_efficiency = (
            0.4 * base_efficiency +
            0.4 * priority_satisfaction +
            0.2 * deadline_satisfaction
        )
        
        return min(1.0, quantum_efficiency)
    
    def get_scheduling_report(self) -> Dict[str, Any]:
        """Get comprehensive scheduling report"""
        return {
            'strategy': self.strategy.value,
            'optimization_iterations': self.optimization_iterations,
            'best_energy': self.best_energy,
            'convergence_history': self.convergence_history,
            'num_windows': len(self.scheduling_windows),
            'resource_constraints': len(self.resource_constraints),
            'quantum_temperature': self.current_temperature,
            'quantum_fluctuation_strength': self.quantum_fluctuation_strength
        }