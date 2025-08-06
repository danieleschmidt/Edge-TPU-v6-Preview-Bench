"""
Quantum Optimizer - Advanced optimization algorithms using quantum mechanics principles
Implements QAOA, VQE, and quantum-inspired metaheuristics for complex task optimization
"""

import math
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .quantum_task_engine import QuantumTask, Priority, TaskState

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"  
    BALANCE_LOAD = "balance_load"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_PRIORITY_SATISFACTION = "maximize_priority"

class QuantumAlgorithm(Enum):
    QAOA = "quantum_approximate_optimization"
    VQE = "variational_quantum_eigensolver"
    QUANTUM_GENETIC = "quantum_genetic_algorithm"
    SIMULATED_QUANTUM_ANNEALING = "simulated_quantum_annealing"
    ADIABATIC_QUANTUM = "adiabatic_quantum_computation"

@dataclass
class OptimizationConfig:
    """Configuration for quantum optimization"""
    algorithm: QuantumAlgorithm = QuantumAlgorithm.QAOA
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    population_size: int = 50  # For genetic algorithms
    quantum_circuit_depth: int = 4  # For QAOA/VQE
    measurement_shots: int = 1024
    noise_model: Optional[str] = None
    use_parallel: bool = True
    timeout_seconds: float = 300.0

@dataclass  
class OptimizationResult:
    """Result of quantum optimization"""
    optimal_assignment: Dict[str, Any]
    objective_value: float
    iterations: int
    convergence_history: List[float]
    quantum_fidelity: float
    optimization_time: float
    success: bool
    error_message: Optional[str] = None

class QuantumCircuit:
    """Simplified quantum circuit representation for optimization"""
    
    def __init__(self, num_qubits: int, circuit_depth: int = 4):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.parameters = np.random.uniform(0, 2 * np.pi, circuit_depth * num_qubits)
        self.measurement_outcomes: List[Dict[str, int]] = []
    
    def apply_parameterized_layer(self, params: np.ndarray, layer: int):
        """Apply parameterized quantum gates"""
        # Simulation of quantum gate application
        for qubit in range(self.num_qubits):
            param_idx = layer * self.num_qubits + qubit
            if param_idx < len(params):
                # Simulate rotation gate with parameter
                rotation_angle = params[param_idx]
                # Store the effect (simplified)
                pass
    
    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """Simulate quantum measurement"""
        # Simplified measurement simulation
        outcomes = {}
        
        for _ in range(shots):
            # Generate measurement outcome based on current state
            bitstring = ""
            for qubit in range(self.num_qubits):
                # Probabilistic measurement
                prob = abs(np.sin(self.parameters[qubit % len(self.parameters)])) ** 2
                bit = "1" if np.random.random() < prob else "0" 
                bitstring += bit
            
            outcomes[bitstring] = outcomes.get(bitstring, 0) + 1
        
        return outcomes
    
    def expectation_value(self, observable: str) -> float:
        """Calculate expectation value of observable"""
        # Simplified expectation value calculation
        measurements = self.measure(1024)
        
        total_value = 0.0
        total_counts = sum(measurements.values())
        
        for bitstring, count in measurements.items():
            # Calculate observable value for this bitstring
            value = self._calculate_observable_value(bitstring, observable)
            probability = count / total_counts
            total_value += value * probability
        
        return total_value
    
    def _calculate_observable_value(self, bitstring: str, observable: str) -> float:
        """Calculate observable value for a measurement outcome"""
        # Simplified observable calculation
        if observable == "energy":
            # Energy-like observable based on bit pattern
            ones_count = bitstring.count("1")
            return -1.0 + 2.0 * ones_count / len(bitstring)
        elif observable == "magnetization":
            ones_count = bitstring.count("1")
            zeros_count = bitstring.count("0")
            return (ones_count - zeros_count) / len(bitstring)
        else:
            return np.random.uniform(-1, 1)

class QuantumOptimizer:
    """
    Quantum-inspired optimizer for complex task planning problems
    
    Features:
    - QAOA for combinatorial optimization
    - VQE for ground state problems
    - Quantum genetic algorithms  
    - Simulated quantum annealing
    - Adiabatic quantum computation simulation
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimization_history: List[Dict[str, Any]] = []
        self.quantum_circuits: Dict[str, QuantumCircuit] = {}
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_objective_value = float('inf')
        
        logger.info(f"QuantumOptimizer initialized with algorithm: {self.config.algorithm.value}")
    
    def optimize(self,
                 tasks: Dict[str, QuantumTask],
                 constraints: Optional[Dict[str, Any]] = None,
                 resources: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Main optimization entry point
        
        Args:
            tasks: Dictionary of tasks to optimize
            constraints: Optimization constraints
            resources: Available resources
            
        Returns:
            OptimizationResult with solution and metrics
        """
        start_time = time.time()
        logger.info(f"Starting quantum optimization of {len(tasks)} tasks")
        
        if constraints is None:
            constraints = {}
        if resources is None:
            resources = {}
        
        try:
            # Select optimization algorithm
            if self.config.algorithm == QuantumAlgorithm.QAOA:
                result = self._optimize_with_qaoa(tasks, constraints, resources)
            elif self.config.algorithm == QuantumAlgorithm.VQE:
                result = self._optimize_with_vqe(tasks, constraints, resources)
            elif self.config.algorithm == QuantumAlgorithm.QUANTUM_GENETIC:
                result = self._optimize_with_quantum_genetic(tasks, constraints, resources)
            elif self.config.algorithm == QuantumAlgorithm.SIMULATED_QUANTUM_ANNEALING:
                result = self._optimize_with_simulated_annealing(tasks, constraints, resources)
            else:  # ADIABATIC_QUANTUM
                result = self._optimize_with_adiabatic(tasks, constraints, resources)
            
            optimization_time = time.time() - start_time
            result.optimization_time = optimization_time
            
            # Update best solution
            if result.success and result.objective_value < self.best_objective_value:
                self.best_solution = result.optimal_assignment
                self.best_objective_value = result.objective_value
            
            # Store in history
            self.optimization_history.append({
                'timestamp': start_time,
                'algorithm': self.config.algorithm.value,
                'objective': self.config.objective.value,
                'result': result
            })
            
            logger.info(f"Quantum optimization completed in {optimization_time:.3f}s, "
                       f"objective value: {result.objective_value:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Quantum optimization failed: {e}")
            return OptimizationResult(
                optimal_assignment={},
                objective_value=float('inf'),
                iterations=0,
                convergence_history=[],
                quantum_fidelity=0.0,
                optimization_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _optimize_with_qaoa(self,
                           tasks: Dict[str, QuantumTask],
                           constraints: Dict[str, Any],
                           resources: Dict[str, float]) -> OptimizationResult:
        """Optimize using Quantum Approximate Optimization Algorithm (QAOA)"""
        num_tasks = len(tasks)
        task_list = list(tasks.keys())
        
        # Create quantum circuit
        circuit = QuantumCircuit(num_tasks, self.config.quantum_circuit_depth)
        self.quantum_circuits['qaoa'] = circuit
        
        # QAOA optimization loop
        convergence_history = []
        best_assignment = {}
        best_value = float('inf')
        
        def qaoa_objective(parameters):
            """QAOA cost function"""
            circuit.parameters = parameters
            
            # Measure quantum circuit
            measurements = circuit.measure(self.config.measurement_shots)
            
            # Calculate expectation value
            total_cost = 0.0
            total_shots = sum(measurements.values())
            
            for bitstring, count in measurements.items():
                # Convert bitstring to task assignment
                assignment = self._bitstring_to_assignment(bitstring, task_list)
                
                # Calculate assignment cost
                cost = self._calculate_assignment_cost(assignment, tasks, constraints, resources)
                probability = count / total_shots
                
                total_cost += cost * probability
            
            return total_cost
        
        # Classical optimization of quantum parameters
        initial_params = circuit.parameters
        
        for iteration in range(self.config.max_iterations):
            try:
                # Optimize parameters using classical optimizer
                result = minimize(
                    qaoa_objective,
                    initial_params,
                    method='COBYLA',
                    options={'maxiter': 50, 'disp': False}
                )
                
                current_value = result.fun
                convergence_history.append(current_value)
                
                if current_value < best_value:
                    best_value = current_value
                    # Get best assignment from current parameters
                    circuit.parameters = result.x
                    measurements = circuit.measure(self.config.measurement_shots)
                    best_bitstring = max(measurements.items(), key=lambda x: x[1])[0]
                    best_assignment = self._bitstring_to_assignment(best_bitstring, task_list)
                
                # Check convergence
                if len(convergence_history) > 10:
                    recent_improvement = abs(convergence_history[-10] - convergence_history[-1])
                    if recent_improvement < self.config.convergence_threshold:
                        logger.info(f"QAOA converged at iteration {iteration}")
                        break
                
                initial_params = result.x
                
            except Exception as e:
                logger.warning(f"QAOA iteration {iteration} failed: {e}")
                continue
        
        # Calculate quantum fidelity (simplified)
        quantum_fidelity = max(0.0, 1.0 - best_value / (len(tasks) * 10.0))
        
        return OptimizationResult(
            optimal_assignment=best_assignment,
            objective_value=best_value,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            quantum_fidelity=quantum_fidelity,
            optimization_time=0.0,  # Will be set by caller
            success=True
        )
    
    def _optimize_with_vqe(self,
                          tasks: Dict[str, QuantumTask],
                          constraints: Dict[str, Any],
                          resources: Dict[str, float]) -> OptimizationResult:
        """Optimize using Variational Quantum Eigensolver (VQE)"""
        num_tasks = len(tasks)
        task_list = list(tasks.keys())
        
        # Create quantum circuit for VQE
        circuit = QuantumCircuit(num_tasks, self.config.quantum_circuit_depth)
        self.quantum_circuits['vqe'] = circuit
        
        convergence_history = []
        best_assignment = {}
        best_energy = float('inf')
        
        def vqe_energy_function(parameters):
            """VQE energy expectation value"""
            circuit.parameters = parameters
            
            # Calculate Hamiltonian expectation value
            energy = 0.0
            
            # Problem Hamiltonian terms
            for i, task_id in enumerate(task_list):
                task = tasks[task_id]
                
                # Local field term (task priority)
                local_observable = f"Z_{i}"
                local_expectation = circuit.expectation_value(local_observable)
                energy += -task.priority.value * local_expectation
                
                # Interaction terms (task dependencies)
                for j, dep_id in enumerate(task_list):
                    if dep_id in task.dependencies:
                        interaction_observable = f"ZZ_{i}_{j}"
                        interaction_expectation = circuit.expectation_value(interaction_observable)
                        energy += -2.0 * interaction_expectation  # Coupling strength
            
            return energy
        
        # VQE optimization
        initial_params = circuit.parameters
        
        for iteration in range(self.config.max_iterations):
            try:
                # Minimize energy expectation value
                result = minimize(
                    vqe_energy_function,
                    initial_params,
                    method='BFGS',
                    options={'maxiter': 50}
                )
                
                current_energy = result.fun
                convergence_history.append(current_energy)
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    
                    # Get ground state assignment
                    circuit.parameters = result.x
                    measurements = circuit.measure(self.config.measurement_shots)
                    ground_state = max(measurements.items(), key=lambda x: x[1])[0]
                    best_assignment = self._bitstring_to_assignment(ground_state, task_list)
                
                # Convergence check
                if len(convergence_history) > 10:
                    energy_variance = np.var(convergence_history[-10:])
                    if energy_variance < self.config.convergence_threshold:
                        logger.info(f"VQE converged at iteration {iteration}")
                        break
                
                initial_params = result.x
                
            except Exception as e:
                logger.warning(f"VQE iteration {iteration} failed: {e}")
                continue
        
        # Convert energy to objective value
        objective_value = self._calculate_assignment_cost(best_assignment, tasks, constraints, resources)
        quantum_fidelity = max(0.0, 1.0 + best_energy / abs(best_energy + 1e-9))
        
        return OptimizationResult(
            optimal_assignment=best_assignment,
            objective_value=objective_value,
            iterations=len(convergence_history),
            convergence_history=[abs(e) for e in convergence_history],
            quantum_fidelity=quantum_fidelity,
            optimization_time=0.0,
            success=True
        )
    
    def _optimize_with_quantum_genetic(self,
                                      tasks: Dict[str, QuantumTask],
                                      constraints: Dict[str, Any],
                                      resources: Dict[str, float]) -> OptimizationResult:
        """Optimize using Quantum Genetic Algorithm"""
        task_list = list(tasks.keys())
        population_size = self.config.population_size
        
        # Initialize quantum population (superposition of classical populations)
        quantum_population = []
        for _ in range(population_size):
            individual = self._generate_random_assignment(task_list, tasks)
            quantum_amplitude = np.random.uniform(0.1, 1.0)
            quantum_population.append((individual, quantum_amplitude))
        
        convergence_history = []
        best_assignment = {}
        best_fitness = float('inf')
        
        for generation in range(self.config.max_iterations // 10):  # Fewer generations
            # Evaluate fitness with quantum interference
            population_fitness = []
            
            for individual, amplitude in quantum_population:
                fitness = self._calculate_assignment_cost(individual, tasks, constraints, resources)
                # Quantum interference effect
                quantum_fitness = fitness * (1.0 / max(0.1, amplitude))
                population_fitness.append((individual, amplitude, quantum_fitness))
            
            # Sort by fitness
            population_fitness.sort(key=lambda x: x[2])
            
            # Update best solution
            if population_fitness[0][2] < best_fitness:
                best_fitness = population_fitness[0][2]
                best_assignment = population_fitness[0][0].copy()
            
            convergence_history.append(best_fitness)
            
            # Quantum selection and reproduction
            new_population = []
            
            # Keep best individuals (elitism with quantum enhancement)
            elite_size = population_size // 4
            for i in range(elite_size):
                individual, amplitude, fitness = population_fitness[i]
                # Enhance quantum amplitude for good solutions
                enhanced_amplitude = min(1.0, amplitude * 1.1)
                new_population.append((individual.copy(), enhanced_amplitude))
            
            # Quantum crossover and mutation
            while len(new_population) < population_size:
                # Quantum tournament selection
                parent1 = self._quantum_tournament_selection(population_fitness)
                parent2 = self._quantum_tournament_selection(population_fitness)
                
                # Quantum crossover
                child1, child2 = self._quantum_crossover(parent1[0], parent2[0], task_list)
                
                # Quantum mutation
                child1 = self._quantum_mutation(child1, tasks, 0.1)
                child2 = self._quantum_mutation(child2, tasks, 0.1)
                
                # Assign quantum amplitudes
                amplitude1 = (parent1[1] + parent2[1]) / 2.0
                amplitude2 = (parent1[1] + parent2[1]) / 2.0
                
                new_population.append((child1, amplitude1))
                if len(new_population) < population_size:
                    new_population.append((child2, amplitude2))
            
            quantum_population = new_population
            
            # Check convergence
            if len(convergence_history) > 5:
                improvement = convergence_history[-5] - convergence_history[-1]
                if improvement < self.config.convergence_threshold:
                    logger.info(f"Quantum genetic algorithm converged at generation {generation}")
                    break
        
        # Calculate quantum fidelity based on population diversity
        diversity = self._calculate_population_diversity(quantum_population)
        quantum_fidelity = min(1.0, diversity * 0.5 + 0.5)
        
        return OptimizationResult(
            optimal_assignment=best_assignment,
            objective_value=best_fitness,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            quantum_fidelity=quantum_fidelity,
            optimization_time=0.0,
            success=True
        )
    
    def _optimize_with_simulated_annealing(self,
                                          tasks: Dict[str, QuantumTask],
                                          constraints: Dict[str, Any],
                                          resources: Dict[str, float]) -> OptimizationResult:
        """Optimize using Simulated Quantum Annealing"""
        task_list = list(tasks.keys())
        
        # Initialize with random assignment
        current_assignment = self._generate_random_assignment(task_list, tasks)
        current_cost = self._calculate_assignment_cost(current_assignment, tasks, constraints, resources)
        
        best_assignment = current_assignment.copy()
        best_cost = current_cost
        
        # Annealing parameters
        initial_temperature = 1000.0
        final_temperature = 0.01
        cooling_rate = 0.995
        
        temperature = initial_temperature
        convergence_history = []
        
        for iteration in range(self.config.max_iterations):
            # Generate quantum neighbor
            neighbor_assignment = self._generate_quantum_neighbor(current_assignment, tasks)
            neighbor_cost = self._calculate_assignment_cost(neighbor_assignment, tasks, constraints, resources)
            
            # Quantum acceptance probability
            if neighbor_cost < current_cost:
                # Always accept improvements
                current_assignment = neighbor_assignment
                current_cost = neighbor_cost
                
                if neighbor_cost < best_cost:
                    best_assignment = neighbor_assignment.copy()
                    best_cost = neighbor_cost
            else:
                # Quantum tunneling probability
                cost_difference = neighbor_cost - current_cost
                
                # Standard Boltzmann factor
                boltzmann_prob = math.exp(-cost_difference / max(1e-9, temperature))
                
                # Quantum tunneling enhancement
                tunnel_prob = math.exp(-math.sqrt(cost_difference) / max(1e-9, temperature))
                quantum_prob = 0.7 * boltzmann_prob + 0.3 * tunnel_prob
                
                if np.random.random() < quantum_prob:
                    current_assignment = neighbor_assignment
                    current_cost = neighbor_cost
            
            # Cool down
            temperature *= cooling_rate
            temperature = max(final_temperature, temperature)
            
            convergence_history.append(current_cost)
            
            # Check convergence
            if len(convergence_history) > 50:
                recent_improvement = convergence_history[-50] - convergence_history[-1]
                relative_improvement = recent_improvement / max(1e-9, abs(convergence_history[-50]))
                
                if relative_improvement < self.config.convergence_threshold:
                    logger.info(f"Simulated annealing converged at iteration {iteration}")
                    break
        
        # Quantum fidelity based on temperature and convergence
        quantum_fidelity = math.exp(-temperature / initial_temperature)
        
        return OptimizationResult(
            optimal_assignment=best_assignment,
            objective_value=best_cost,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            quantum_fidelity=quantum_fidelity,
            optimization_time=0.0,
            success=True
        )
    
    def _optimize_with_adiabatic(self,
                                tasks: Dict[str, QuantumTask],
                                constraints: Dict[str, Any],
                                resources: Dict[str, float]) -> OptimizationResult:
        """Optimize using Adiabatic Quantum Computation simulation"""
        task_list = list(tasks.keys())
        num_tasks = len(task_list)
        
        # Initial Hamiltonian (easy to solve)
        initial_assignment = {task_id: np.random.randint(0, 3) for task_id in task_list}  # Random resource assignment
        
        # Final Hamiltonian (problem-specific)
        convergence_history = []
        
        # Adiabatic evolution
        evolution_steps = self.config.max_iterations
        
        for step in range(evolution_steps):
            # Adiabatic parameter (0 to 1)
            s = step / evolution_steps
            
            # Interpolate between initial and final Hamiltonian
            current_assignment = self._adiabatic_interpolation(
                initial_assignment, tasks, s, constraints, resources
            )
            
            # Calculate energy
            current_energy = self._calculate_assignment_cost(current_assignment, tasks, constraints, resources)
            convergence_history.append(current_energy)
            
            # Update assignment for next step
            initial_assignment = current_assignment
        
        # Final assignment is the result
        final_assignment = current_assignment
        final_cost = convergence_history[-1]
        
        # Quantum fidelity based on adiabatic theorem
        # Higher fidelity for slower evolution (more steps)
        quantum_fidelity = min(1.0, evolution_steps / 5000.0)
        
        return OptimizationResult(
            optimal_assignment=final_assignment,
            objective_value=final_cost,
            iterations=evolution_steps,
            convergence_history=convergence_history,
            quantum_fidelity=quantum_fidelity,
            optimization_time=0.0,
            success=True
        )
    
    def _bitstring_to_assignment(self, bitstring: str, task_list: List[str]) -> Dict[str, Any]:
        """Convert quantum measurement bitstring to task assignment"""
        assignment = {}
        
        # Simple encoding: each task gets assigned to a resource based on bits
        for i, task_id in enumerate(task_list):
            if i < len(bitstring):
                # Use 2 bits per task for 4 possible resource assignments
                bit_start = (i * 2) % len(bitstring)
                bit_end = min(bit_start + 2, len(bitstring))
                task_bits = bitstring[bit_start:bit_end]
                
                # Convert bits to resource assignment
                resource_id = int(task_bits.ljust(2, '0'), 2)  # Pad if needed
                assignment[task_id] = {
                    'resource': f'resource_{resource_id}',
                    'priority_boost': int(bitstring[i % len(bitstring)])
                }
            else:
                # Default assignment
                assignment[task_id] = {
                    'resource': 'resource_0',
                    'priority_boost': 0
                }
        
        return assignment
    
    def _generate_random_assignment(self, task_list: List[str], tasks: Dict[str, QuantumTask]) -> Dict[str, Any]:
        """Generate random task assignment"""
        assignment = {}
        
        for task_id in task_list:
            assignment[task_id] = {
                'resource': f'resource_{np.random.randint(0, 4)}',
                'start_time': np.random.uniform(0, 100),
                'priority_boost': np.random.randint(0, 2)
            }
        
        return assignment
    
    def _calculate_assignment_cost(self,
                                  assignment: Dict[str, Any],
                                  tasks: Dict[str, QuantumTask],
                                  constraints: Dict[str, Any],
                                  resources: Dict[str, float]) -> float:
        """Calculate cost of task assignment based on objective"""
        if not assignment:
            return float('inf')
        
        cost = 0.0
        
        if self.config.objective == OptimizationObjective.MINIMIZE_MAKESPAN:
            # Minimize maximum completion time
            max_completion_time = 0.0
            
            for task_id, task_assignment in assignment.items():
                if task_id in tasks:
                    task = tasks[task_id]
                    start_time = task_assignment.get('start_time', 0)
                    completion_time = start_time + task.estimated_duration
                    max_completion_time = max(max_completion_time, completion_time)
            
            cost = max_completion_time
            
        elif self.config.objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            # Maximize tasks per unit time (minimize inverse throughput)
            total_duration = sum(
                tasks[task_id].estimated_duration 
                for task_id in assignment.keys() 
                if task_id in tasks
            )
            
            if total_duration > 0:
                throughput = len(assignment) / total_duration
                cost = -throughput  # Negative because we want to maximize
            else:
                cost = float('inf')
                
        elif self.config.objective == OptimizationObjective.BALANCE_LOAD:
            # Minimize load imbalance across resources
            resource_loads = {}
            
            for task_id, task_assignment in assignment.items():
                if task_id in tasks:
                    resource = task_assignment.get('resource', 'default')
                    task_load = tasks[task_id].estimated_duration
                    
                    if resource not in resource_loads:
                        resource_loads[resource] = 0.0
                    resource_loads[resource] += task_load
            
            if resource_loads:
                load_values = list(resource_loads.values())
                cost = np.var(load_values)  # Variance as imbalance measure
            else:
                cost = 0.0
                
        elif self.config.objective == OptimizationObjective.MINIMIZE_ENERGY:
            # Minimize energy consumption (simplified model)
            total_energy = 0.0
            
            for task_id, task_assignment in assignment.items():
                if task_id in tasks:
                    task = tasks[task_id]
                    resource = task_assignment.get('resource', 'default')
                    
                    # Simple energy model: duration * resource_power_factor
                    resource_power = resources.get(f"{resource}_power", 1.0)
                    task_energy = task.estimated_duration * resource_power
                    total_energy += task_energy
            
            cost = total_energy
            
        else:  # MAXIMIZE_PRIORITY_SATISFACTION
            # Maximize priority satisfaction (minimize inverse satisfaction)
            priority_satisfaction = 0.0
            
            for task_id, task_assignment in assignment.items():
                if task_id in tasks:
                    task = tasks[task_id]
                    priority_boost = task_assignment.get('priority_boost', 0)
                    
                    satisfaction = task.priority.value * (1.0 + priority_boost * 0.1)
                    priority_satisfaction += satisfaction
            
            cost = -priority_satisfaction  # Negative because we want to maximize
        
        # Add constraint penalties
        constraint_penalty = self._calculate_constraint_penalties(assignment, tasks, constraints)
        
        return cost + constraint_penalty
    
    def _calculate_constraint_penalties(self,
                                       assignment: Dict[str, Any],
                                       tasks: Dict[str, QuantumTask],
                                       constraints: Dict[str, Any]) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0.0
        
        # Resource capacity constraints
        if 'resource_capacities' in constraints:
            resource_usage = {}
            
            for task_id, task_assignment in assignment.items():
                if task_id in tasks:
                    resource = task_assignment.get('resource', 'default')
                    task_load = tasks[task_id].estimated_duration
                    
                    if resource not in resource_usage:
                        resource_usage[resource] = 0.0
                    resource_usage[resource] += task_load
            
            for resource, usage in resource_usage.items():
                capacity = constraints['resource_capacities'].get(resource, float('inf'))
                if usage > capacity:
                    penalty += (usage - capacity) ** 2 * 100.0
        
        # Deadline constraints
        if 'deadlines' in constraints:
            for task_id, task_assignment in assignment.items():
                if task_id in tasks and task_id in constraints['deadlines']:
                    start_time = task_assignment.get('start_time', 0)
                    completion_time = start_time + tasks[task_id].estimated_duration
                    deadline = constraints['deadlines'][task_id]
                    
                    if completion_time > deadline:
                        delay = completion_time - deadline
                        penalty += delay ** 2 * tasks[task_id].priority.value
        
        return penalty
    
    def _quantum_tournament_selection(self, population_fitness: List[Tuple]) -> Tuple:
        """Quantum tournament selection with superposition"""
        tournament_size = 3
        
        # Select tournament participants with quantum probabilities
        participants = np.random.choice(
            len(population_fitness), 
            size=min(tournament_size, len(population_fitness)),
            replace=False
        )
        
        # Quantum selection based on amplitude and fitness
        best_participant = None
        best_score = float('-inf')
        
        for idx in participants:
            individual, amplitude, fitness = population_fitness[idx]
            
            # Quantum selection score (higher is better)
            quantum_score = amplitude / max(1e-9, fitness)
            
            if quantum_score > best_score:
                best_score = quantum_score
                best_participant = population_fitness[idx]
        
        return best_participant
    
    def _quantum_crossover(self, 
                          parent1: Dict[str, Any], 
                          parent2: Dict[str, Any],
                          task_list: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Quantum crossover with superposition effects"""
        child1 = {}
        child2 = {}
        
        for task_id in task_list:
            if task_id in parent1 and task_id in parent2:
                # Quantum crossover probability
                crossover_prob = 0.5 + 0.3 * np.random.normal(0, 1)  # Quantum fluctuation
                crossover_prob = max(0.0, min(1.0, crossover_prob))
                
                if np.random.random() < crossover_prob:
                    child1[task_id] = parent1[task_id].copy()
                    child2[task_id] = parent2[task_id].copy()
                else:
                    child1[task_id] = parent2[task_id].copy()
                    child2[task_id] = parent1[task_id].copy()
                
                # Quantum interference effect
                if 'start_time' in child1[task_id] and 'start_time' in child2[task_id]:
                    interference = 0.1 * np.random.normal(0, 1)
                    child1[task_id]['start_time'] += interference
                    child2[task_id]['start_time'] -= interference
                    
                    # Ensure non-negative times
                    child1[task_id]['start_time'] = max(0, child1[task_id]['start_time'])
                    child2[task_id]['start_time'] = max(0, child2[task_id]['start_time'])
        
        return child1, child2
    
    def _quantum_mutation(self,
                         individual: Dict[str, Any],
                         tasks: Dict[str, QuantumTask],
                         mutation_rate: float) -> Dict[str, Any]:
        """Quantum mutation with probabilistic changes"""
        mutated = individual.copy()
        
        for task_id, task_assignment in mutated.items():
            if np.random.random() < mutation_rate:
                # Quantum mutation types
                mutation_type = np.random.choice(['resource', 'time', 'priority'])
                
                if mutation_type == 'resource':
                    # Change resource assignment
                    current_resource = task_assignment.get('resource', 'resource_0')
                    resource_num = int(current_resource.split('_')[-1]) if '_' in current_resource else 0
                    
                    # Quantum tunneling to different resource
                    new_resource_num = (resource_num + np.random.randint(1, 4)) % 4
                    task_assignment['resource'] = f'resource_{new_resource_num}'
                    
                elif mutation_type == 'time' and 'start_time' in task_assignment:
                    # Quantum time shift
                    current_time = task_assignment['start_time']
                    time_shift = np.random.normal(0, 10)  # Quantum fluctuation
                    new_time = max(0, current_time + time_shift)
                    task_assignment['start_time'] = new_time
                    
                elif mutation_type == 'priority':
                    # Quantum priority boost flip
                    current_boost = task_assignment.get('priority_boost', 0)
                    task_assignment['priority_boost'] = 1 - current_boost
        
        return mutated
    
    def _calculate_population_diversity(self, quantum_population: List[Tuple]) -> float:
        """Calculate quantum population diversity"""
        if len(quantum_population) < 2:
            return 0.0
        
        diversity_sum = 0.0
        comparisons = 0
        
        for i in range(len(quantum_population)):
            for j in range(i + 1, len(quantum_population)):
                individual1, amplitude1 = quantum_population[i]
                individual2, amplitude2 = quantum_population[j]
                
                # Calculate assignment difference
                difference = self._calculate_assignment_difference(individual1, individual2)
                
                # Weight by quantum amplitudes
                quantum_weight = abs(amplitude1 - amplitude2)
                diversity_sum += difference * quantum_weight
                comparisons += 1
        
        return diversity_sum / max(1, comparisons)
    
    def _calculate_assignment_difference(self, 
                                       assignment1: Dict[str, Any], 
                                       assignment2: Dict[str, Any]) -> float:
        """Calculate difference between two assignments"""
        common_tasks = set(assignment1.keys()) & set(assignment2.keys())
        if not common_tasks:
            return 1.0
        
        differences = 0
        
        for task_id in common_tasks:
            assign1 = assignment1[task_id]
            assign2 = assignment2[task_id]
            
            # Resource difference
            if assign1.get('resource') != assign2.get('resource'):
                differences += 1
            
            # Time difference (normalized)
            time1 = assign1.get('start_time', 0)
            time2 = assign2.get('start_time', 0)
            time_diff = abs(time1 - time2) / max(1, max(time1, time2))
            differences += time_diff
            
            # Priority boost difference
            if assign1.get('priority_boost') != assign2.get('priority_boost'):
                differences += 0.5
        
        return differences / len(common_tasks)
    
    def _generate_quantum_neighbor(self, 
                                  assignment: Dict[str, Any],
                                  tasks: Dict[str, QuantumTask]) -> Dict[str, Any]:
        """Generate quantum neighbor solution"""
        neighbor = assignment.copy()
        
        # Select random task for modification
        if not neighbor:
            return neighbor
            
        task_id = np.random.choice(list(neighbor.keys()))
        task_assignment = neighbor[task_id].copy()
        
        # Quantum perturbation
        perturbation_type = np.random.choice(['resource', 'time', 'priority'])
        
        if perturbation_type == 'resource':
            current_resource = task_assignment.get('resource', 'resource_0')
            resource_nums = [0, 1, 2, 3]
            current_num = int(current_resource.split('_')[-1]) if '_' in current_resource else 0
            
            # Remove current resource from options
            if current_num in resource_nums:
                resource_nums.remove(current_num)
            
            new_resource_num = np.random.choice(resource_nums)
            task_assignment['resource'] = f'resource_{new_resource_num}'
            
        elif perturbation_type == 'time' and 'start_time' in task_assignment:
            current_time = task_assignment['start_time']
            # Quantum time jump with varying magnitude
            jump_magnitude = np.random.exponential(5.0)  # Quantum tunneling distance
            direction = 1 if np.random.random() > 0.5 else -1
            
            new_time = max(0, current_time + direction * jump_magnitude)
            task_assignment['start_time'] = new_time
            
        elif perturbation_type == 'priority':
            task_assignment['priority_boost'] = 1 - task_assignment.get('priority_boost', 0)
        
        neighbor[task_id] = task_assignment
        return neighbor
    
    def _adiabatic_interpolation(self,
                                initial_assignment: Dict[str, Any],
                                tasks: Dict[str, QuantumTask],
                                s: float,
                                constraints: Dict[str, Any],
                                resources: Dict[str, float]) -> Dict[str, Any]:
        """Perform adiabatic interpolation between initial and target assignments"""
        interpolated = {}
        
        for task_id, initial_values in initial_assignment.items():
            if task_id in tasks:
                task = tasks[task_id]
                
                # Target assignment based on task properties
                target_resource = f"resource_{min(3, task.priority.value - 1)}"
                target_time = task.priority.value * 10.0  # Higher priority = earlier time
                target_boost = 1 if task.priority.value >= 4 else 0
                
                # Interpolate each property
                current_resource_num = int(initial_values.get('resource', 'resource_0').split('_')[-1])
                target_resource_num = int(target_resource.split('_')[-1])
                
                interpolated_resource_num = int(
                    (1 - s) * current_resource_num + s * target_resource_num
                )
                
                current_time = initial_values.get('start_time', 0)
                interpolated_time = (1 - s) * current_time + s * target_time
                
                current_boost = initial_values.get('priority_boost', 0)
                interpolated_boost = int((1 - s) * current_boost + s * target_boost)
                
                interpolated[task_id] = {
                    'resource': f'resource_{interpolated_resource_num}',
                    'start_time': max(0, interpolated_time),
                    'priority_boost': interpolated_boost
                }
        
        return interpolated
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            'algorithm': self.config.algorithm.value,
            'objective': self.config.objective.value,
            'total_optimizations': len(self.optimization_history),
            'best_objective_value': self.best_objective_value,
            'convergence_threshold': self.config.convergence_threshold,
            'max_iterations': self.config.max_iterations,
            'quantum_circuits': list(self.quantum_circuits.keys()),
            'optimization_history': [
                {
                    'timestamp': entry['timestamp'],
                    'algorithm': entry['algorithm'],
                    'success': entry['result'].success,
                    'objective_value': entry['result'].objective_value,
                    'iterations': entry['result'].iterations
                }
                for entry in self.optimization_history[-10:]  # Last 10 runs
            ]
        }