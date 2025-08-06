"""
Quantum Heuristics - Advanced heuristic algorithms with robust error handling
Implements quantum-inspired optimization heuristics with comprehensive validation
"""

import math
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import asyncio

from .quantum_task_engine import QuantumTask, Priority, TaskState

logger = logging.getLogger(__name__)

class HeuristicType(Enum):
    QUANTUM_GENETIC = "quantum_genetic"
    PARTICLE_SWARM = "quantum_particle_swarm"
    SIMULATED_ANNEALING = "quantum_simulated_annealing"
    ANT_COLONY = "quantum_ant_colony"
    TABU_SEARCH = "quantum_tabu_search"
    VARIABLE_NEIGHBORHOOD = "quantum_variable_neighborhood"

class OptimizationStatus(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    RUNNING = "running"
    CONVERGED = "converged"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class HeuristicConfig:
    """Configuration for quantum heuristic algorithms"""
    algorithm_type: HeuristicType = HeuristicType.QUANTUM_GENETIC
    max_iterations: int = 1000
    population_size: int = 50
    convergence_threshold: float = 1e-6
    timeout_seconds: float = 300.0
    seed: Optional[int] = None
    parallel_workers: int = 4
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 2.0
    elitism_ratio: float = 0.1
    quantum_interference_strength: float = 0.2
    adaptive_parameters: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0 < self.convergence_threshold < 1:
            raise ValueError("convergence_threshold must be between 0 and 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1")
        if not 0 <= self.elitism_ratio <= 1:
            raise ValueError("elitism_ratio must be between 0 and 1")

@dataclass
class HeuristicResult:
    """Result of quantum heuristic optimization"""
    best_solution: Dict[str, Any]
    best_fitness: float
    iterations_completed: int
    convergence_history: List[float]
    status: OptimizationStatus
    execution_time: float
    quantum_metrics: Dict[str, float]
    error_message: Optional[str] = None
    population_diversity: Optional[List[float]] = None
    
    @property
    def success(self) -> bool:
        return self.status in [OptimizationStatus.CONVERGED, OptimizationStatus.TIMEOUT] and self.error_message is None

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class QuantumHeuristicBase(ABC):
    """
    Abstract base class for quantum-inspired heuristic algorithms
    
    Provides common functionality and error handling for all heuristics
    """
    
    def __init__(self, config: HeuristicConfig):
        self.config = config
        self.status = OptimizationStatus.NOT_STARTED
        self.start_time: Optional[float] = None
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_fitness = float('inf')
        self.iteration = 0
        self.convergence_history: List[float] = []
        self.quantum_metrics: Dict[str, float] = {}
        self.population_diversity_history: List[float] = []
        self.error_message: Optional[str] = None
        
        # Set random seed for reproducibility
        if config.seed is not None:
            np.random.seed(config.seed)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config.algorithm_type.value}")
    
    @abstractmethod
    def initialize_population(self, tasks: Dict[str, QuantumTask]) -> List[Dict[str, Any]]:
        """Initialize the population for the heuristic"""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, solution: Dict[str, Any], tasks: Dict[str, QuantumTask]) -> float:
        """Evaluate fitness of a solution"""
        pass
    
    @abstractmethod
    def evolve_population(self, population: List[Dict[str, Any]], tasks: Dict[str, QuantumTask]) -> List[Dict[str, Any]]:
        """Evolve the population for one iteration"""
        pass
    
    def optimize(self, 
                 tasks: Dict[str, QuantumTask],
                 constraints: Optional[Dict[str, Any]] = None) -> HeuristicResult:
        """
        Main optimization method with comprehensive error handling
        
        Args:
            tasks: Dictionary of tasks to optimize
            constraints: Optional constraints dictionary
            
        Returns:
            HeuristicResult with optimization results and metrics
        """
        self.start_time = time.time()
        self.status = OptimizationStatus.INITIALIZING
        
        try:
            # Input validation
            self._validate_inputs(tasks, constraints)
            
            # Initialize optimization
            population = self.initialize_population(tasks)
            self.status = OptimizationStatus.RUNNING
            
            # Main optimization loop
            for self.iteration in range(self.config.max_iterations):
                try:
                    # Check timeout
                    if self._check_timeout():
                        self.status = OptimizationStatus.TIMEOUT
                        break
                    
                    # Evolve population
                    population = self.evolve_population(population, tasks)
                    
                    # Evaluate and update best solution
                    self._update_best_solution(population, tasks)
                    
                    # Check convergence
                    if self._check_convergence():
                        self.status = OptimizationStatus.CONVERGED
                        break
                    
                    # Update metrics
                    self._update_metrics(population, tasks)
                    
                    # Adaptive parameter adjustment
                    if self.config.adaptive_parameters:
                        self._adapt_parameters()
                    
                except Exception as iteration_error:
                    logger.warning(f"Error in iteration {self.iteration}: {iteration_error}")
                    # Continue with next iteration unless it's a critical error
                    if isinstance(iteration_error, (MemoryError, KeyboardInterrupt)):
                        raise
            
            # Finalize results
            execution_time = time.time() - self.start_time
            
            return HeuristicResult(
                best_solution=self.best_solution or {},
                best_fitness=self.best_fitness,
                iterations_completed=self.iteration + 1,
                convergence_history=self.convergence_history.copy(),
                status=self.status,
                execution_time=execution_time,
                quantum_metrics=self.quantum_metrics.copy(),
                population_diversity=self.population_diversity_history.copy(),
                error_message=self.error_message
            )
            
        except Exception as e:
            self.status = OptimizationStatus.ERROR
            self.error_message = str(e)
            execution_time = time.time() - self.start_time if self.start_time else 0.0
            
            logger.error(f"Optimization failed: {e}")
            
            return HeuristicResult(
                best_solution=self.best_solution or {},
                best_fitness=self.best_fitness,
                iterations_completed=self.iteration,
                convergence_history=self.convergence_history.copy(),
                status=self.status,
                execution_time=execution_time,
                quantum_metrics=self.quantum_metrics.copy(),
                error_message=self.error_message
            )
    
    def _validate_inputs(self, tasks: Dict[str, QuantumTask], constraints: Optional[Dict[str, Any]]):
        """Validate input parameters"""
        if not tasks:
            raise ValidationError("Tasks dictionary cannot be empty")
        
        for task_id, task in tasks.items():
            if not isinstance(task, QuantumTask):
                raise ValidationError(f"Invalid task type for {task_id}: {type(task)}")
            
            if task.estimated_duration < 0:
                raise ValidationError(f"Task {task_id} has negative duration: {task.estimated_duration}")
            
            if not 0 <= task.quantum_weight <= 1:
                raise ValidationError(f"Task {task_id} has invalid quantum weight: {task.quantum_weight}")
        
        if constraints:
            self._validate_constraints(constraints)
    
    def _validate_constraints(self, constraints: Dict[str, Any]):
        """Validate constraint dictionary"""
        valid_constraint_types = {'resource_limits', 'deadlines', 'precedence', 'capacity'}
        
        for constraint_type, constraint_data in constraints.items():
            if constraint_type not in valid_constraint_types:
                logger.warning(f"Unknown constraint type: {constraint_type}")
            
            if not isinstance(constraint_data, dict):
                raise ValidationError(f"Constraint {constraint_type} must be a dictionary")
    
    def _check_timeout(self) -> bool:
        """Check if optimization has timed out"""
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        return elapsed > self.config.timeout_seconds
    
    def _update_best_solution(self, population: List[Dict[str, Any]], tasks: Dict[str, QuantumTask]):
        """Update best solution from current population"""
        try:
            for solution in population:
                fitness = self.evaluate_fitness(solution, tasks)
                
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = solution.copy()
            
            self.convergence_history.append(self.best_fitness)
            
        except Exception as e:
            logger.warning(f"Error updating best solution: {e}")
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        if len(self.convergence_history) < 10:
            return False
        
        # Check if improvement is below threshold
        recent_history = self.convergence_history[-10:]
        improvement = abs(recent_history[0] - recent_history[-1])
        relative_improvement = improvement / max(abs(recent_history[0]), 1e-9)
        
        return relative_improvement < self.config.convergence_threshold
    
    def _update_metrics(self, population: List[Dict[str, Any]], tasks: Dict[str, QuantumTask]):
        """Update quantum metrics"""
        try:
            # Population diversity
            diversity = self._calculate_population_diversity(population)
            self.population_diversity_history.append(diversity)
            
            # Quantum coherence (simplified metric)
            coherence = self._calculate_quantum_coherence(population, tasks)
            
            # Update quantum metrics dictionary
            self.quantum_metrics.update({
                'population_diversity': diversity,
                'quantum_coherence': coherence,
                'convergence_rate': self._calculate_convergence_rate(),
                'exploration_exploitation_ratio': self._calculate_exploration_ratio(),
                'quantum_entanglement_strength': self._calculate_entanglement_strength(population, tasks)
            })
            
        except Exception as e:
            logger.warning(f"Error updating metrics: {e}")
    
    def _adapt_parameters(self):
        """Adapt algorithm parameters based on current performance"""
        try:
            if len(self.convergence_history) < 5:
                return
            
            # Reduce mutation rate if converging well
            convergence_rate = self._calculate_convergence_rate()
            if convergence_rate < 0.01:  # Slow convergence
                self.config.mutation_rate = min(0.3, self.config.mutation_rate * 1.1)
            elif convergence_rate > 0.1:  # Fast convergence
                self.config.mutation_rate = max(0.01, self.config.mutation_rate * 0.9)
            
            # Adjust quantum interference based on diversity
            if self.population_diversity_history:
                current_diversity = self.population_diversity_history[-1]
                if current_diversity < 0.1:  # Low diversity
                    self.config.quantum_interference_strength = min(0.5, 
                                                                 self.config.quantum_interference_strength * 1.2)
                elif current_diversity > 0.8:  # High diversity
                    self.config.quantum_interference_strength = max(0.05,
                                                                  self.config.quantum_interference_strength * 0.8)
            
        except Exception as e:
            logger.warning(f"Error adapting parameters: {e}")
    
    def _calculate_population_diversity(self, population: List[Dict[str, Any]]) -> float:
        """Calculate diversity of population"""
        if len(population) < 2:
            return 0.0
        
        try:
            total_distance = 0.0
            comparisons = 0
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    distance = self._solution_distance(population[i], population[j])
                    total_distance += distance
                    comparisons += 1
            
            return total_distance / max(1, comparisons)
            
        except Exception as e:
            logger.warning(f"Error calculating diversity: {e}")
            return 0.0
    
    def _solution_distance(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> float:
        """Calculate distance between two solutions"""
        try:
            if not solution1 or not solution2:
                return 1.0
            
            common_keys = set(solution1.keys()) & set(solution2.keys())
            if not common_keys:
                return 1.0
            
            differences = 0
            for key in common_keys:
                if solution1[key] != solution2[key]:
                    differences += 1
            
            return differences / len(common_keys)
            
        except Exception:
            return 1.0
    
    def _calculate_quantum_coherence(self, population: List[Dict[str, Any]], tasks: Dict[str, QuantumTask]) -> float:
        """Calculate quantum coherence of population"""
        try:
            if not population:
                return 0.0
            
            # Simplified coherence based on solution quality variance
            fitnesses = [self.evaluate_fitness(solution, tasks) for solution in population]
            
            if len(fitnesses) < 2:
                return 1.0
            
            fitness_variance = np.var(fitnesses)
            mean_fitness = np.mean(fitnesses)
            
            # Coherence is higher when solutions are similar (low variance relative to mean)
            coherence = 1.0 / (1.0 + fitness_variance / max(abs(mean_fitness), 1e-9))
            
            return min(1.0, coherence)
            
        except Exception as e:
            logger.warning(f"Error calculating coherence: {e}")
            return 0.0
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate rate of convergence"""
        if len(self.convergence_history) < 5:
            return 0.0
        
        try:
            recent_history = self.convergence_history[-5:]
            improvement = abs(recent_history[0] - recent_history[-1])
            return improvement / max(abs(recent_history[0]), 1e-9)
            
        except Exception:
            return 0.0
    
    def _calculate_exploration_ratio(self) -> float:
        """Calculate exploration vs exploitation ratio"""
        try:
            if not self.population_diversity_history:
                return 0.5
            
            current_diversity = self.population_diversity_history[-1]
            return min(1.0, current_diversity * 2.0)  # Scale diversity to 0-1 range
            
        except Exception:
            return 0.5
    
    def _calculate_entanglement_strength(self, population: List[Dict[str, Any]], tasks: Dict[str, QuantumTask]) -> float:
        """Calculate quantum entanglement strength"""
        try:
            if not population or not tasks:
                return 0.0
            
            # Simplified entanglement based on task dependency correlation
            entanglement_sum = 0.0
            task_pairs = 0
            
            for task_id, task in tasks.items():
                for dep_id in task.dependencies:
                    if dep_id in tasks:
                        # Check correlation in population solutions
                        correlation = self._calculate_task_correlation(population, task_id, dep_id)
                        entanglement_sum += abs(correlation)
                        task_pairs += 1
            
            return entanglement_sum / max(1, task_pairs)
            
        except Exception as e:
            logger.warning(f"Error calculating entanglement: {e}")
            return 0.0
    
    def _calculate_task_correlation(self, population: List[Dict[str, Any]], task1: str, task2: str) -> float:
        """Calculate correlation between task assignments in population"""
        try:
            values1 = []
            values2 = []
            
            for solution in population:
                if task1 in solution and task2 in solution:
                    # Convert assignments to numeric values for correlation
                    val1 = hash(str(solution[task1])) % 1000
                    val2 = hash(str(solution[task2])) % 1000
                    values1.append(val1)
                    values2.append(val2)
            
            if len(values1) < 2:
                return 0.0
            
            correlation = np.corrcoef(values1, values2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0

class QuantumGeneticHeuristic(QuantumHeuristicBase):
    """
    Quantum-inspired genetic algorithm with robust error handling
    """
    
    def initialize_population(self, tasks: Dict[str, QuantumTask]) -> List[Dict[str, Any]]:
        """Initialize population with random solutions"""
        try:
            population = []
            task_list = list(tasks.keys())
            
            for _ in range(self.config.population_size):
                solution = {}
                
                for task_id in task_list:
                    # Random resource assignment
                    resource_id = np.random.randint(0, 4)  # 4 possible resources
                    start_time = np.random.uniform(0, 100)
                    priority_boost = np.random.choice([0, 1])
                    
                    solution[task_id] = {
                        'resource': f'resource_{resource_id}',
                        'start_time': start_time,
                        'priority_boost': priority_boost,
                        'quantum_state': np.random.uniform(0, 1)  # Quantum amplitude
                    }
                
                population.append(solution)
            
            logger.debug(f"Initialized population of size {len(population)}")
            return population
            
        except Exception as e:
            logger.error(f"Failed to initialize population: {e}")
            raise
    
    def evaluate_fitness(self, solution: Dict[str, Any], tasks: Dict[str, QuantumTask]) -> float:
        """Evaluate fitness of a solution with error handling"""
        try:
            if not solution or not tasks:
                return float('inf')
            
            fitness = 0.0
            
            # Makespan objective (minimize maximum completion time)
            max_completion_time = 0.0
            resource_usage = {}
            
            for task_id, assignment in solution.items():
                if task_id not in tasks:
                    continue
                
                task = tasks[task_id]
                
                # Extract assignment details with error handling
                resource = assignment.get('resource', 'resource_0')
                start_time = assignment.get('start_time', 0.0)
                priority_boost = assignment.get('priority_boost', 0)
                
                # Validate values
                start_time = max(0.0, float(start_time))
                completion_time = start_time + task.estimated_duration
                
                max_completion_time = max(max_completion_time, completion_time)
                
                # Resource utilization tracking
                if resource not in resource_usage:
                    resource_usage[resource] = []
                resource_usage[resource].append((start_time, completion_time))
                
                # Priority satisfaction
                priority_score = task.priority.value * (1 + priority_boost * 0.1)
                fitness -= priority_score * 0.01  # Reward high-priority task satisfaction
            
            # Makespan penalty
            fitness += max_completion_time
            
            # Resource conflict penalty
            for resource, intervals in resource_usage.items():
                conflicts = self._count_resource_conflicts(intervals)
                fitness += conflicts * 10.0  # Penalty for resource conflicts
            
            # Load balancing bonus
            if len(resource_usage) > 1:
                loads = [len(intervals) for intervals in resource_usage.values()]
                load_variance = np.var(loads)
                fitness += load_variance  # Penalty for imbalanced loads
            
            return max(0.0, fitness)  # Ensure non-negative fitness
            
        except Exception as e:
            logger.warning(f"Error evaluating fitness: {e}")
            return float('inf')
    
    def evolve_population(self, population: List[Dict[str, Any]], tasks: Dict[str, QuantumTask]) -> List[Dict[str, Any]]:
        """Evolve population using quantum genetic operators"""
        try:
            new_population = []
            
            # Evaluate fitness for all solutions
            fitness_scores = []
            for solution in population:
                fitness = self.evaluate_fitness(solution, tasks)
                fitness_scores.append(fitness)
            
            # Sort by fitness (lower is better)
            population_fitness = list(zip(population, fitness_scores))
            population_fitness.sort(key=lambda x: x[1])
            
            # Elitism - keep best solutions
            elite_count = max(1, int(self.config.elitism_ratio * len(population)))
            for i in range(elite_count):
                new_population.append(population_fitness[i][0].copy())
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                try:
                    # Selection
                    parent1 = self._quantum_selection(population_fitness)
                    parent2 = self._quantum_selection(population_fitness)
                    
                    # Crossover
                    if np.random.random() < self.config.crossover_rate:
                        child1, child2 = self._quantum_crossover(parent1, parent2, tasks)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    child1 = self._quantum_mutation(child1, tasks)
                    child2 = self._quantum_mutation(child2, tasks)
                    
                    new_population.extend([child1, child2])
                    
                except Exception as e:
                    logger.warning(f"Error in offspring generation: {e}")
                    # Add random solution as fallback
                    fallback_population = self.initialize_population(tasks)
                    if fallback_population:
                        new_population.append(fallback_population[0])
            
            # Trim to exact population size
            return new_population[:self.config.population_size]
            
        except Exception as e:
            logger.error(f"Error evolving population: {e}")
            # Return current population as fallback
            return population
    
    def _count_resource_conflicts(self, intervals: List[Tuple[float, float]]) -> int:
        """Count overlapping intervals (resource conflicts)"""
        try:
            if len(intervals) < 2:
                return 0
            
            # Sort intervals by start time
            sorted_intervals = sorted(intervals)
            conflicts = 0
            
            for i in range(len(sorted_intervals) - 1):
                current_end = sorted_intervals[i][1]
                next_start = sorted_intervals[i + 1][0]
                
                if current_end > next_start:
                    conflicts += 1
            
            return conflicts
            
        except Exception:
            return len(intervals)  # Conservative estimate
    
    def _quantum_selection(self, population_fitness: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Quantum tournament selection"""
        try:
            tournament_size = min(3, len(population_fitness))
            tournament_indices = np.random.choice(len(population_fitness), tournament_size, replace=False)
            
            # Add quantum interference to selection
            best_index = tournament_indices[0]
            best_fitness = population_fitness[best_index][1]
            
            for idx in tournament_indices[1:]:
                fitness = population_fitness[idx][1]
                
                # Quantum interference effect
                quantum_boost = np.random.normal(0, self.config.quantum_interference_strength)
                adjusted_fitness = fitness * (1 + quantum_boost)
                
                if adjusted_fitness < best_fitness:
                    best_fitness = adjusted_fitness
                    best_index = idx
            
            return population_fitness[best_index][0].copy()
            
        except Exception as e:
            logger.warning(f"Error in quantum selection: {e}")
            # Fallback to first solution
            return population_fitness[0][0].copy() if population_fitness else {}
    
    def _quantum_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], tasks: Dict[str, QuantumTask]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Quantum crossover with superposition effects"""
        try:
            child1 = {}
            child2 = {}
            
            common_tasks = set(parent1.keys()) & set(parent2.keys())
            
            for task_id in common_tasks:
                if task_id not in tasks:
                    continue
                
                # Quantum crossover probability
                crossover_prob = 0.5 + np.random.normal(0, self.config.quantum_interference_strength)
                crossover_prob = max(0.0, min(1.0, crossover_prob))
                
                if np.random.random() < crossover_prob:
                    child1[task_id] = parent1[task_id].copy()
                    child2[task_id] = parent2[task_id].copy()
                else:
                    child1[task_id] = parent2[task_id].copy()
                    child2[task_id] = parent1[task_id].copy()
                
                # Quantum interference in continuous values
                if 'start_time' in child1[task_id] and 'start_time' in child2[task_id]:
                    interference = np.random.normal(0, 1.0)
                    child1[task_id]['start_time'] += interference
                    child2[task_id]['start_time'] -= interference
                    
                    # Ensure non-negative times
                    child1[task_id]['start_time'] = max(0, child1[task_id]['start_time'])
                    child2[task_id]['start_time'] = max(0, child2[task_id]['start_time'])
                
                # Update quantum states
                if 'quantum_state' in child1[task_id]:
                    child1[task_id]['quantum_state'] = np.random.uniform(0, 1)
                if 'quantum_state' in child2[task_id]:
                    child2[task_id]['quantum_state'] = np.random.uniform(0, 1)
            
            return child1, child2
            
        except Exception as e:
            logger.warning(f"Error in quantum crossover: {e}")
            return parent1.copy(), parent2.copy()
    
    def _quantum_mutation(self, solution: Dict[str, Any], tasks: Dict[str, QuantumTask]) -> Dict[str, Any]:
        """Quantum mutation with probabilistic changes"""
        try:
            mutated = solution.copy()
            
            for task_id, assignment in mutated.items():
                if task_id not in tasks:
                    continue
                
                if np.random.random() < self.config.mutation_rate:
                    # Choose mutation type
                    mutation_type = np.random.choice(['resource', 'time', 'priority', 'quantum'])
                    
                    try:
                        if mutation_type == 'resource':
                            # Change resource assignment
                            current_resource = assignment.get('resource', 'resource_0')
                            resource_num = int(current_resource.split('_')[-1]) if '_' in current_resource else 0
                            new_resource_num = (resource_num + np.random.randint(1, 4)) % 4
                            assignment['resource'] = f'resource_{new_resource_num}'
                            
                        elif mutation_type == 'time' and 'start_time' in assignment:
                            # Quantum time shift
                            current_time = assignment['start_time']
                            time_shift = np.random.normal(0, 5.0)
                            assignment['start_time'] = max(0, current_time + time_shift)
                            
                        elif mutation_type == 'priority':
                            # Priority boost flip
                            assignment['priority_boost'] = 1 - assignment.get('priority_boost', 0)
                            
                        elif mutation_type == 'quantum' and 'quantum_state' in assignment:
                            # Quantum state mutation
                            assignment['quantum_state'] = np.random.uniform(0, 1)
                            
                    except Exception as e:
                        logger.warning(f"Error in mutation of type {mutation_type}: {e}")
                        continue
            
            return mutated
            
        except Exception as e:
            logger.warning(f"Error in quantum mutation: {e}")
            return solution

class QuantumHeuristics:
    """
    Main interface for quantum heuristic algorithms with comprehensive error handling
    """
    
    def __init__(self):
        self.available_algorithms = {
            HeuristicType.QUANTUM_GENETIC: QuantumGeneticHeuristic,
            # Additional algorithms can be added here
        }
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("QuantumHeuristics initialized with available algorithms")
    
    def optimize(self, 
                 tasks: Dict[str, QuantumTask],
                 algorithm: HeuristicType = HeuristicType.QUANTUM_GENETIC,
                 config: Optional[HeuristicConfig] = None,
                 constraints: Optional[Dict[str, Any]] = None) -> HeuristicResult:
        """
        Optimize task assignment using specified quantum heuristic
        
        Args:
            tasks: Dictionary of tasks to optimize
            algorithm: Heuristic algorithm to use
            config: Algorithm configuration
            constraints: Optimization constraints
            
        Returns:
            HeuristicResult with optimization results
        """
        try:
            # Validate inputs
            if not tasks:
                raise ValidationError("Tasks dictionary cannot be empty")
            
            if algorithm not in self.available_algorithms:
                raise ValidationError(f"Algorithm {algorithm} not available")
            
            # Use default config if none provided
            if config is None:
                config = HeuristicConfig(algorithm_type=algorithm)
            
            # Create and run heuristic
            heuristic_class = self.available_algorithms[algorithm]
            heuristic = heuristic_class(config)
            
            result = heuristic.optimize(tasks, constraints)
            
            # Store in history
            self.optimization_history.append({
                'timestamp': time.time(),
                'algorithm': algorithm.value,
                'result': result,
                'num_tasks': len(tasks),
                'success': result.success
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            
            # Return error result
            return HeuristicResult(
                best_solution={},
                best_fitness=float('inf'),
                iterations_completed=0,
                convergence_history=[],
                status=OptimizationStatus.ERROR,
                execution_time=0.0,
                quantum_metrics={},
                error_message=str(e)
            )
    
    async def optimize_async(self,
                            tasks: Dict[str, QuantumTask],
                            algorithm: HeuristicType = HeuristicType.QUANTUM_GENETIC,
                            config: Optional[HeuristicConfig] = None,
                            constraints: Optional[Dict[str, Any]] = None) -> HeuristicResult:
        """Asynchronous optimization with timeout handling"""
        try:
            timeout = config.timeout_seconds if config else 300.0
            
            # Run optimization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor, 
                    self.optimize, 
                    tasks, algorithm, config, constraints
                )
                
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
                
        except asyncio.TimeoutError:
            logger.warning(f"Async optimization timed out after {timeout}s")
            return HeuristicResult(
                best_solution={},
                best_fitness=float('inf'),
                iterations_completed=0,
                convergence_history=[],
                status=OptimizationStatus.TIMEOUT,
                execution_time=timeout,
                quantum_metrics={},
                error_message="Optimization timed out"
            )
        except Exception as e:
            logger.error(f"Async optimization failed: {e}")
            return HeuristicResult(
                best_solution={},
                best_fitness=float('inf'),
                iterations_completed=0,
                convergence_history=[],
                status=OptimizationStatus.ERROR,
                execution_time=0.0,
                quantum_metrics={},
                error_message=str(e)
            )
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        try:
            if not self.optimization_history:
                return {'message': 'No optimizations performed yet'}
            
            successful_runs = [run for run in self.optimization_history if run['success']]
            
            report = {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': len(successful_runs),
                'success_rate': len(successful_runs) / len(self.optimization_history),
                'algorithms_used': list(set(run['algorithm'] for run in self.optimization_history)),
                'average_execution_time': np.mean([
                    run['result'].execution_time for run in successful_runs
                ]) if successful_runs else 0.0,
                'best_fitness_achieved': min([
                    run['result'].best_fitness for run in successful_runs
                ]) if successful_runs else float('inf'),
                'recent_runs': self.optimization_history[-5:]  # Last 5 runs
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}
    
    def clear_history(self):
        """Clear optimization history"""
        self.optimization_history.clear()
        logger.info("Optimization history cleared")