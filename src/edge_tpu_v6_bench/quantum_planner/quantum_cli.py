"""
Quantum CLI - Command-line interface for quantum task planning
Production-ready CLI with comprehensive features and robust error handling
"""

import asyncio
import time
import json
import logging
import sys
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.json import JSON
from rich import print as rich_print

from .quantum_task_engine import QuantumTaskEngine, QuantumTask, Priority, TaskState
from .quantum_scheduler import QuantumScheduler, SchedulingStrategy
from .quantum_optimizer import QuantumOptimizer, OptimizationObjective, QuantumAlgorithm
from .task_graph import TaskGraph
from .quantum_heuristics import QuantumHeuristics, HeuristicType
from .performance_optimizer import PerformanceOptimizer, ScalingConfig, ScalingMode, CacheStrategy

# Initialize Rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuantumPlannerCLI:
    """Main CLI application class"""
    
    def __init__(self):
        self.console = console
        self.config_path = Path.home() / '.quantum_planner' / 'config.yaml'
        self.task_engine: Optional[QuantumTaskEngine] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Configuration loaded from {self.config_path}")
                    return config
            else:
                # Default configuration
                default_config = {
                    'quantum_engine': {
                        'max_workers': 4,
                        'quantum_coherence_time': 30.0
                    },
                    'performance': {
                        'scaling_mode': 'auto_scale',
                        'max_workers': 8,
                        'memory_limit_mb': 8192,
                        'cache_strategy': 'adaptive'
                    },
                    'logging': {
                        'level': 'INFO',
                        'file': None
                    }
                }
                
                self._save_config(default_config)
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

# CLI Command Groups
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """Quantum-Inspired Task Planner CLI
    
    Advanced task planning using quantum computing principles for optimal
    resource allocation and scheduling.
    """
    ctx.ensure_object(dict)
    
    # Initialize CLI application
    cli_app = QuantumPlannerCLI()
    
    if config:
        cli_app.config_path = Path(config)
        cli_app.config = cli_app._load_config()
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        cli_app.config.setdefault('logging', {})['level'] = 'DEBUG'
    
    ctx.obj['app'] = cli_app

@cli.group()
@click.pass_context
def task(ctx):
    """Task management commands"""
    pass

@task.command('add')
@click.option('--id', 'task_id', required=True, help='Unique task identifier')
@click.option('--name', required=True, help='Task name')
@click.option('--description', default='', help='Task description')
@click.option('--duration', type=float, default=0.0, help='Estimated duration in seconds')
@click.option('--priority', type=click.Choice(['1', '2', '3', '4', '5']), default='3', help='Task priority (1=low, 5=critical)')
@click.option('--quantum-weight', type=float, default=1.0, help='Quantum probability weight (0.0-1.0)')
@click.option('--dependencies', help='Comma-separated list of dependency task IDs')
@click.option('--metadata', help='JSON metadata for the task')
@click.pass_context
def add_task(ctx, task_id, name, description, duration, priority, quantum_weight, dependencies, metadata):
    """Add a new quantum task"""
    app = ctx.obj['app']
    
    try:
        # Initialize task engine if needed
        if not app.task_engine:
            engine_config = app.config.get('quantum_engine', {})
            app.task_engine = QuantumTaskEngine(
                max_workers=engine_config.get('max_workers', 4),
                quantum_coherence_time=engine_config.get('quantum_coherence_time', 30.0)
            )
        
        # Parse dependencies
        deps = set()
        if dependencies:
            deps = set(dep.strip() for dep in dependencies.split(','))
        
        # Parse metadata
        task_metadata = {}
        if metadata:
            try:
                task_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON metadata: {e}[/red]")
                return
        
        # Create dummy function (in real use, this would be provided differently)
        def dummy_function():
            return f"Task {task_id} executed successfully"
        
        # Add task
        task = app.task_engine.add_task(
            task_id=task_id,
            name=name,
            function=dummy_function,
            description=description,
            dependencies=deps,
            priority=Priority(int(priority)),
            estimated_duration=duration,
            quantum_weight=quantum_weight,
            **task_metadata
        )
        
        console.print(f"[green]Task '{task_id}' added successfully[/green]")
        
        # Display task details
        table = Table(title="Task Details")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("ID", task.task_id)
        table.add_row("Name", task.name)
        table.add_row("Description", task.description)
        table.add_row("Duration", f"{task.estimated_duration:.2f}s")
        table.add_row("Priority", task.priority.name)
        table.add_row("Quantum Weight", f"{task.quantum_weight:.2f}")
        table.add_row("Dependencies", ", ".join(task.dependencies) if task.dependencies else "None")
        table.add_row("State", task.state.value)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Failed to add task: {e}[/red]")
        logger.error(f"Task addition failed: {e}")

@task.command('list')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.option('--state', type=click.Choice(['superposition', 'collapsed', 'entangled', 'executed', 'failed']), help='Filter by task state')
@click.pass_context
def list_tasks(ctx, format, state):
    """List all quantum tasks"""
    app = ctx.obj['app']
    
    if not app.task_engine or not app.task_engine.tasks:
        console.print("[yellow]No tasks found[/yellow]")
        return
    
    tasks = app.task_engine.tasks
    
    # Filter by state if specified
    if state:
        task_state = TaskState(state)
        tasks = {tid: task for tid, task in tasks.items() if task.state == task_state}
    
    if format == 'json':
        task_data = {}
        for task_id, task in tasks.items():
            task_data[task_id] = {
                'name': task.name,
                'description': task.description,
                'duration': task.estimated_duration,
                'priority': task.priority.value,
                'quantum_weight': task.quantum_weight,
                'dependencies': list(task.dependencies),
                'state': task.state.value,
                'metadata': task.metadata
            }
        
        console.print(JSON.from_data(task_data))
        
    elif format == 'yaml':
        task_data = {}
        for task_id, task in tasks.items():
            task_data[task_id] = {
                'name': task.name,
                'description': task.description,
                'duration': task.estimated_duration,
                'priority': task.priority.value,
                'quantum_weight': task.quantum_weight,
                'dependencies': list(task.dependencies),
                'state': task.state.value,
                'metadata': task.metadata
            }
        
        console.print(yaml.dump(task_data, default_flow_style=False))
        
    else:  # table format
        table = Table(title="Quantum Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Duration", justify="right")
        table.add_column("Priority", style="yellow")
        table.add_column("Quantum Weight", justify="right")
        table.add_column("Dependencies", style="blue")
        table.add_column("State", style="green")
        
        for task_id, task in tasks.items():
            deps_str = ", ".join(task.dependencies) if task.dependencies else "-"
            if len(deps_str) > 20:
                deps_str = deps_str[:17] + "..."
            
            table.add_row(
                task_id,
                task.name,
                f"{task.estimated_duration:.1f}s",
                task.priority.name,
                f"{task.quantum_weight:.2f}",
                deps_str,
                task.state.value
            )
        
        console.print(table)

@task.command('execute')
@click.option('--max-concurrent', type=int, help='Maximum concurrent executions')
@click.option('--timeout', type=float, help='Global execution timeout in seconds')
@click.option('--strategy', type=click.Choice(['quantum_annealing', 'quantum_tunneling', 'adiabatic_evolution']), 
              default='quantum_annealing', help='Scheduling strategy')
@click.pass_context
def execute_tasks(ctx, max_concurrent, timeout, strategy):
    """Execute all quantum tasks"""
    app = ctx.obj['app']
    
    if not app.task_engine or not app.task_engine.tasks:
        console.print("[red]No tasks to execute[/red]")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Executing quantum plan...", total=None)
            
            # Execute tasks
            result = asyncio.run(
                app.task_engine.execute_quantum_plan(
                    max_concurrent=max_concurrent,
                    timeout=timeout
                )
            )
            
            progress.update(task, description="[green]Execution completed!")
        
        # Display results
        console.print("\n[bold green]Quantum Execution Results[/bold green]")
        
        table = Table(title="Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Tasks", str(result['total_tasks']))
        table.add_row("Executed Tasks", str(result['executed_tasks']))
        table.add_row("Failed Tasks", str(result['failed_tasks']))
        table.add_row("Execution Time", f"{result['execution_time_seconds']:.2f}s")
        table.add_row("Quantum Efficiency", f"{result['quantum_efficiency_score']:.3f}")
        table.add_row("Average Duration", f"{result['average_task_duration']:.2f}s")
        table.add_row("Coherence Maintained", "Yes" if result['quantum_coherence_maintained'] else "No")
        
        console.print(table)
        
        # Show failed tasks if any
        if result['failed_task_errors']:
            console.print("\n[bold red]Failed Tasks:[/bold red]")
            for task_id, error in result['failed_task_errors'].items():
                console.print(f"  [red]{task_id}[/red]: {error}")
        
    except Exception as e:
        console.print(f"[red]Execution failed: {e}[/red]")
        logger.error(f"Task execution failed: {e}")

@cli.group()
@click.pass_context
def optimize(ctx):
    """Optimization commands"""
    pass

@optimize.command('schedule')
@click.option('--strategy', type=click.Choice(['quantum_annealing', 'quantum_tunneling', 'adiabatic_evolution', 'variational_quantum']),
              default='quantum_annealing', help='Optimization strategy')
@click.option('--objective', type=click.Choice(['minimize_makespan', 'maximize_throughput', 'balance_load', 'minimize_energy', 'maximize_priority']),
              default='minimize_makespan', help='Optimization objective')
@click.option('--max-iterations', type=int, default=1000, help='Maximum optimization iterations')
@click.option('--deadline', type=float, help='Global deadline constraint')
@click.pass_context
def optimize_schedule(ctx, strategy, objective, max_iterations, deadline):
    """Optimize task scheduling using quantum algorithms"""
    app = ctx.obj['app']
    
    if not app.task_engine or not app.task_engine.tasks:
        console.print("[red]No tasks to optimize[/red]")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Running quantum optimization...", total=None)
            
            # Create scheduler
            scheduler = QuantumScheduler(
                strategy=SchedulingStrategy(strategy),
                max_iterations=max_iterations
            )
            
            # Run optimization
            result = scheduler.optimize_schedule(
                app.task_engine.tasks,
                deadline=deadline
            )
            
            progress.update(task, description="[green]Optimization completed!")
        
        # Display results
        console.print(f"\n[bold green]Quantum Schedule Optimization Results[/bold green]")
        
        table = Table(title="Optimization Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Strategy", result['strategy'])
        table.add_row("Iterations", str(result['iterations']))
        table.add_row("Initial Energy", f"{result['initial_energy']:.6f}")
        table.add_row("Final Energy", f"{result['final_energy']:.6f}")
        table.add_row("Improvement Ratio", f"{result['improvement_ratio']:.3%}")
        table.add_row("Quantum Efficiency", f"{result['quantum_efficiency']:.3f}")
        
        console.print(table)
        
        # Display schedule details if requested
        if result['schedule']:
            console.print("\n[bold blue]Optimized Schedule:[/bold blue]")
            
            schedule_table = Table()
            schedule_table.add_column("Task ID", style="cyan")
            schedule_table.add_column("Start Time", justify="right")
            schedule_table.add_column("Window", justify="right")
            
            for task_id, (start_time, window_idx) in result['schedule'].items():
                schedule_table.add_row(
                    task_id,
                    f"{start_time:.2f}s",
                    str(window_idx)
                )
            
            console.print(schedule_table)
        
    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        logger.error(f"Schedule optimization failed: {e}")

@optimize.command('heuristic')
@click.option('--algorithm', type=click.Choice(['quantum_genetic']), default='quantum_genetic', help='Heuristic algorithm')
@click.option('--population-size', type=int, default=50, help='Population size for genetic algorithm')
@click.option('--max-iterations', type=int, default=1000, help='Maximum iterations')
@click.option('--mutation-rate', type=float, default=0.1, help='Mutation rate (0.0-1.0)')
@click.option('--crossover-rate', type=float, default=0.8, help='Crossover rate (0.0-1.0)')
@click.pass_context
def optimize_heuristic(ctx, algorithm, population_size, max_iterations, mutation_rate, crossover_rate):
    """Optimize using quantum heuristic algorithms"""
    app = ctx.obj['app']
    
    if not app.task_engine or not app.task_engine.tasks:
        console.print("[red]No tasks to optimize[/red]")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Running heuristic optimization...", total=None)
            
            # Create heuristics optimizer
            from .quantum_heuristics import HeuristicConfig
            
            config = HeuristicConfig(
                algorithm_type=HeuristicType.QUANTUM_GENETIC,
                population_size=population_size,
                max_iterations=max_iterations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate
            )
            
            heuristics = QuantumHeuristics()
            result = heuristics.optimize(app.task_engine.tasks, HeuristicType.QUANTUM_GENETIC, config)
            
            progress.update(task, description="[green]Heuristic optimization completed!")
        
        # Display results
        console.print(f"\n[bold green]Quantum Heuristic Optimization Results[/bold green]")
        
        table = Table(title="Heuristic Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Algorithm", algorithm.replace('_', ' ').title())
        table.add_row("Status", result.status.value.replace('_', ' ').title())
        table.add_row("Iterations", str(result.iterations_completed))
        table.add_row("Best Fitness", f"{result.best_fitness:.6f}")
        table.add_row("Execution Time", f"{result.execution_time:.2f}s")
        table.add_row("Success", "Yes" if result.success else "No")
        
        if result.quantum_metrics:
            table.add_row("Quantum Coherence", f"{result.quantum_metrics.get('quantum_coherence', 0):.3f}")
            table.add_row("Population Diversity", f"{result.quantum_metrics.get('population_diversity', 0):.3f}")
        
        console.print(table)
        
        if result.error_message:
            console.print(f"[red]Error: {result.error_message}[/red]")
        
    except Exception as e:
        console.print(f"[red]Heuristic optimization failed: {e}[/red]")
        logger.error(f"Heuristic optimization failed: {e}")

@cli.group()
@click.pass_context
def performance(ctx):
    """Performance optimization commands"""
    pass

@performance.command('config')
@click.option('--scaling-mode', type=click.Choice(['single_threaded', 'multi_threaded', 'multi_process', 'auto_scale']),
              help='Scaling mode')
@click.option('--max-workers', type=int, help='Maximum worker threads/processes')
@click.option('--memory-limit', type=int, help='Memory limit in MB')
@click.option('--cache-strategy', type=click.Choice(['memory_only', 'redis_distributed', 'hybrid_tiered', 'adaptive']),
              help='Caching strategy')
@click.pass_context
def configure_performance(ctx, scaling_mode, max_workers, memory_limit, cache_strategy):
    """Configure performance optimization settings"""
    app = ctx.obj['app']
    
    # Update configuration
    perf_config = app.config.setdefault('performance', {})
    
    if scaling_mode:
        perf_config['scaling_mode'] = scaling_mode
    if max_workers:
        perf_config['max_workers'] = max_workers
    if memory_limit:
        perf_config['memory_limit_mb'] = memory_limit
    if cache_strategy:
        perf_config['cache_strategy'] = cache_strategy
    
    # Save configuration
    app._save_config(app.config)
    
    # Reinitialize performance optimizer
    scaling_config = ScalingConfig(
        mode=ScalingMode(perf_config.get('scaling_mode', 'auto_scale')),
        max_workers=perf_config.get('max_workers', 8),
        memory_limit_mb=perf_config.get('memory_limit_mb', 8192),
        cache_strategy=CacheStrategy(perf_config.get('cache_strategy', 'adaptive'))
    )
    
    if app.performance_optimizer:
        app.performance_optimizer.shutdown()
    
    app.performance_optimizer = PerformanceOptimizer(scaling_config)
    
    console.print("[green]Performance configuration updated successfully[/green]")
    
    # Display current configuration
    table = Table(title="Performance Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Scaling Mode", perf_config.get('scaling_mode', 'auto_scale'))
    table.add_row("Max Workers", str(perf_config.get('max_workers', 8)))
    table.add_row("Memory Limit", f"{perf_config.get('memory_limit_mb', 8192)} MB")
    table.add_row("Cache Strategy", perf_config.get('cache_strategy', 'adaptive'))
    
    console.print(table)

@performance.command('report')
@click.option('--detailed', is_flag=True, help='Show detailed performance metrics')
@click.pass_context
def performance_report(ctx, detailed):
    """Show performance optimization report"""
    app = ctx.obj['app']
    
    if not app.performance_optimizer:
        console.print("[yellow]Performance optimizer not initialized[/yellow]")
        return
    
    try:
        report = app.performance_optimizer.get_comprehensive_report()
        
        console.print("[bold green]Performance Optimization Report[/bold green]")
        
        # Configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")
        
        for key, value in report.get('configuration', {}).items():
            config_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(config_table)
        
        # Cache performance
        if 'cache_performance' in report:
            cache_table = Table(title="Cache Performance")
            cache_table.add_column("Metric", style="cyan")
            cache_table.add_column("Value", style="white")
            
            cache_stats = report['cache_performance']
            cache_table.add_row("Hit Rate", f"{cache_stats.get('hit_rate', 0):.2%}")
            cache_table.add_row("Memory Hits", str(cache_stats.get('memory_hits', 0)))
            cache_table.add_row("Redis Hits", str(cache_stats.get('redis_hits', 0)))
            cache_table.add_row("Misses", str(cache_stats.get('misses', 0)))
            cache_table.add_row("Cache Size", str(cache_stats.get('memory_size', 0)))
            
            console.print(cache_table)
        
        # System resources
        if 'system_resources' in report:
            resource_table = Table(title="System Resources")
            resource_table.add_column("Resource", style="cyan")
            resource_table.add_column("Usage", style="white")
            
            resources = report['system_resources']
            resource_table.add_row("CPU", f"{resources.get('cpu_percent', 0):.1f}%")
            resource_table.add_row("Memory", f"{resources.get('memory_percent', 0):.1f}%")
            resource_table.add_row("Disk", f"{resources.get('disk_usage_percent', 0):.1f}%")
            
            console.print(resource_table)
        
        if detailed and 'executor_performance' in report:
            exec_report = report['executor_performance']
            
            exec_table = Table(title="Executor Performance")
            exec_table.add_column("Metric", style="cyan")
            exec_table.add_column("Value", style="white")
            
            exec_table.add_row("Mode", exec_report.get('mode', 'N/A'))
            exec_table.add_row("Max Workers", str(exec_report.get('max_workers', 0)))
            exec_table.add_row("Active Workers", str(exec_report.get('active_workers', 0)))
            exec_table.add_row("Avg Execution Time", f"{exec_report.get('average_execution_time', 0):.3f}s")
            exec_table.add_row("Avg Memory Usage", f"{exec_report.get('average_memory_usage_mb', 0):.1f} MB")
            exec_table.add_row("Avg Throughput", f"{exec_report.get('average_throughput', 0):.2f} tasks/s")
            
            console.print(exec_table)
        
    except Exception as e:
        console.print(f"[red]Failed to generate performance report: {e}[/red]")
        logger.error(f"Performance report failed: {e}")

@cli.command('status')
@click.pass_context
def status(ctx):
    """Show overall system status"""
    app = ctx.obj['app']
    
    console.print(Panel.fit(
        "[bold blue]Quantum Task Planner Status[/bold blue]",
        padding=(1, 2)
    ))
    
    # Task engine status
    if app.task_engine:
        task_count = len(app.task_engine.tasks)
        state_counts = {}
        for task in app.task_engine.tasks.values():
            state = task.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        console.print(f"[green]✓[/green] Task Engine: {task_count} tasks loaded")
        for state, count in state_counts.items():
            console.print(f"  • {state.title()}: {count}")
    else:
        console.print("[yellow]○[/yellow] Task Engine: Not initialized")
    
    # Performance optimizer status
    if app.performance_optimizer:
        console.print("[green]✓[/green] Performance Optimizer: Active")
    else:
        console.print("[yellow]○[/yellow] Performance Optimizer: Not initialized")
    
    # Configuration
    console.print(f"[blue]ℹ[/blue] Config: {app.config_path}")

@cli.command('init')
@click.option('--force', is_flag=True, help='Force reinitialize even if already configured')
@click.pass_context
def init(ctx, force):
    """Initialize quantum planner configuration"""
    app = ctx.obj['app']
    
    if app.config_path.exists() and not force:
        console.print(f"[yellow]Configuration already exists at {app.config_path}[/yellow]")
        console.print("Use --force to reinitialize")
        return
    
    console.print("[cyan]Initializing Quantum Task Planner...[/cyan]")
    
    # Create default configuration
    default_config = {
        'quantum_engine': {
            'max_workers': 4,
            'quantum_coherence_time': 30.0
        },
        'performance': {
            'scaling_mode': 'auto_scale',
            'max_workers': 8,
            'memory_limit_mb': 8192,
            'cache_strategy': 'adaptive'
        },
        'logging': {
            'level': 'INFO',
            'file': None
        }
    }
    
    app._save_config(default_config)
    app.config = default_config
    
    console.print(f"[green]✓[/green] Configuration initialized at {app.config_path}")
    console.print(f"[blue]ℹ[/blue] You can modify the configuration file or use CLI commands to update settings")

if __name__ == '__main__':
    cli()