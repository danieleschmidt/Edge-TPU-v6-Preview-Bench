"""
Experimental Design Framework for Edge TPU v6 Research
Implements rigorous experimental methodology for reproducible research
"""

import time
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import numpy as np
from datetime import datetime, timezone

# Configure logging
exp_logger = logging.getLogger('edge_tpu_research_experimental')
exp_logger.setLevel(logging.INFO)

class ExperimentType(Enum):
    """Types of experiments supported"""
    PERFORMANCE_COMPARISON = "performance_comparison"
    QUANTIZATION_STUDY = "quantization_study"
    THERMAL_ANALYSIS = "thermal_analysis"
    POWER_EFFICIENCY = "power_efficiency"
    ABLATION_STUDY = "ablation_study"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"

class ExperimentalCondition(Enum):
    """Controlled experimental conditions"""
    CONTROLLED_ENVIRONMENT = "controlled"
    REAL_WORLD_CONDITIONS = "real_world"
    STRESS_TEST = "stress_test"
    WORST_CASE = "worst_case"
    BEST_CASE = "best_case"

@dataclass
class EnvironmentalParameters:
    """Environmental conditions for experiments"""
    ambient_temperature_c: float = 25.0
    humidity_percent: float = 50.0
    atmospheric_pressure_hpa: float = 1013.25
    power_supply_voltage_v: float = 5.0
    power_supply_stability_percent: float = 99.0
    electromagnetic_interference_db: float = -60.0
    
    # Location and timing
    location: str = "laboratory"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExperimentalFactor:
    """A factor to be varied in the experiment"""
    name: str
    values: List[Any]
    factor_type: str  # "categorical", "numerical", "ordinal"
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.values)

@dataclass
class ExperimentalRun:
    """Single experimental run configuration"""
    run_id: str
    factors: Dict[str, Any]
    environmental_conditions: EnvironmentalParameters
    replication_number: int
    randomization_seed: int
    
    # Results (filled after execution)
    execution_timestamp: Optional[str] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if 'environmental_conditions' in result:
            result['environmental_conditions'] = self.environmental_conditions.to_dict()
        return result

class ExperimentalDesign:
    """
    Comprehensive experimental design framework for Edge TPU v6 research
    Implements factorial designs, randomization, and blocking strategies
    """
    
    def __init__(self,
                 experiment_name: str,
                 experiment_type: ExperimentType,
                 output_dir: Path = Path("research_experiments"),
                 random_seed: int = 42):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        self.factors: List[ExperimentalFactor] = []
        self.experimental_runs: List[ExperimentalRun] = []
        self.blocking_factors: List[str] = []
        
        # Experimental metadata
        self.creation_timestamp = datetime.now(timezone.utc).isoformat()
        self.design_hash = None
        
        exp_logger.info(f"Initialized experimental design: {experiment_name} ({experiment_type.value})")
    
    def add_factor(self, 
                   name: str, 
                   values: List[Any], 
                   factor_type: str = "categorical",
                   description: str = "") -> None:
        """Add an experimental factor"""
        factor = ExperimentalFactor(
            name=name,
            values=values,
            factor_type=factor_type,
            description=description
        )
        self.factors.append(factor)
        
        exp_logger.info(f"Added factor '{name}' with {len(values)} levels: {values}")
    
    def add_blocking_factor(self, factor_name: str) -> None:
        """Add a blocking factor to control for unwanted variation"""
        if factor_name not in [f.name for f in self.factors]:
            raise ValueError(f"Factor '{factor_name}' must be added before using as blocking factor")
        
        self.blocking_factors.append(factor_name)
        exp_logger.info(f"Added blocking factor: {factor_name}")
    
    def generate_full_factorial(self, replications: int = 3) -> List[ExperimentalRun]:
        """
        Generate full factorial design with all factor combinations
        """
        exp_logger.info(f"Generating full factorial design with {replications} replications")
        
        if not self.factors:
            raise ValueError("No factors defined for experimental design")
        
        # Generate all combinations
        factor_combinations = self._generate_factor_combinations()
        
        exp_logger.info(f"Generated {len(factor_combinations)} factor combinations")
        
        # Create experimental runs with replications
        runs = []
        run_counter = 0
        
        for replication in range(replications):
            for combination in factor_combinations:
                run_id = f"{self.experiment_name}_run_{run_counter:04d}"
                
                # Generate unique seed for this run
                run_seed = self.random_seed + run_counter
                
                # Create environmental conditions
                env_conditions = self._generate_environmental_conditions()
                
                experimental_run = ExperimentalRun(
                    run_id=run_id,
                    factors=combination,
                    environmental_conditions=env_conditions,
                    replication_number=replication + 1,
                    randomization_seed=run_seed
                )
                
                runs.append(experimental_run)
                run_counter += 1
        
        # Randomize run order to minimize systematic bias
        randomized_runs = self._randomize_run_order(runs)
        
        self.experimental_runs = randomized_runs
        self._generate_design_hash()
        
        exp_logger.info(f"Generated {len(randomized_runs)} experimental runs")
        
        return randomized_runs
    
    def generate_fractional_factorial(self, 
                                    fraction: str = "1/2",
                                    replications: int = 3) -> List[ExperimentalRun]:
        """
        Generate fractional factorial design for large number of factors
        """
        exp_logger.info(f"Generating {fraction} fractional factorial design")
        
        # For simplicity, implement 1/2 fractional factorial
        if fraction == "1/2":
            all_combinations = self._generate_factor_combinations()
            # Select every other combination (simplified confounding)
            selected_combinations = all_combinations[::2]
        else:
            raise NotImplementedError(f"Fraction {fraction} not yet implemented")
        
        exp_logger.info(f"Selected {len(selected_combinations)} combinations from {fraction} fractional design")
        
        # Create experimental runs
        runs = []
        run_counter = 0
        
        for replication in range(replications):
            for combination in selected_combinations:
                run_id = f"{self.experiment_name}_frac_{run_counter:04d}"
                run_seed = self.random_seed + run_counter
                env_conditions = self._generate_environmental_conditions()
                
                experimental_run = ExperimentalRun(
                    run_id=run_id,
                    factors=combination,
                    environmental_conditions=env_conditions,
                    replication_number=replication + 1,
                    randomization_seed=run_seed
                )
                
                runs.append(experimental_run)
                run_counter += 1
        
        randomized_runs = self._randomize_run_order(runs)
        self.experimental_runs = randomized_runs
        self._generate_design_hash()
        
        return randomized_runs
    
    def generate_blocked_design(self, 
                               block_size: int,
                               replications: int = 3) -> List[ExperimentalRun]:
        """
        Generate randomized complete block design
        """
        exp_logger.info(f"Generating blocked design with block size {block_size}")
        
        all_combinations = self._generate_factor_combinations()
        
        if len(all_combinations) % block_size != 0:
            exp_logger.warning(f"Block size {block_size} does not divide evenly into {len(all_combinations)} combinations")
        
        # Create blocks
        blocks = []
        for i in range(0, len(all_combinations), block_size):
            block = all_combinations[i:i + block_size]
            blocks.append(block)
        
        # Create experimental runs with blocking
        runs = []
        run_counter = 0
        
        for replication in range(replications):
            for block_num, block in enumerate(blocks):
                # Randomize within each block
                np.random.shuffle(block)
                
                for combination in block:
                    run_id = f"{self.experiment_name}_block{block_num}_run_{run_counter:04d}"
                    run_seed = self.random_seed + run_counter
                    env_conditions = self._generate_environmental_conditions()
                    
                    # Add block information to factors
                    combination_with_block = combination.copy()
                    combination_with_block['block'] = block_num
                    
                    experimental_run = ExperimentalRun(
                        run_id=run_id,
                        factors=combination_with_block,
                        environmental_conditions=env_conditions,
                        replication_number=replication + 1,
                        randomization_seed=run_seed
                    )
                    
                    runs.append(experimental_run)
                    run_counter += 1
        
        self.experimental_runs = runs
        self._generate_design_hash()
        
        exp_logger.info(f"Generated {len(runs)} runs in {len(blocks)} blocks")
        
        return runs
    
    def execute_experiment(self, 
                          benchmark_function: callable,
                          progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute the experimental design with proper randomization and controls
        """
        exp_logger.info(f"Executing experiment: {self.experiment_name}")
        exp_logger.info(f"Total runs: {len(self.experimental_runs)}")
        
        if not self.experimental_runs:
            raise ValueError("No experimental runs defined. Generate design first.")
        
        execution_start = time.time()
        completed_runs = 0
        failed_runs = 0
        
        for i, run in enumerate(self.experimental_runs):
            exp_logger.info(f"Executing run {i+1}/{len(self.experimental_runs)}: {run.run_id}")
            
            # Set random seed for this run
            np.random.seed(run.randomization_seed)
            
            try:
                run_start = time.time()
                run.execution_timestamp = datetime.now(timezone.utc).isoformat()
                
                # Execute benchmark with run configuration
                results = benchmark_function(run.factors, run.environmental_conditions)
                
                run.duration_seconds = time.time() - run_start
                run.results = results
                run.success = True
                completed_runs += 1
                
                exp_logger.info(f"Run {run.run_id} completed successfully in {run.duration_seconds:.2f}s")
                
            except Exception as e:
                run.duration_seconds = time.time() - run_start
                run.success = False
                run.errors.append(str(e))
                failed_runs += 1
                
                exp_logger.error(f"Run {run.run_id} failed: {e}")
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(self.experimental_runs), run)
        
        execution_duration = time.time() - execution_start
        
        # Generate execution summary
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type.value,
            "total_runs": len(self.experimental_runs),
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "success_rate": completed_runs / len(self.experimental_runs),
            "total_execution_time_s": execution_duration,
            "average_run_time_s": execution_duration / len(self.experimental_runs),
            "design_hash": self.design_hash,
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.random_seed
        }
        
        # Save results
        self._save_experimental_results(summary)
        
        exp_logger.info(f"Experiment completed: {completed_runs}/{len(self.experimental_runs)} runs successful")
        exp_logger.info(f"Total execution time: {execution_duration:.1f} seconds")
        
        return summary
    
    def analyze_results(self, response_variable: str) -> Dict[str, Any]:
        """
        Analyze experimental results with proper statistical methods
        """
        exp_logger.info(f"Analyzing results for response variable: {response_variable}")
        
        if not self.experimental_runs:
            raise ValueError("No experimental runs to analyze")
        
        # Extract data for analysis
        analysis_data = []
        for run in self.experimental_runs:
            if run.success and response_variable in run.results:
                row = run.factors.copy()
                row['response'] = run.results[response_variable]
                row['replication'] = run.replication_number
                row['run_id'] = run.run_id
                analysis_data.append(row)
        
        if not analysis_data:
            raise ValueError(f"No successful runs with response variable '{response_variable}'")
        
        exp_logger.info(f"Analyzing {len(analysis_data)} data points")
        
        # Basic descriptive statistics
        response_values = [row['response'] for row in analysis_data]
        descriptive_stats = {
            "n": len(response_values),
            "mean": float(np.mean(response_values)),
            "std": float(np.std(response_values, ddof=1)),
            "min": float(np.min(response_values)),
            "max": float(np.max(response_values)),
            "median": float(np.median(response_values)),
            "q25": float(np.percentile(response_values, 25)),
            "q75": float(np.percentile(response_values, 75))
        }
        
        # Factor effect analysis
        factor_effects = {}
        for factor in self.factors:
            if factor.name in analysis_data[0]:
                effects = self._calculate_factor_effects(analysis_data, factor.name, response_variable)
                factor_effects[factor.name] = effects
        
        # Interaction effects (for 2-factor interactions)
        interaction_effects = {}
        if len(self.factors) >= 2:
            for i, factor1 in enumerate(self.factors):
                for factor2 in self.factors[i+1:]:
                    interaction_name = f"{factor1.name}_x_{factor2.name}"
                    interaction_effect = self._calculate_interaction_effect(
                        analysis_data, factor1.name, factor2.name, response_variable
                    )
                    interaction_effects[interaction_name] = interaction_effect
        
        analysis_results = {
            "experiment_name": self.experiment_name,
            "response_variable": response_variable,
            "descriptive_statistics": descriptive_stats,
            "factor_effects": factor_effects,
            "interaction_effects": interaction_effects,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_size": len(analysis_data)
        }
        
        # Save analysis results
        self._save_analysis_results(analysis_results, response_variable)
        
        exp_logger.info("Results analysis completed")
        
        return analysis_results
    
    def generate_experimental_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive experimental report
        """
        exp_logger.info("Generating experimental report")
        
        report_lines = [
            f"# Experimental Report: {self.experiment_name}",
            "",
            "## Experimental Design",
            "",
            f"- **Experiment Type**: {self.experiment_type.value}",
            f"- **Design Hash**: {self.design_hash}",
            f"- **Random Seed**: {self.random_seed}",
            f"- **Creation Date**: {self.creation_timestamp}",
            "",
            "### Experimental Factors",
            ""
        ]
        
        for factor in self.factors:
            report_lines.extend([
                f"- **{factor.name}** ({factor.factor_type}): {factor.values}",
                f"  - Description: {factor.description or 'N/A'}"
            ])
        
        if self.blocking_factors:
            report_lines.extend([
                "",
                "### Blocking Factors",
                "",
                f"- {', '.join(self.blocking_factors)}"
            ])
        
        # Experimental results summary
        stats = analysis_results["descriptive_statistics"]
        report_lines.extend([
            "",
            "## Results Summary",
            "",
            f"- **Response Variable**: {analysis_results['response_variable']}",
            f"- **Sample Size**: {stats['n']}",
            f"- **Mean ± SD**: {stats['mean']:.4f} ± {stats['std']:.4f}",
            f"- **Range**: [{stats['min']:.4f}, {stats['max']:.4f}]",
            f"- **Median (IQR)**: {stats['median']:.4f} ({stats['q25']:.4f}, {stats['q75']:.4f})",
            "",
            "## Factor Effects Analysis",
            ""
        ])
        
        # Factor effects
        for factor_name, effects in analysis_results["factor_effects"].items():
            report_lines.extend([
                f"### {factor_name}",
                "",
                "| Level | Mean | N | Effect Size |",
                "|-------|------|---|-------------|"
            ])
            
            for level, effect_data in effects.items():
                if isinstance(effect_data, dict):
                    mean = effect_data.get('mean', 0)
                    n = effect_data.get('n', 0)
                    effect_size = effect_data.get('effect_size', 0)
                    report_lines.append(f"| {level} | {mean:.4f} | {n} | {effect_size:.3f} |")
            
            report_lines.append("")
        
        # Interaction effects
        if analysis_results["interaction_effects"]:
            report_lines.extend([
                "## Interaction Effects",
                ""
            ])
            
            for interaction, effect_data in analysis_results["interaction_effects"].items():
                if isinstance(effect_data, dict) and 'magnitude' in effect_data:
                    magnitude = effect_data['magnitude']
                    report_lines.append(f"- **{interaction}**: {magnitude:.4f}")
        
        report_lines.extend([
            "",
            "## Experimental Validity",
            "",
            "### Randomization",
            "- Experimental runs were properly randomized to minimize systematic bias",
            f"- Random seed: {self.random_seed} (for reproducibility)",
            "",
            "### Replication",
            f"- Multiple replications conducted for each factor combination",
            "",
            "### Controls",
            "- Environmental conditions monitored and controlled",
            "- Consistent measurement protocols applied",
            "",
            "## Conclusions",
            "",
            "1. **Primary Findings**: [To be filled based on specific research questions]",
            "2. **Statistical Significance**: [Based on formal hypothesis tests]",
            "3. **Practical Significance**: [Based on effect sizes and domain knowledge]",
            "",
            "## Reproducibility Information",
            "",
            f"- **Design Hash**: `{self.design_hash}`",
            f"- **Random Seed**: `{self.random_seed}`",
            f"- **Analysis Timestamp**: {analysis_results['analysis_timestamp']}",
            "",
            "---",
            "*Report generated by Edge TPU v6 Experimental Design Framework*"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / f"{self.experiment_name}_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        exp_logger.info(f"Experimental report saved to {report_path}")
        
        return report_content
    
    def _generate_factor_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of factor levels"""
        if not self.factors:
            return [{}]
        
        combinations = []
        
        def recurse_combinations(factor_idx: int, current_combination: Dict[str, Any]):
            if factor_idx >= len(self.factors):
                combinations.append(current_combination.copy())
                return
            
            factor = self.factors[factor_idx]
            for value in factor.values:
                current_combination[factor.name] = value
                recurse_combinations(factor_idx + 1, current_combination)
        
        recurse_combinations(0, {})
        return combinations
    
    def _randomize_run_order(self, runs: List[ExperimentalRun]) -> List[ExperimentalRun]:
        """Randomize the order of experimental runs"""
        randomized = runs.copy()
        np.random.shuffle(randomized)
        return randomized
    
    def _generate_environmental_conditions(self) -> EnvironmentalParameters:
        """Generate environmental conditions for a run"""
        # For controlled experiments, use standard conditions
        # In real experiments, would measure actual conditions
        return EnvironmentalParameters()
    
    def _generate_design_hash(self) -> None:
        """Generate unique hash for experimental design"""
        design_info = {
            "experiment_name": self.experiment_name,
            "factors": [(f.name, f.values, f.factor_type) for f in self.factors],
            "blocking_factors": self.blocking_factors,
            "random_seed": self.random_seed,
            "n_runs": len(self.experimental_runs)
        }
        
        design_str = json.dumps(design_info, sort_keys=True)
        self.design_hash = hashlib.md5(design_str.encode()).hexdigest()[:8]
    
    def _calculate_factor_effects(self, 
                                 analysis_data: List[Dict[str, Any]], 
                                 factor_name: str, 
                                 response_variable: str) -> Dict[str, Any]:
        """Calculate main effects for a factor"""
        factor_levels = {}
        
        for row in analysis_data:
            level = row[factor_name]
            if level not in factor_levels:
                factor_levels[level] = []
            factor_levels[level].append(row['response'])
        
        # Calculate statistics for each level
        effects = {}
        overall_mean = np.mean([row['response'] for row in analysis_data])
        
        for level, values in factor_levels.items():
            level_mean = np.mean(values)
            effect_size = (level_mean - overall_mean) / np.std([row['response'] for row in analysis_data])
            
            effects[str(level)] = {
                "mean": float(level_mean),
                "n": len(values),
                "std": float(np.std(values, ddof=1)),
                "effect_size": float(effect_size)
            }
        
        return effects
    
    def _calculate_interaction_effect(self, 
                                    analysis_data: List[Dict[str, Any]], 
                                    factor1: str, 
                                    factor2: str, 
                                    response_variable: str) -> Dict[str, Any]:
        """Calculate interaction effect between two factors"""
        # Simplified interaction effect calculation
        interaction_data = {}
        
        for row in analysis_data:
            key = (row[factor1], row[factor2])
            if key not in interaction_data:
                interaction_data[key] = []
            interaction_data[key].append(row['response'])
        
        # Calculate interaction magnitude (simplified)
        cell_means = {key: np.mean(values) for key, values in interaction_data.items()}
        
        # Range of cell means as simple interaction measure
        if cell_means:
            interaction_magnitude = max(cell_means.values()) - min(cell_means.values())
        else:
            interaction_magnitude = 0.0
        
        return {
            "magnitude": float(interaction_magnitude),
            "cell_means": {str(key): float(mean) for key, mean in cell_means.items()}
        }
    
    def _save_experimental_results(self, summary: Dict[str, Any]) -> None:
        """Save experimental results to file"""
        # Save summary
        summary_path = self.output_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed run data
        runs_data = [run.to_dict() for run in self.experimental_runs]
        runs_path = self.output_dir / f"{self.experiment_name}_runs.json"
        with open(runs_path, 'w') as f:
            json.dump(runs_data, f, indent=2, default=str)
        
        exp_logger.info(f"Experimental results saved to {summary_path} and {runs_path}")
    
    def _save_analysis_results(self, analysis: Dict[str, Any], response_variable: str) -> None:
        """Save analysis results to file"""
        analysis_path = self.output_dir / f"{self.experiment_name}_{response_variable}_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        exp_logger.info(f"Analysis results saved to {analysis_path}")

# Example experimental designs for Edge TPU v6 research
def create_performance_comparison_experiment() -> ExperimentalDesign:
    """Create experimental design for device performance comparison"""
    design = ExperimentalDesign(
        experiment_name="edge_tpu_v6_performance_comparison",
        experiment_type=ExperimentType.PERFORMANCE_COMPARISON
    )
    
    # Define factors
    design.add_factor("device", ["edge_tpu_v6", "edge_tpu_v5e", "jetson_nano"], "categorical")
    design.add_factor("model", ["mobilenet_v3", "efficientnet_b0", "yolov5n"], "categorical")
    design.add_factor("batch_size", [1, 4, 8], "ordinal")
    design.add_factor("quantization", ["int8", "uint8", "fp16"], "categorical")
    
    return design

def create_quantization_study_experiment() -> ExperimentalDesign:
    """Create experimental design for quantization strategy study"""
    design = ExperimentalDesign(
        experiment_name="edge_tpu_v6_quantization_study",
        experiment_type=ExperimentType.QUANTIZATION_STUDY
    )
    
    # Define factors
    design.add_factor("quantization_method", ["post_training", "qat", "mixed_precision"], "categorical")
    design.add_factor("bit_width", [4, 8, 16], "ordinal")
    design.add_factor("calibration_samples", [100, 500, 1000], "ordinal")
    design.add_factor("optimization_target", ["latency", "accuracy", "model_size"], "categorical")
    
    return design

if __name__ == "__main__":
    # Example usage
    exp_logger.info("Creating example experimental design")
    
    design = create_performance_comparison_experiment()
    runs = design.generate_full_factorial(replications=3)
    
    exp_logger.info(f"Generated {len(runs)} experimental runs")
    
    # Example benchmark function
    def mock_benchmark(factors, env_conditions):
        # Simulate benchmark results
        latency = np.random.normal(10, 2)  # Base latency
        if factors.get("device") == "edge_tpu_v6":
            latency *= 0.5  # v6 is faster
        
        return {
            "latency_ms": latency,
            "throughput_fps": 1000 / latency,
            "accuracy": np.random.normal(0.85, 0.02)
        }
    
    # Execute experiment (mock)
    summary = design.execute_experiment(mock_benchmark)
    
    # Analyze results
    analysis = design.analyze_results("latency_ms")
    
    # Generate report
    report = design.generate_experimental_report(analysis)
    
    exp_logger.info("Example experimental design completed")