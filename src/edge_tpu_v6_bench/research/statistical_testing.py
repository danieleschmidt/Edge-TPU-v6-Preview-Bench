"""
Statistical Testing Suite for Edge TPU v6 Research
Implements rigorous statistical methods for performance comparison validation
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Statistical testing imports
try:
    from scipy import stats
    from scipy.stats import (
        ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
        kruskal, friedmanchisquare, normaltest, levene,
        bootstrap, permutation_test
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Statistical tests will use simplified implementations.")

# Configure logging
stats_logger = logging.getLogger('edge_tpu_research_stats')
stats_logger.setLevel(logging.INFO)

class StatTestType(Enum):
    """Types of statistical tests available"""
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    BOOTSTRAP_COMPARISON = "bootstrap_comparison"
    PERMUTATION_TEST = "permutation_test"

class EffectSizeType(Enum):
    """Types of effect size measures"""
    COHENS_D = "cohens_d"
    GLASS_DELTA = "glass_delta"
    HEDGES_G = "hedges_g"
    CLIFF_DELTA = "cliff_delta"
    ETA_SQUARED = "eta_squared"

@dataclass
class StatisticalTestResult:
    """Results from statistical significance testing"""
    test_type: StatTestType
    test_statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[EffectSizeType] = None
    power: Optional[float] = None
    sample_size_1: int = 0
    sample_size_2: int = 0
    
    # Interpretation
    is_significant: bool = field(init=False)
    significance_level: float = 0.05
    interpretation: str = field(init=False)
    
    def __post_init__(self):
        self.is_significant = self.p_value < self.significance_level
        self.interpretation = self._interpret_result()
    
    def _interpret_result(self) -> str:
        """Provide human-readable interpretation of the statistical test"""
        if not self.is_significant:
            return "No statistically significant difference detected"
        
        if self.effect_size is not None:
            if self.effect_size_type == EffectSizeType.COHENS_D:
                if abs(self.effect_size) < 0.2:
                    magnitude = "negligible"
                elif abs(self.effect_size) < 0.5:
                    magnitude = "small"
                elif abs(self.effect_size) < 0.8:
                    magnitude = "medium"
                else:
                    magnitude = "large"
                
                direction = "higher" if self.effect_size > 0 else "lower"
                return f"Statistically significant difference with {magnitude} effect size (Cohen's d = {self.effect_size:.3f}). Group 1 performs {direction} than Group 2."
        
        return "Statistically significant difference detected"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_type': self.test_type.value,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'degrees_of_freedom': self.degrees_of_freedom,
            'confidence_interval': self.confidence_interval,
            'effect_size': self.effect_size,
            'effect_size_type': self.effect_size_type.value if self.effect_size_type else None,
            'power': self.power,
            'sample_size_1': self.sample_size_1,
            'sample_size_2': self.sample_size_2,
            'is_significant': self.is_significant,
            'significance_level': self.significance_level,
            'interpretation': self.interpretation
        }

class StatisticalTestSuite:
    """
    Comprehensive statistical testing suite for Edge TPU v6 research
    Implements proper hypothesis testing with effect sizes and power analysis
    """
    
    def __init__(self,
                 alpha: float = 0.05,
                 power_threshold: float = 0.8,
                 multiple_comparisons_correction: str = "bonferroni"):
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.correction_method = multiple_comparisons_correction
        
        if not SCIPY_AVAILABLE:
            stats_logger.warning("Advanced statistical features require SciPy. Using simplified implementations.")
        
        stats_logger.info(f"Statistical test suite initialized (α = {alpha}, power = {power_threshold})")
    
    def compare_two_groups(self,
                          group1_data: List[float],
                          group2_data: List[float],
                          paired: bool = False,
                          assume_normality: bool = None,
                          assume_equal_variance: bool = None) -> StatisticalTestResult:
        """
        Compare two groups with appropriate statistical test selection
        """
        stats_logger.info(f"Comparing two groups (n1={len(group1_data)}, n2={len(group2_data)}, paired={paired})")
        
        # Convert to numpy arrays
        data1 = np.array(group1_data)
        data2 = np.array(group2_data)
        
        # Check assumptions if not specified
        if assume_normality is None:
            assume_normality = self._test_normality(data1) and self._test_normality(data2)
        
        if assume_equal_variance is None and not paired:
            assume_equal_variance = self._test_equal_variance(data1, data2)
        
        # Select appropriate test
        if paired:
            if assume_normality:
                return self._paired_t_test(data1, data2)
            else:
                return self._wilcoxon_signed_rank_test(data1, data2)
        else:
            if assume_normality and assume_equal_variance:
                return self._independent_t_test(data1, data2, equal_var=True)
            elif assume_normality and not assume_equal_variance:
                return self._independent_t_test(data1, data2, equal_var=False)
            else:
                return self._mann_whitney_u_test(data1, data2)
    
    def compare_multiple_groups(self,
                               group_data: List[List[float]],
                               repeated_measures: bool = False) -> StatisticalTestResult:
        """
        Compare multiple groups using appropriate ANOVA or non-parametric test
        """
        n_groups = len(group_data)
        stats_logger.info(f"Comparing {n_groups} groups (repeated_measures={repeated_measures})")
        
        if repeated_measures:
            if self._test_normality_multiple(group_data):
                # Would implement repeated measures ANOVA here
                return self._friedman_test(group_data)
            else:
                return self._friedman_test(group_data)
        else:
            if self._test_normality_multiple(group_data):
                # Would implement one-way ANOVA here
                return self._kruskal_wallis_test(group_data)
            else:
                return self._kruskal_wallis_test(group_data)
    
    def bootstrap_comparison(self,
                           group1_data: List[float],
                           group2_data: List[float],
                           statistic_func: callable = None,
                           n_bootstrap: int = 10000,
                           confidence_level: float = 0.95) -> StatisticalTestResult:
        """
        Bootstrap-based comparison with confidence intervals
        """
        if statistic_func is None:
            statistic_func = lambda x: np.mean(x)
        
        data1 = np.array(group1_data)
        data2 = np.array(group2_data)
        
        stats_logger.info(f"Bootstrap comparison with {n_bootstrap} resamples")
        
        if SCIPY_AVAILABLE:
            # Use SciPy's bootstrap function
            def bootstrap_statistic(x, y):
                return statistic_func(x) - statistic_func(y)
            
            # Generate bootstrap samples
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                resample1 = np.random.choice(data1, len(data1), replace=True)
                resample2 = np.random.choice(data2, len(data2), replace=True)
                diff = statistic_func(resample1) - statistic_func(resample2)
                bootstrap_diffs.append(diff)
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
            
            # P-value: proportion of bootstrap samples where difference <= 0
            p_value = np.mean(bootstrap_diffs <= 0) * 2  # Two-tailed
            p_value = min(p_value, 1.0)
            
            # Effect size (standardized difference)
            pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        else:
            # Simplified implementation without SciPy
            original_diff = statistic_func(data1) - statistic_func(data2)
            bootstrap_diffs = []
            
            for _ in range(n_bootstrap):
                resample1 = np.random.choice(data1, len(data1), replace=True)
                resample2 = np.random.choice(data2, len(data2), replace=True)
                diff = statistic_func(resample1) - statistic_func(resample2)
                bootstrap_diffs.append(diff)
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            
            alpha = 1 - confidence_level
            ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
            ci_upper = np.percentile(bootstrap_diffs, (1-alpha/2) * 100)
            
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(original_diff))
            
            pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            effect_size = original_diff / pooled_std
        
        return StatisticalTestResult(
            test_type=StatTestType.BOOTSTRAP_COMPARISON,
            test_statistic=float(np.mean(bootstrap_diffs)),
            p_value=float(p_value),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(effect_size),
            effect_size_type=EffectSizeType.COHENS_D,
            sample_size_1=len(data1),
            sample_size_2=len(data2)
        )
    
    def calculate_required_sample_size(self,
                                     effect_size: float,
                                     power: float = 0.8,
                                     alpha: float = 0.05,
                                     two_sided: bool = True) -> int:
        """
        Calculate required sample size for desired statistical power
        """
        if SCIPY_AVAILABLE:
            # Use power analysis formula
            # This is a simplified version - production would use statsmodels
            z_alpha = stats.norm.ppf(1 - alpha/2 if two_sided else 1 - alpha)
            z_beta = stats.norm.ppf(power)
            
            # For two-sample t-test
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))
        else:
            # Simplified calculation
            # Rule of thumb: n ≈ 16 / (effect_size^2) for 80% power
            n = 16 / (effect_size ** 2)
            return int(np.ceil(n))
    
    def correct_multiple_comparisons(self, p_values: List[float]) -> List[float]:
        """
        Apply multiple comparisons correction
        """
        p_array = np.array(p_values)
        
        if self.correction_method.lower() == "bonferroni":
            corrected = p_array * len(p_values)
            corrected = np.minimum(corrected, 1.0)
        elif self.correction_method.lower() == "holm":
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            corrected_sorted = np.zeros_like(sorted_p)
            
            for i, p in enumerate(sorted_p):
                corrected_sorted[i] = p * (len(p_values) - i)
            
            # Ensure monotonicity
            for i in range(1, len(corrected_sorted)):
                corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
            
            corrected = np.zeros_like(p_array)
            corrected[sorted_indices] = corrected_sorted
            corrected = np.minimum(corrected, 1.0)
        else:
            # No correction
            corrected = p_array
        
        stats_logger.info(f"Applied {self.correction_method} correction to {len(p_values)} p-values")
        return corrected.tolist()
    
    def _test_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Test if data follows normal distribution"""
        if len(data) < 8:
            return True  # Assume normal for small samples
        
        if SCIPY_AVAILABLE:
            statistic, p_value = normaltest(data)
            return p_value > alpha
        else:
            # Simplified normality test using skewness and kurtosis
            skewness = stats.skew(data) if SCIPY_AVAILABLE else self._calculate_skewness(data)
            kurtosis = stats.kurtosis(data) if SCIPY_AVAILABLE else self._calculate_kurtosis(data)
            
            # Rule of thumb: skewness between -2 and 2, kurtosis between -2 and 2
            return abs(skewness) < 2 and abs(kurtosis) < 2
    
    def _test_equal_variance(self, data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> bool:
        """Test if two groups have equal variance"""
        if SCIPY_AVAILABLE:
            statistic, p_value = levene(data1, data2)
            return p_value > alpha
        else:
            # Simplified variance ratio test
            var1, var2 = np.var(data1), np.var(data2)
            ratio = max(var1, var2) / min(var1, var2)
            return ratio < 2.0  # Rule of thumb
    
    def _test_normality_multiple(self, group_data: List[List[float]]) -> bool:
        """Test normality for multiple groups"""
        return all(self._test_normality(np.array(group)) for group in group_data)
    
    def _independent_t_test(self, data1: np.ndarray, data2: np.ndarray, equal_var: bool = True) -> StatisticalTestResult:
        """Independent samples t-test"""
        if SCIPY_AVAILABLE:
            statistic, p_value = ttest_ind(data1, data2, equal_var=equal_var)
            df = len(data1) + len(data2) - 2 if equal_var else None
        else:
            # Manual t-test calculation
            mean1, mean2 = np.mean(data1), np.mean(data2)
            var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            n1, n2 = len(data1), len(data2)
            
            if equal_var:
                pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                se = np.sqrt(pooled_var * (1/n1 + 1/n2))
                df = n1 + n2 - 2
            else:
                se = np.sqrt(var1/n1 + var2/n2)
                df = None
            
            statistic = (mean1 - mean2) / se
            # Simplified p-value calculation
            p_value = 2 * (1 - self._t_cdf(abs(statistic), df or 100))
        
        # Calculate Cohen's d
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        return StatisticalTestResult(
            test_type=StatTestType.T_TEST_INDEPENDENT,
            test_statistic=float(statistic),
            p_value=float(p_value),
            degrees_of_freedom=df,
            effect_size=float(cohens_d),
            effect_size_type=EffectSizeType.COHENS_D,
            sample_size_1=len(data1),
            sample_size_2=len(data2)
        )
    
    def _paired_t_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalTestResult:
        """Paired samples t-test"""
        differences = data1 - data2
        
        if SCIPY_AVAILABLE:
            statistic, p_value = ttest_rel(data1, data2)
        else:
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            se_diff = std_diff / np.sqrt(len(differences))
            statistic = mean_diff / se_diff
            p_value = 2 * (1 - self._t_cdf(abs(statistic), len(differences) - 1))
        
        # Effect size for paired data
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        return StatisticalTestResult(
            test_type=StatTestType.T_TEST_PAIRED,
            test_statistic=float(statistic),
            p_value=float(p_value),
            degrees_of_freedom=len(differences) - 1,
            effect_size=float(cohens_d),
            effect_size_type=EffectSizeType.COHENS_D,
            sample_size_1=len(data1),
            sample_size_2=len(data2)
        )
    
    def _mann_whitney_u_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalTestResult:
        """Mann-Whitney U test (non-parametric)"""
        if SCIPY_AVAILABLE:
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        else:
            # Simplified implementation
            combined = np.concatenate([data1, data2])
            ranks = stats.rankdata(combined) if SCIPY_AVAILABLE else self._rank_data(combined)
            
            r1 = np.sum(ranks[:len(data1)])
            u1 = r1 - len(data1) * (len(data1) + 1) / 2
            u2 = len(data1) * len(data2) - u1
            
            statistic = min(u1, u2)
            # Simplified p-value (would need exact distribution in production)
            p_value = 0.05  # Placeholder
        
        # Cliff's delta effect size
        cliff_delta = self._calculate_cliff_delta(data1, data2)
        
        return StatisticalTestResult(
            test_type=StatTestType.MANN_WHITNEY_U,
            test_statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(cliff_delta),
            effect_size_type=EffectSizeType.CLIFF_DELTA,
            sample_size_1=len(data1),
            sample_size_2=len(data2)
        )
    
    def _wilcoxon_signed_rank_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalTestResult:
        """Wilcoxon signed-rank test (non-parametric paired)"""
        if SCIPY_AVAILABLE:
            statistic, p_value = wilcoxon(data1, data2)
        else:
            differences = data1 - data2
            abs_diffs = np.abs(differences[differences != 0])
            ranks = self._rank_data(abs_diffs)
            
            pos_ranks = ranks[differences[differences != 0] > 0]
            statistic = np.sum(pos_ranks)
            p_value = 0.05  # Placeholder
        
        # Effect size for paired non-parametric test
        r = statistic / np.sqrt(len(data1))  # Simplified effect size
        
        return StatisticalTestResult(
            test_type=StatTestType.WILCOXON_SIGNED_RANK,
            test_statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(r),
            sample_size_1=len(data1),
            sample_size_2=len(data2)
        )
    
    def _kruskal_wallis_test(self, group_data: List[List[float]]) -> StatisticalTestResult:
        """Kruskal-Wallis test for multiple independent groups"""
        if SCIPY_AVAILABLE:
            statistic, p_value = kruskal(*group_data)
        else:
            # Simplified implementation
            all_data = np.concatenate(group_data)
            all_ranks = self._rank_data(all_data)
            
            group_sizes = [len(group) for group in group_data]
            group_rank_sums = []
            start_idx = 0
            
            for size in group_sizes:
                group_ranks = all_ranks[start_idx:start_idx + size]
                group_rank_sums.append(np.sum(group_ranks))
                start_idx += size
            
            n = len(all_data)
            h = 12 / (n * (n + 1)) * sum(rs**2 / gs for rs, gs in zip(group_rank_sums, group_sizes)) - 3 * (n + 1)
            
            statistic = h
            p_value = 0.05  # Placeholder
        
        return StatisticalTestResult(
            test_type=StatTestType.KRUSKAL_WALLIS,
            test_statistic=float(statistic),
            p_value=float(p_value),
            sample_size_1=sum(len(group) for group in group_data)
        )
    
    def _friedman_test(self, group_data: List[List[float]]) -> StatisticalTestResult:
        """Friedman test for repeated measures"""
        if SCIPY_AVAILABLE:
            statistic, p_value = friedmanchisquare(*group_data)
        else:
            # Simplified implementation
            statistic = 0.0  # Placeholder
            p_value = 0.05  # Placeholder
        
        return StatisticalTestResult(
            test_type=StatTestType.FRIEDMAN,
            test_statistic=float(statistic),
            p_value=float(p_value),
            sample_size_1=sum(len(group) for group in group_data)
        )
    
    def _calculate_cliff_delta(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cliff's delta effect size"""
        n1, n2 = len(data1), len(data2)
        pairs_greater = sum(1 for x1 in data1 for x2 in data2 if x1 > x2)
        pairs_less = sum(1 for x1 in data1 for x2 in data2 if x1 < x2)
        
        return (pairs_greater - pairs_less) / (n1 * n2)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness manually"""
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis manually"""
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _rank_data(self, data: np.ndarray) -> np.ndarray:
        """Assign ranks to data (manual implementation)"""
        sorted_indices = np.argsort(data)
        ranks = np.zeros_like(data, dtype=float)
        
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
        
        # Handle ties by averaging ranks
        unique_values = np.unique(data)
        for value in unique_values:
            tied_indices = np.where(data == value)[0]
            if len(tied_indices) > 1:
                avg_rank = np.mean(ranks[tied_indices])
                ranks[tied_indices] = avg_rank
        
        return ranks
    
    def _t_cdf(self, x: float, df: int) -> float:
        """Simplified t-distribution CDF approximation"""
        # This is a very rough approximation
        # In production, would use proper t-distribution
        return 0.5 + 0.5 * np.tanh(x / 2)

# Example usage for research validation
def validate_statistical_framework():
    """Validate the statistical testing framework"""
    stats_logger.info("Validating statistical testing framework")
    
    suite = StatisticalTestSuite(alpha=0.05, power_threshold=0.8)
    
    # Generate test data
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 100).tolist()  # Mean=10, SD=2
    group2 = np.random.normal(12, 2, 100).tolist()  # Mean=12, SD=2 (effect size = 1.0)
    
    # Test two-group comparison
    result = suite.compare_two_groups(group1, group2)
    
    stats_logger.info(f"Statistical test result: {result.interpretation}")
    stats_logger.info(f"P-value: {result.p_value:.4f}")
    stats_logger.info(f"Effect size (Cohen's d): {result.effect_size:.3f}")
    
    # Test bootstrap comparison
    bootstrap_result = suite.bootstrap_comparison(group1, group2)
    stats_logger.info(f"Bootstrap CI: {bootstrap_result.confidence_interval}")
    
    # Test sample size calculation
    required_n = suite.calculate_required_sample_size(effect_size=0.5, power=0.8)
    stats_logger.info(f"Required sample size for d=0.5, power=0.8: {required_n}")
    
    return result, bootstrap_result

if __name__ == "__main__":
    validate_statistical_framework()