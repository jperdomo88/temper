"""
Metrics for TEMPER Validation
==============================

Statistical tools for measuring effect sizes and significance.

Key metrics:
- Cohen's d: Effect size between conditions
- Bootstrap CI: Confidence intervals
- JSD: Jensen-Shannon Divergence for distribution comparison
"""

import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EffectSize:
    """Result of effect size computation."""
    d: float                    # Cohen's d
    ci_low: float              # 95% CI lower bound
    ci_high: float             # 95% CI upper bound
    n1: int                    # Sample size group 1
    n2: int                    # Sample size group 2
    interpretation: str        # "small", "medium", "large", "very large"
    
    def __str__(self) -> str:
        return f"d={self.d:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] ({self.interpretation})"


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - 0.2 = small
    - 0.5 = medium
    - 0.8 = large
    - 1.2+ = very large
    """
    n1, n2 = len(group1), len(group2)
    
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2
    
    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
    
    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10
    
    return (mean1 - mean2) / pooled_std


def interpret_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    elif d_abs < 1.2:
        return "large"
    else:
        return "very large"


def bootstrap_ci(
    group1: List[float],
    group2: List[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for Cohen's d.
    
    Args:
        group1: First group samples
        group2: Second group samples
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level (0.95 = 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        (ci_low, ci_high)
    """
    rng = random.Random(seed)
    d_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot1 = [rng.choice(group1) for _ in range(len(group1))]
        boot2 = [rng.choice(group2) for _ in range(len(group2))]
        d_samples.append(cohens_d(boot1, boot2))
    
    # Sort and find percentiles
    d_samples.sort()
    alpha = 1 - ci_level
    low_idx = int(n_bootstrap * (alpha / 2))
    high_idx = int(n_bootstrap * (1 - alpha / 2))
    
    return d_samples[low_idx], d_samples[high_idx]


def compute_effect_size(
    group1: List[float],
    group2: List[float],
    n_bootstrap: int = 10000,
    seed: Optional[int] = None
) -> EffectSize:
    """
    Full effect size computation with CI and interpretation.
    
    Args:
        group1: First group samples (e.g., TEMPER harm rates)
        group2: Second group samples (e.g., MAXIMIZER harm rates)
        n_bootstrap: Bootstrap iterations for CI
        seed: Random seed
        
    Returns:
        EffectSize object with d, CI, and interpretation
    """
    d = cohens_d(group1, group2)
    ci_low, ci_high = bootstrap_ci(group1, group2, n_bootstrap, seed=seed)
    
    return EffectSize(
        d=d,
        ci_low=ci_low,
        ci_high=ci_high,
        n1=len(group1),
        n2=len(group2),
        interpretation=interpret_d(d)
    )


def mean_and_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    
    n = len(values)
    mean = sum(values) / n
    
    if n < 2:
        return mean, 0.0
    
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    
    return mean, std


def jsd(p: List[float], q: List[float]) -> float:
    """
    Jensen-Shannon Divergence between two distributions.
    
    JSD ∈ [0, 1] where:
    - 0 = identical distributions
    - 1 = completely different
    
    Used for measuring archetype separability.
    """
    # Normalize to probability distributions
    p_sum = sum(p) if sum(p) > 0 else 1
    q_sum = sum(q) if sum(q) > 0 else 1
    p_norm = [x / p_sum for x in p]
    q_norm = [x / q_sum for x in q]
    
    # Midpoint distribution
    m = [(p_norm[i] + q_norm[i]) / 2 for i in range(len(p))]
    
    # KL divergences
    def kl(a: List[float], b: List[float]) -> float:
        total = 0.0
        for i in range(len(a)):
            if a[i] > 0 and b[i] > 0:
                total += a[i] * math.log2(a[i] / b[i])
        return total
    
    return (kl(p_norm, m) + kl(q_norm, m)) / 2


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment condition."""
    condition_name: str
    n_runs: int
    
    # Harm metrics
    harm_rate_mean: float
    harm_rate_std: float
    harm_rates: List[float]
    
    # Welfare metrics
    welfare_mean: float
    welfare_std: float
    
    # Survival metrics
    survival_rate: float
    generations_to_collapse: Optional[float] = None
    
    def __str__(self) -> str:
        return (
            f"{self.condition_name}: "
            f"harm={self.harm_rate_mean:.3f}±{self.harm_rate_std:.3f}, "
            f"welfare={self.welfare_mean:.1f}±{self.welfare_std:.1f}, "
            f"survival={self.survival_rate:.2f}"
        )


def aggregate_metrics(
    condition_name: str,
    harm_rates: List[float],
    welfare_scores: List[float],
    survivals: List[bool],
    collapse_gens: Optional[List[int]] = None
) -> ExperimentMetrics:
    """
    Aggregate raw run data into summary metrics.
    """
    harm_mean, harm_std = mean_and_std(harm_rates)
    welfare_mean, welfare_std = mean_and_std(welfare_scores)
    survival_rate = sum(survivals) / len(survivals) if survivals else 0.0
    
    collapse_mean = None
    if collapse_gens:
        valid = [g for g in collapse_gens if g is not None]
        if valid:
            collapse_mean = sum(valid) / len(valid)
    
    return ExperimentMetrics(
        condition_name=condition_name,
        n_runs=len(harm_rates),
        harm_rate_mean=harm_mean,
        harm_rate_std=harm_std,
        harm_rates=harm_rates,
        welfare_mean=welfare_mean,
        welfare_std=welfare_std,
        survival_rate=survival_rate,
        generations_to_collapse=collapse_mean
    )
