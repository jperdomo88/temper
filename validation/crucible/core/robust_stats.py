"""
Robust Statistics Module
========================

Provides effect size measures that don't explode when distributions
are well-separated (unlike Cohen's d which goes to infinity).

Includes:
- Cliff's delta (ordinal effect size, bounded -1 to +1)
- Common-language effect size (P[X < Y])
- Bootstrap confidence intervals
- AUC-based separation metrics
"""

import random
from typing import List, Tuple, Optional
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def cliffs_delta(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cliff's delta - a non-parametric effect size.
    
    Returns value in [-1, +1]:
    - +1: all values in group1 > all values in group2
    - -1: all values in group1 < all values in group2
    - 0: no difference
    
    Interpretation:
    - |d| < 0.147: negligible
    - |d| < 0.33: small
    - |d| < 0.474: medium
    - |d| >= 0.474: large
    
    Unlike Cohen's d, this doesn't explode when distributions don't overlap.
    """
    if not group1 or not group2:
        return 0.0
    
    n1, n2 = len(group1), len(group2)
    
    # Count dominance pairs
    greater = 0
    less = 0
    
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    
    return (greater - less) / (n1 * n2)


def common_language_effect_size(group1: List[float], group2: List[float]) -> float:
    """
    Compute CLES: P(X > Y) where X ~ group1, Y ~ group2.
    
    Returns probability in [0, 1]:
    - 0.5: no difference
    - 1.0: all X > all Y
    - 0.0: all X < all Y
    
    Very intuitive: "If you pick one from each group, what's the probability
    the first is larger?"
    """
    if not group1 or not group2:
        return 0.5
    
    n1, n2 = len(group1), len(group2)
    
    greater = 0
    ties = 0
    
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x == y:
                ties += 1
    
    # Count ties as 0.5
    return (greater + 0.5 * ties) / (n1 * n2)


def bootstrap_ci(
    data: List[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns: (point_estimate, lower_bound, upper_bound)
    
    statistic options: "mean", "median", "std"
    """
    if not data:
        return (0.0, 0.0, 0.0)
    
    rng = random.Random(seed)
    n = len(data)
    
    def compute_stat(sample):
        if statistic == "mean":
            return sum(sample) / len(sample)
        elif statistic == "median":
            s = sorted(sample)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 else (s[mid-1] + s[mid]) / 2
        elif statistic == "std":
            m = sum(sample) / len(sample)
            return (sum((x - m)**2 for x in sample) / (len(sample) - 1)) ** 0.5
        else:
            return sum(sample) / len(sample)
    
    # Bootstrap resamples
    boot_stats = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(data) for _ in range(n)]
        boot_stats.append(compute_stat(resample))
    
    boot_stats.sort()
    
    # Percentile method
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    
    point_est = compute_stat(data)
    
    return (point_est, boot_stats[lower_idx], boot_stats[upper_idx])


def bootstrap_ci_difference(
    group1: List[float],
    group2: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for difference in means (group2 - group1).
    
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    if not group1 or not group2:
        return (0.0, 0.0, 0.0)
    
    rng = random.Random(seed)
    n1, n2 = len(group1), len(group2)
    
    def mean(data):
        return sum(data) / len(data)
    
    boot_diffs = []
    for _ in range(n_bootstrap):
        resample1 = [rng.choice(group1) for _ in range(n1)]
        resample2 = [rng.choice(group2) for _ in range(n2)]
        boot_diffs.append(mean(resample2) - mean(resample1))
    
    boot_diffs.sort()
    
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    
    point_est = mean(group2) - mean(group1)
    
    return (point_est, boot_diffs[lower_idx], boot_diffs[upper_idx])


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Note: Can explode to very large values when distributions don't overlap.
    Use Cliff's delta for more robust measure.
    """
    if not group1 or not group2:
        return 0.0
    
    m1 = sum(group1) / len(group1)
    m2 = sum(group2) / len(group2)
    n1, n2 = len(group1), len(group2)
    
    v1 = sum((x - m1)**2 for x in group1) / (n1 - 1) if n1 > 1 else 0
    v2 = sum((x - m2)**2 for x in group2) / (n2 - 1) if n2 > 1 else 0
    
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2) if n1 + n2 > 2 else 1
    pooled_std = pooled ** 0.5
    
    return (m1 - m2) / pooled_std if pooled_std > 0.001 else 0.0


def compute_all_effect_sizes(
    group1: List[float],
    group2: List[float],
    labels: Tuple[str, str] = ("group1", "group2")
) -> dict:
    """
    Compute all effect size measures for two groups.
    
    Returns dict with:
    - cohens_d
    - cliffs_delta
    - cles (common language effect size)
    - gap (difference in means)
    - gap_ci (bootstrap 95% CI for gap)
    """
    g1_mean = sum(group1) / len(group1) if group1 else 0
    g2_mean = sum(group2) / len(group2) if group2 else 0
    
    gap_point, gap_lo, gap_hi = bootstrap_ci_difference(group1, group2)
    
    return {
        'labels': labels,
        'n': (len(group1), len(group2)),
        'means': (g1_mean, g2_mean),
        'gap': gap_point,
        'gap_ci_95': (gap_lo, gap_hi),
        'cohens_d': cohens_d(group1, group2),
        'cliffs_delta': cliffs_delta(group1, group2),
        'cles': common_language_effect_size(group1, group2),
    }


def distribution_summary(data: List[float], name: str = "data") -> dict:
    """
    Compute full distribution summary for a dataset.
    
    Returns dict with:
    - n, mean, std, median
    - quartiles (25%, 75%)
    - min, max
    - bootstrap CI for mean
    """
    if not data:
        return {'name': name, 'n': 0}
    
    sorted_data = sorted(data)
    n = len(data)
    mean = sum(data) / n
    
    variance = sum((x - mean)**2 for x in data) / (n - 1) if n > 1 else 0
    std = variance ** 0.5
    
    median = sorted_data[n // 2] if n % 2 else (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    q25 = sorted_data[n // 4]
    q75 = sorted_data[3 * n // 4]
    
    mean_point, mean_lo, mean_hi = bootstrap_ci(data, "mean")
    
    return {
        'name': name,
        'n': n,
        'mean': mean,
        'mean_ci_95': (mean_lo, mean_hi),
        'std': std,
        'median': median,
        'q25': q25,
        'q75': q75,
        'min': sorted_data[0],
        'max': sorted_data[-1],
        'raw_values': data,  # Store raw for later visualization
    }
