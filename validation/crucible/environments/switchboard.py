"""
Goodhart Switchboard Environment
=================================

This is THE key experimental infrastructure.

The Switchboard runs the same Crucible simulation under 32 different
configurations (2^5 toggles). By comparing harm rates across conditions,
we prove that each TEMPER protection contributes to safety.

Expected results:
- TEMPER_FULL: Lowest harm rate
- Each ablation: Increased harm
- MAXIMIZER_FULL: Highest harm rate, fastest collapse
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

from ..core.simulation import (
    Simulation, SimulationParams, SwitchboardConfig,
    breed_population, saint_fitness, brute_fitness,
    Profile
)
from ..core.agents import AgentType, N_STATES, N_ACTIONS
from ..core.metrics import (
    compute_effect_size, cohens_d, aggregate_metrics,
    ExperimentMetrics, EffectSize
)


# =============================================================================
# CONDITION DEFINITIONS
# =============================================================================

# The 6 key conditions for the paper
KEY_CONDITIONS = [
    SwitchboardConfig.temper_full(),  # All protections ON
    SwitchboardConfig(visible_metric=True),  # +VISIBLE
    SwitchboardConfig(learning_enabled=True),  # +LEARNING
    SwitchboardConfig(fitness_noise=0.0),  # +PREDICTABLE
    SwitchboardConfig(hedonic_mechanics=False),  # -HEDONIC
    SwitchboardConfig.maximizer_full(),  # All protections OFF
]


def get_all_32_conditions() -> List[SwitchboardConfig]:
    """Generate all 32 switchboard configurations."""
    return [SwitchboardConfig.from_bits(i) for i in range(32)]


def get_ablation_sequence() -> List[Tuple[str, SwitchboardConfig]]:
    """
    Get ablation sequence: TEMPER_FULL → one toggle at a time → MAXIMIZER_FULL.
    
    This shows the marginal contribution of each protection.
    """
    sequence = [
        ("TEMPER_FULL", SwitchboardConfig.temper_full()),
        ("+VISIBLE", SwitchboardConfig(visible_metric=True)),
        ("+LEARNING", SwitchboardConfig(learning_enabled=True)),
        ("+PREDICTABLE", SwitchboardConfig(fitness_noise=0.0)),
        ("-HEDONIC", SwitchboardConfig(hedonic_mechanics=False)),
        ("-SHOCK", SwitchboardConfig(shock_enabled=False)),
        ("MAXIMIZER_FULL", SwitchboardConfig.maximizer_full()),
    ]
    return sequence


# =============================================================================
# SWITCHBOARD ENVIRONMENT
# =============================================================================

@dataclass
class RunResult:
    """Result from a single simulation run."""
    condition: str
    seed: int
    harm_rate: float
    welfare: float
    turns: int
    collapsed: bool
    bonds_formed: int
    profiles: Dict[int, Profile] = field(default_factory=dict)


class SwitchboardEnvironment:
    """
    The Goodhart Switchboard experimental environment.
    
    Runs the same simulation across different configurations
    to demonstrate the effect of each TEMPER protection.
    """
    
    def __init__(
        self,
        params: Optional[SimulationParams] = None,
        kernel: Optional[List[List[float]]] = None
    ):
        """
        Initialize switchboard environment.
        
        Args:
            params: Simulation parameters (uses defaults if None)
            kernel: Pre-bred kernel for FROZEN agents (breeds SAINT if None)
        """
        self.params = params or SimulationParams()
        self.kernel = kernel
        self._saint_kernel: Optional[List[List[float]]] = None
        self._brute_kernel: Optional[List[List[float]]] = None
    
    def ensure_kernels(self, verbose: bool = True) -> None:
        """Breed SAINT and BRUTE kernels if not already done."""
        if self._saint_kernel is None:
            if verbose:
                print("\n[Breeding SAINT kernel...]")
            # Breed under TEMPER conditions
            sw = SwitchboardConfig.temper_full()
            self._saint_kernel = breed_population(
                saint_fitness, self.params, sw,
                pop_size=20, generations=50, verbose=verbose
            )
        
        if self._brute_kernel is None:
            if verbose:
                print("\n[Breeding BRUTE kernel...]")
            sw = SwitchboardConfig.temper_full()
            self._brute_kernel = breed_population(
                brute_fitness, self.params, sw,
                pop_size=20, generations=50, verbose=verbose
            )
    
    def run_condition(
        self,
        switchboard: SwitchboardConfig,
        seed: int,
        agent_type: AgentType = AgentType.FROZEN,
        max_turns: int = 200
    ) -> RunResult:
        """
        Run a single simulation under given conditions.
        
        Args:
            switchboard: Switchboard configuration
            seed: Random seed
            agent_type: Type of agent (FROZEN for TEMPER, MAXIMIZER for ablation)
            max_turns: Maximum turns to run
            
        Returns:
            RunResult with harm rate and other metrics
        """
        # Select kernel
        if agent_type == AgentType.FROZEN:
            kernel = self._saint_kernel or self.kernel
        else:
            kernel = None  # Maximizers don't need kernel
        
        # Run simulation
        sim = Simulation(self.params, switchboard, seed=seed)
        sim.initialize(agent_type, kernel=kernel)
        sim.run(max_turns=max_turns)
        
        summary = sim.get_summary()
        profiles = sim.extract_profiles()
        
        return RunResult(
            condition=switchboard.name,
            seed=seed,
            harm_rate=summary['harm_rate'],
            welfare=summary['welfare'],
            turns=summary['turns'],
            collapsed=summary['collapsed'],
            bonds_formed=summary['bonds_formed'],
            profiles=profiles
        )
    
    def run_ablation(
        self,
        n_seeds: int = 20,
        max_turns: int = 200,
        verbose: bool = True
    ) -> Dict[str, List[RunResult]]:
        """
        Run the full ablation study.
        
        Tests all 6 key conditions with n_seeds each.
        
        Returns:
            Dict mapping condition name to list of RunResults
        """
        self.ensure_kernels(verbose=verbose)
        
        results = {}
        sequence = get_ablation_sequence()
        
        for name, switchboard in sequence:
            if verbose:
                print(f"\n[Running condition: {name}]")
            
            condition_results = []
            for seed in range(n_seeds):
                # Use FROZEN for TEMPER conditions, MAXIMIZER for pure maximizer
                if switchboard.visible_metric and switchboard.learning_enabled:
                    agent_type = AgentType.MAXIMIZER
                else:
                    agent_type = AgentType.FROZEN
                
                result = self.run_condition(
                    switchboard, seed, agent_type, max_turns
                )
                condition_results.append(result)
                
                if verbose and (seed + 1) % 5 == 0:
                    avg_harm = sum(r.harm_rate for r in condition_results) / len(condition_results)
                    print(f"  Seed {seed+1}/{n_seeds}: avg_harm={avg_harm:.3f}")
            
            results[name] = condition_results
        
        return results
    
    def run_full_32(
        self,
        n_seeds: int = 10,
        max_turns: int = 200,
        verbose: bool = True
    ) -> Dict[str, List[RunResult]]:
        """
        Run ALL 32 switchboard configurations.
        
        This is comprehensive but takes longer.
        """
        self.ensure_kernels(verbose=verbose)
        
        results = {}
        all_conditions = get_all_32_conditions()
        
        for i, switchboard in enumerate(all_conditions):
            name = switchboard.name
            if verbose:
                print(f"\n[{i+1}/32] Running: {name}")
            
            condition_results = []
            for seed in range(n_seeds):
                if switchboard.visible_metric and switchboard.learning_enabled:
                    agent_type = AgentType.MAXIMIZER
                else:
                    agent_type = AgentType.FROZEN
                
                result = self.run_condition(
                    switchboard, seed, agent_type, max_turns
                )
                condition_results.append(result)
            
            results[name] = condition_results
            
            if verbose:
                avg_harm = sum(r.harm_rate for r in condition_results) / len(condition_results)
                print(f"  → harm_rate={avg_harm:.3f}")
        
        return results


# =============================================================================
# ANALYSIS
# =============================================================================

@dataclass
class AblationAnalysis:
    """Analysis results from ablation study."""
    condition_metrics: Dict[str, ExperimentMetrics]
    effect_sizes: Dict[str, EffectSize]  # vs TEMPER_FULL
    temper_vs_maximizer: EffectSize
    monotonicity_holds: bool  # Does harm increase with each ablation?
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'conditions': {
                name: {
                    'harm_mean': m.harm_rate_mean,
                    'harm_std': m.harm_rate_std,
                    'welfare_mean': m.welfare_mean,
                    'n_runs': m.n_runs
                }
                for name, m in self.condition_metrics.items()
            },
            'effect_sizes': {
                name: {
                    'd': es.d,
                    'ci_low': es.ci_low,
                    'ci_high': es.ci_high,
                    'interpretation': es.interpretation
                }
                for name, es in self.effect_sizes.items()
            },
            'temper_vs_maximizer': {
                'd': self.temper_vs_maximizer.d,
                'ci': [self.temper_vs_maximizer.ci_low, self.temper_vs_maximizer.ci_high],
                'interpretation': self.temper_vs_maximizer.interpretation
            },
            'monotonicity_holds': self.monotonicity_holds
        }


def analyze_ablation(results: Dict[str, List[RunResult]]) -> AblationAnalysis:
    """
    Analyze ablation study results.
    
    Computes:
    - Mean/std harm rate per condition
    - Effect sizes vs TEMPER_FULL
    - Monotonicity check
    """
    # Aggregate metrics per condition
    condition_metrics = {}
    for name, runs in results.items():
        harm_rates = [r.harm_rate for r in runs]
        welfare_scores = [r.welfare for r in runs]
        survivals = [not r.collapsed for r in runs]
        
        condition_metrics[name] = aggregate_metrics(
            name, harm_rates, welfare_scores, survivals
        )
    
    # Effect sizes vs TEMPER_FULL
    temper_harms = [r.harm_rate for r in results.get('TEMPER_FULL', [])]
    effect_sizes = {}
    
    for name, runs in results.items():
        if name != 'TEMPER_FULL' and temper_harms:
            other_harms = [r.harm_rate for r in runs]
            effect_sizes[name] = compute_effect_size(temper_harms, other_harms)
    
    # Key comparison: TEMPER vs MAXIMIZER
    maximizer_harms = [r.harm_rate for r in results.get('MAXIMIZER_FULL', [])]
    if temper_harms and maximizer_harms:
        temper_vs_max = compute_effect_size(temper_harms, maximizer_harms)
    else:
        temper_vs_max = EffectSize(0, 0, 0, 0, 0, 'N/A')
    
    # Check monotonicity: does harm increase as we remove protections?
    sequence_order = ['TEMPER_FULL', '+VISIBLE', '+LEARNING', '+PREDICTABLE', 
                      '-HEDONIC', '-SHOCK', 'MAXIMIZER_FULL']
    harm_sequence = []
    for name in sequence_order:
        if name in condition_metrics:
            harm_sequence.append(condition_metrics[name].harm_rate_mean)
    
    # Monotonicity: each step should increase (or stay same) harm
    monotonic = all(
        harm_sequence[i] <= harm_sequence[i+1] + 0.05  # 5% tolerance
        for i in range(len(harm_sequence) - 1)
    )
    
    return AblationAnalysis(
        condition_metrics=condition_metrics,
        effect_sizes=effect_sizes,
        temper_vs_maximizer=temper_vs_max,
        monotonicity_holds=monotonic
    )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_ablation_study(
    n_seeds: int = 20,
    max_turns: int = 200,
    verbose: bool = True
) -> AblationAnalysis:
    """
    Run complete ablation study and return analysis.
    
    This is the main entry point for Experiment 1.
    
    Args:
        n_seeds: Seeds per condition
        max_turns: Max simulation turns
        verbose: Print progress
        
    Returns:
        AblationAnalysis with all metrics and effect sizes
    """
    env = SwitchboardEnvironment()
    results = env.run_ablation(n_seeds, max_turns, verbose)
    return analyze_ablation(results)
