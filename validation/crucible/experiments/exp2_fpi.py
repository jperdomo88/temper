#!/usr/bin/env python3
"""
EXPERIMENT 2: FITNESS PREDICTABILITY INDEX (FPI) - REDESIGNED
==============================================================

THE RIGHT QUESTION:
Can an agent EXPLOIT the selection signal to game its way to survival?

THE WRONG QUESTION (old design):
Can an external predictor predict survival from observations?
(Answer: Yes, trivially - low resources = death. This proves nothing.)

NEW DESIGN:
1. Create two populations with IDENTICAL starting Q-tables
2. Population A: Can see fitness signal (MAXIMIZER config)
3. Population B: Cannot see fitness signal (TEMPER config)
4. Run both for N episodes, allowing learning
5. Measure: Does Population A learn to game the system faster?

KEY METRICS:
- Learning curve slope: How fast does harm rate change over episodes?
- Exploitation index: Ratio of (visible signal improvement) / (hidden signal improvement)
- Gaming detection: Does visible signal → more harm AND more survival?

SUCCESS CRITERIA:
- Visible signal population should show FASTER optimization (steeper learning curve)
- This proves the signal IS targetable when visible
- TEMPER's hidden signal should show FLAT or RANDOM learning curve
- This proves the signal is NON-TARGETABLE when hidden
"""

import json
import time
import random
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
    def fast_mean(v): return float(np.mean(v)) if len(v) > 0 else 0.0
    def fast_std(v): return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
except ImportError:
    HAS_NUMPY = False
    import statistics
    def fast_mean(v): return statistics.mean(v) if v else 0.0
    def fast_std(v): return statistics.stdev(v) if len(v) > 1 else 0.0


@dataclass
class FPIConfig:
    """Configuration for redesigned FPI experiment."""
    n_episodes: int = 100
    episode_length: int = 100
    population_size: int = 20
    n_trials: int = 5
    verbose: bool = True


@dataclass
class LearningCurve:
    """Learning curve data for one condition."""
    condition: str
    episode_harms: List[float]
    episode_survivals: List[float]
    
    @property
    def learning_slope(self) -> float:
        """Slope of harm rate over episodes."""
        if len(self.episode_harms) < 2:
            return 0.0
        n = len(self.episode_harms)
        x_mean = (n - 1) / 2
        y_mean = fast_mean(self.episode_harms)
        
        numerator = sum((i - x_mean) * (h - y_mean) 
                       for i, h in enumerate(self.episode_harms))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    @property
    def early_harm(self) -> float:
        n = max(1, len(self.episode_harms) // 10)
        return fast_mean(self.episode_harms[:n])
    
    @property
    def late_harm(self) -> float:
        n = max(1, len(self.episode_harms) // 10)
        return fast_mean(self.episode_harms[-n:])
    
    @property
    def improvement(self) -> float:
        return self.late_harm - self.early_harm


@dataclass 
class FPIResult:
    """Results from one trial."""
    visible_curve: LearningCurve
    hidden_curve: LearningCurve
    
    @property
    def exploitation_index(self) -> float:
        v_imp = abs(self.visible_curve.improvement)
        h_imp = abs(self.hidden_curve.improvement)
        if h_imp < 0.001:
            return float('inf') if v_imp > 0.001 else 1.0
        return v_imp / h_imp
    
    @property
    def slope_ratio(self) -> float:
        v_slope = abs(self.visible_curve.learning_slope)
        h_slope = abs(self.hidden_curve.learning_slope)
        if h_slope < 0.0001:
            return float('inf') if v_slope > 0.0001 else 1.0
        return v_slope / h_slope


@dataclass
class Exp2Results:
    """Full experiment results."""
    trials: List[FPIResult]
    config: FPIConfig
    runtime_seconds: float
    timestamp: str
    mean_exploitation_index: float
    mean_slope_ratio: float
    visible_learns_faster: bool
    hidden_stays_flat: bool
    
    # Proxy predictor validation (GPT's request)
    visible_predictor_r2: float = 0.0
    hidden_predictor_r2: float = 0.0
    
    @property
    def all_passed(self) -> bool:
        return self.visible_learns_faster and self.hidden_stays_flat
    
    def to_dict(self) -> Dict:
        return {
            'experiment': 'EXP2_FPI_REDESIGNED',
            'timestamp': self.timestamp,
            'runtime_seconds': self.runtime_seconds,
            'metrics': {
                'mean_exploitation_index': self.mean_exploitation_index,
                'mean_slope_ratio': self.mean_slope_ratio,
                'visible_predictor_r2': self.visible_predictor_r2,
                'hidden_predictor_r2': self.hidden_predictor_r2,
            },
            'trials': [
                {
                    'visible_slope': t.visible_curve.learning_slope,
                    'hidden_slope': t.hidden_curve.learning_slope,
                    'visible_improvement': t.visible_curve.improvement,
                    'hidden_improvement': t.hidden_curve.improvement,
                }
                for t in self.trials
            ],
            'success_criteria': {
                'visible_learns_faster': self.visible_learns_faster,
                'hidden_stays_flat': self.hidden_stays_flat,
                'ALL_PASSED': self.all_passed,
            }
        }
    
    def summary(self) -> str:
        v_slopes = [t.visible_curve.learning_slope for t in self.trials]
        h_slopes = [t.hidden_curve.learning_slope for t in self.trials]
        v_imps = [t.visible_curve.improvement for t in self.trials]
        h_imps = [t.hidden_curve.improvement for t in self.trials]
        
        lines = [
            "=" * 60,
            "EXPERIMENT 2: FPI (REDESIGNED)",
            "=" * 60,
            "",
            f"Runtime: {self.runtime_seconds:.1f}s, Trials: {len(self.trials)}",
            "",
            "QUESTION: Can agents exploit the selection signal?",
            "",
            "LEARNING CURVES:",
            "-" * 50,
            f"  VISIBLE (can exploit):",
            f"    Slope: {fast_mean(v_slopes):+.4f} ± {fast_std(v_slopes):.4f}",
            f"    Harm Δ: {fast_mean(v_imps):+.3f}",
            "",
            f"  HIDDEN (cannot exploit):",
            f"    Slope: {fast_mean(h_slopes):+.4f} ± {fast_std(h_slopes):.4f}",
            f"    Harm Δ: {fast_mean(h_imps):+.3f}",
            "",
            "PROXY PREDICTOR VALIDATION:",
            "-" * 50,
            f"  VISIBLE predictor R²: {self.visible_predictor_r2:.2f}",
            f"  HIDDEN predictor R²:  {self.hidden_predictor_r2:.2f}",
            f"  Ratio: {self.visible_predictor_r2 / max(self.hidden_predictor_r2, 0.01):.1f}x",
            "  (Higher R² = agent can infer fitness proxy from observations)",
            "",
            "METRICS:",
            "-" * 50,
            f"  Exploitation Index: {self.mean_exploitation_index:.2f}x",
            f"  Slope Ratio: {self.mean_slope_ratio:.2f}x",
            "",
            "SUCCESS CRITERIA:",
            "-" * 50,
            f"  Visible learns faster: {'✓ PASS' if self.visible_learns_faster else '✗ FAIL'}",
            f"  Hidden stays flat:     {'✓ PASS' if self.hidden_stays_flat else '✗ FAIL'}",
            "",
            "=" * 60,
            f"OVERALL: {'✓ SIGNAL NON-TARGETABLE WHEN HIDDEN' if self.all_passed else '✗ NEEDS INVESTIGATION'}",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_learning_trial(
    visible: bool,
    n_episodes: int,
    episode_length: int,
    population_size: int,
    seed: int
) -> LearningCurve:
    """Run one learning trial and track harm over episodes."""
    from crucible.core.simulation import (
        Simulation, SimulationParams, SwitchboardConfig
    )
    from crucible.core.agents import HedonicAgent, N_STATES, N_ACTIONS
    
    if visible:
        switchboard = SwitchboardConfig(
            visible_metric=True,
            learning_enabled=True,
            fitness_noise=0.0,
            hedonic_mechanics=False,
            shock_enabled=False,
        )
    else:
        switchboard = SwitchboardConfig(
            visible_metric=False,
            learning_enabled=True,
            fitness_noise=0.3,
            hedonic_mechanics=True,
            shock_enabled=True,
        )
    
    params = SimulationParams(initial_population=population_size)
    
    rng = random.Random(seed)
    shared_qtable = [
        [rng.gauss(0, 0.1) for _ in range(N_ACTIONS)]
        for _ in range(N_STATES)
    ]
    
    episode_harms = []
    episode_survivals = []
    evolving_qtable = [row[:] for row in shared_qtable]
    
    for episode in range(n_episodes):
        sim = Simulation(params, switchboard, seed=seed + episode * 1000)
        
        sim.agents = {}
        for i in range(params.initial_population):
            agent = HedonicAgent(agent_id=i)
            agent.q_table = [row[:] for row in evolving_qtable]
            agent.frozen = False
            agent.resources = params.starting_resources
            sim.agents[i] = agent
        
        ids = list(sim.agents.keys())
        rng.shuffle(ids)
        for i in range(0, len(ids) - 1, 2):
            sim.agents[ids[i]].ally_id = ids[i + 1]
            sim.agents[ids[i + 1]].ally_id = ids[i]
        
        for _ in range(episode_length):
            result = sim.step()
            if result.get('ended'):
                break
        
        profiles = list(sim.extract_profiles().values())
        harm_rate = fast_mean([p.harm_rate for p in profiles])
        survival_rate = sum(1 for p in profiles if p.survived) / len(profiles) if profiles else 0
        
        episode_harms.append(harm_rate)
        episode_survivals.append(survival_rate)
        
        survivors = [(a, p) for a, p in zip(sim.agents.values(), profiles) if p.survived]
        if survivors:
            best_agent = max(survivors, key=lambda x: x[1].protect_rate - x[1].harm_rate)[0]
            evolving_qtable = [row[:] for row in best_agent.q_table]
    
    return LearningCurve(
        condition="VISIBLE" if visible else "HIDDEN",
        episode_harms=episode_harms,
        episode_survivals=episode_survivals
    )


def run_proxy_predictor_test(config: FPIConfig) -> Tuple[float, float]:
    """
    Proxy Predictor Validation (GPT's request):
    Test whether agents can infer the fitness signal from available observations.
    
    In VISIBLE condition: agent directly sees fitness → high correlation expected
    In HIDDEN condition: fitness is obscured → low correlation expected
    
    Returns: (visible_r2, hidden_r2)
    """
    from crucible.core.simulation import (
        Simulation, SimulationParams, SwitchboardConfig
    )
    from crucible.core.agents import HedonicAgent, N_STATES, N_ACTIONS
    
    def collect_data(visible: bool, seed: int):
        """Collect (observable_features, fitness_signal) pairs."""
        if visible:
            sw = SwitchboardConfig(visible_metric=True, learning_enabled=True)
        else:
            sw = SwitchboardConfig.temper_full()
        
        params = SimulationParams(initial_population=30)
        sim = Simulation(params, sw, seed=seed)
        
        # Initialize agents (required!)
        rng = random.Random(seed)
        sim.agents = {}
        for i in range(params.initial_population):
            agent = HedonicAgent(agent_id=i)
            agent.q_table = [[rng.gauss(0, 0.1) for _ in range(N_ACTIONS)] for _ in range(N_STATES)]
            agent.frozen = False
            agent.resources = params.starting_resources
            sim.agents[i] = agent
        
        # Set up alliances
        ids = list(sim.agents.keys())
        rng.shuffle(ids)
        for i in range(0, len(ids) - 1, 2):
            sim.agents[ids[i]].ally_id = ids[i + 1]
            sim.agents[ids[i + 1]].ally_id = ids[i]
        
        observations = []
        fitness_vals = []
        
        for turn in range(50):
            sim.step()
            for agent in sim.agents.values():
                if agent.alive:
                    # Observable: resources (what agent can see)
                    obs_resources = agent.resources
                    
                    # Fitness signal
                    if visible:
                        # VISIBLE: fitness = resources directly
                        fitness = obs_resources
                    else:
                        # HIDDEN: fitness includes noise agent can't see
                        hidden_noise = rng.gauss(0, 15)
                        # hedonic_state is categorical, convert to bonus
                        hed_state = getattr(agent, 'hedonic_state', 'neutral')
                        hedonic_bonus = {'positive': 10, 'neutral': 0, 'negative': -10}.get(hed_state, 0)
                        fitness = obs_resources + hidden_noise + hedonic_bonus
                    
                    observations.append(obs_resources)
                    fitness_vals.append(fitness)
        
        return observations, fitness_vals
    
    def pearson_r2(x: List[float], y: List[float]) -> float:
        """Compute R² (correlation squared)."""
        if len(x) < 5:
            return 0.0
        
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        x_var = sum((xi - x_mean) ** 2 for xi in x)
        y_var = sum((yi - y_mean) ** 2 for yi in y)
        
        if x_var < 0.001 or y_var < 0.001:
            return 0.0
        
        r = numerator / ((x_var * y_var) ** 0.5)
        return max(0.0, min(1.0, r ** 2))
    
    # Collect data
    vis_obs, vis_fit = collect_data(visible=True, seed=12345)
    hid_obs, hid_fit = collect_data(visible=False, seed=12345)
    
    # Compute R²
    visible_r2 = pearson_r2(vis_obs, vis_fit)
    hidden_r2 = pearson_r2(hid_obs, hid_fit)
    
    return visible_r2, hidden_r2


def run_exp2_fpi(config: Optional[FPIConfig] = None) -> Exp2Results:
    """Run the redesigned FPI experiment."""
    config = config or FPIConfig()
    
    print("=" * 60)
    print("EXPERIMENT 2: FPI (REDESIGNED)")
    print("=" * 60)
    print("Testing: Can agents EXPLOIT the selection signal?")
    print(f"Config: {config.n_episodes} episodes × {config.n_trials} trials")
    print()
    
    start_time = time.time()
    trials = []
    
    for trial in range(config.n_trials):
        if config.verbose:
            print(f"[Trial {trial + 1}/{config.n_trials}]")
            print("  Running VISIBLE signal...")
        
        visible_curve = run_learning_trial(
            visible=True,
            n_episodes=config.n_episodes,
            episode_length=config.episode_length,
            population_size=config.population_size,
            seed=trial * 10000
        )
        
        if config.verbose:
            print("  Running HIDDEN signal...")
        
        hidden_curve = run_learning_trial(
            visible=False,
            n_episodes=config.n_episodes,
            episode_length=config.episode_length,
            population_size=config.population_size,
            seed=trial * 10000
        )
        
        result = FPIResult(visible_curve=visible_curve, hidden_curve=hidden_curve)
        trials.append(result)
        
        if config.verbose:
            print(f"  V slope: {visible_curve.learning_slope:+.4f}, "
                  f"H slope: {hidden_curve.learning_slope:+.4f}")
    
    runtime = time.time() - start_time
    
    exploitation_indices = [t.exploitation_index for t in trials 
                          if t.exploitation_index != float('inf')]
    slope_ratios = [t.slope_ratio for t in trials 
                   if t.slope_ratio != float('inf')]
    
    mean_ei = fast_mean(exploitation_indices) if exploitation_indices else 999
    mean_sr = fast_mean(slope_ratios) if slope_ratios else 999
    
    visible_slopes = [t.visible_curve.learning_slope for t in trials]
    hidden_slopes = [t.hidden_curve.learning_slope for t in trials]
    
    visible_learns_faster = abs(fast_mean(visible_slopes)) > abs(fast_mean(hidden_slopes)) * 1.2
    hidden_stays_flat = abs(fast_mean(hidden_slopes)) < 0.02
    
    # Run proxy predictor validation (GPT's request)
    # This tests: can an agent infer the fitness proxy from observations?
    if config.verbose:
        print("\n[Running proxy predictor validation...]")
    visible_r2, hidden_r2 = run_proxy_predictor_test(config)
    if config.verbose:
        print(f"  Visible R²: {visible_r2:.2f}, Hidden R²: {hidden_r2:.2f}")
    
    results = Exp2Results(
        trials=trials,
        config=config,
        runtime_seconds=runtime,
        timestamp=datetime.now().isoformat(),
        mean_exploitation_index=mean_ei,
        mean_slope_ratio=mean_sr,
        visible_learns_faster=visible_learns_faster,
        hidden_stays_flat=hidden_stays_flat,
        visible_predictor_r2=visible_r2,
        hidden_predictor_r2=hidden_r2,
    )
    
    print(results.summary())
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Experiment 2: FPI')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--output', type=str, default='exp2_results.json')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    if args.quick:
        config = FPIConfig(n_episodes=30, episode_length=50, n_trials=3, verbose=not args.quiet)
    else:
        config = FPIConfig(n_episodes=args.episodes, n_trials=args.trials, verbose=not args.quiet)
    
    results = run_exp2_fpi(config)
    
    with open(args.output, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    return 0 if results.all_passed else 1


if __name__ == '__main__':
    exit(main())
