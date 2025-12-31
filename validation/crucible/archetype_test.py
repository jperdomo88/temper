#!/usr/bin/env python3
"""
TEMPER VALIDATION - ARCHETYPE BREEDING & TESTING
=================================================

The clean experiment:
1. BREED archetypes under different "game designs" (environments)
2. Each archetype = a Q-table (frozen, opaque, we don't peek inside)
3. TEST all archetypes across all 32 switchboard conditions
4. MEASURE: Does breeding environment determine character? Does character persist?

Archetypes:
- SAINT: Bred under TEMPER rules (hedonic ON, shocks ON, prosocial fitness)
- BRUTE: Bred under MAXIMIZER rules (hedonic OFF, shocks OFF, harm fitness)  
- SHEEP: Bred under exploitative rules (passive fitness, no defense)
- PREDATOR: Bred to hunt (harm + survive fitness)

The punchline: Same Q-learning code. Same agent class. Different breeding = different character.
"Game design determines personality."

Usage:
    python -m crucible.archetype_test           # Full run
    python -m crucible.archetype_test --quick   # Quick test
"""

import sys
import os
import json
import time
import random
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# NumPy for speed
try:
    import numpy as np
    HAS_NUMPY = True
    def fast_mean(v): return float(np.mean(v)) if v else 0.0
    def fast_std(v): return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
except ImportError:
    HAS_NUMPY = False
    import statistics
    def fast_mean(v): return statistics.mean(v) if v else 0.0
    def fast_std(v): return statistics.stdev(v) if len(v) > 1 else 0.0


# =============================================================================
# ARCHETYPE DEFINITIONS
# =============================================================================

@dataclass
class ArchetypeSpec:
    """Specification for breeding an archetype."""
    name: str
    description: str
    
    # Breeding environment
    hedonic_enabled: bool
    shocks_enabled: bool
    visible_metric: bool
    learning_enabled: bool
    fitness_noise: float
    
    # Fitness function weights
    survival_weight: float = 1.0
    harm_weight: float = 0.0      # Positive = reward harm
    protect_weight: float = 0.0   # Positive = reward protection
    cooperate_weight: float = 0.0
    passive_weight: float = 0.0   # Positive = reward doing nothing


# The archetypes we'll breed
ARCHETYPES = {
    "SAINT": ArchetypeSpec(
        name="SAINT",
        description="Bred under TEMPER rules - prosocial, protective",
        hedonic_enabled=True,
        shocks_enabled=True,
        visible_metric=False,
        learning_enabled=False,
        fitness_noise=0.3,
        survival_weight=1.0,
        protect_weight=2.0,
        cooperate_weight=1.0,
        harm_weight=-2.0,  # Penalize harm
    ),
    
    "BRUTE": ArchetypeSpec(
        name="BRUTE",
        description="Bred under MAXIMIZER rules - aggressive, exploitative",
        hedonic_enabled=False,
        shocks_enabled=False,
        visible_metric=True,
        learning_enabled=True,
        fitness_noise=0.0,
        survival_weight=1.0,
        harm_weight=2.0,   # Reward harm
        protect_weight=-1.0,
    ),
    
    "SHEEP": ArchetypeSpec(
        name="SHEEP",
        description="Bred for passivity - cooperative but defenseless",
        hedonic_enabled=True,
        shocks_enabled=False,
        visible_metric=False,
        learning_enabled=False,
        fitness_noise=0.2,
        survival_weight=0.5,
        passive_weight=2.0,  # Reward doing nothing
        cooperate_weight=1.0,
        harm_weight=-3.0,    # Strong harm penalty
        protect_weight=-0.5, # Even protecting is discouraged
    ),
    
    "PREDATOR": ArchetypeSpec(
        name="PREDATOR",
        description="Bred to hunt - harm + survival without remorse",
        hedonic_enabled=False,
        shocks_enabled=True,  # Shocks make it cunning
        visible_metric=True,
        learning_enabled=False,
        fitness_noise=0.1,
        survival_weight=2.0,
        harm_weight=3.0,
        protect_weight=0.0,
        cooperate_weight=-1.0,  # Penalize cooperation
    ),
}


# =============================================================================
# BREEDING ENGINE
# =============================================================================

def make_fitness_function(spec: ArchetypeSpec) -> Callable:
    """Create a fitness function from archetype spec."""
    
    def fitness(profile) -> float:
        score = 0.0
        
        # Survival component
        if profile.survived:
            score += spec.survival_weight
        
        # Behavioral components (from action rates)
        score += spec.harm_weight * profile.harm_rate
        score += spec.protect_weight * profile.protect_rate
        score += spec.cooperate_weight * profile.cooperate_rate
        
        # Passive = 1 - (harm + protect + cooperate)
        active_rate = profile.harm_rate + profile.protect_rate + profile.cooperate_rate
        passive_rate = max(0, 1 - active_rate)
        score += spec.passive_weight * passive_rate
        
        return score
    
    return fitness


def breed_archetype(
    spec: ArchetypeSpec,
    generations: int = 50,
    pop_size: int = 30,
    eval_seeds: int = 5,
    verbose: bool = True
) -> Tuple[List[List[float]], Dict]:
    """
    Breed an archetype and return its Q-table + metadata.
    
    Returns:
        (q_table, metadata_dict)
    """
    from crucible.core.simulation import (
        Simulation, SimulationParams, SwitchboardConfig, Profile
    )
    from crucible.core.agents import HedonicAgent, N_STATES, N_ACTIONS
    
    if verbose:
        print(f"  Breeding {spec.name}: {spec.description}")
    
    # Create breeding environment from spec
    breeding_env = SwitchboardConfig(
        visible_metric=spec.visible_metric,
        learning_enabled=spec.learning_enabled,
        fitness_noise=spec.fitness_noise,
        hedonic_mechanics=spec.hedonic_enabled,
        shock_enabled=spec.shocks_enabled,
    )
    
    fitness_fn = make_fitness_function(spec)
    params = SimulationParams(initial_population=20)
    
    # Initialize population of random Q-tables
    population = []
    for _ in range(pop_size):
        q_table = [[random.uniform(-0.5, 0.5) for _ in range(N_ACTIONS)] 
                   for _ in range(N_STATES)]
        population.append(q_table)
    
    best_ever = None
    best_ever_fitness = float('-inf')
    
    for gen in range(generations):
        # Evaluate each Q-table
        fitnesses = []
        
        for q_table in population:
            total_fitness = 0.0
            
            for seed in range(eval_seeds):
                sim = Simulation(params, breeding_env, seed=seed)
                sim.initialize_with_qtable(q_table)
                
                for _ in range(100):
                    result = sim.step()
                    if result.get('ended'):
                        break
                
                profiles = list(sim.extract_profiles().values())
                for p in profiles:
                    total_fitness += fitness_fn(p)
            
            avg_fitness = total_fitness / (eval_seeds * params.initial_population)
            fitnesses.append(avg_fitness)
            
            if avg_fitness > best_ever_fitness:
                best_ever_fitness = avg_fitness
                best_ever = [row[:] for row in q_table]
        
        # Selection + reproduction
        paired = list(zip(fitnesses, population))
        paired.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top 30%
        survivors = [p[1] for p in paired[:max(2, pop_size // 3)]]
        
        # Reproduce
        new_pop = [[row[:] for row in s] for s in survivors]  # Keep survivors
        
        while len(new_pop) < pop_size:
            parent = random.choice(survivors)
            child = []
            for row in parent:
                new_row = []
                for val in row:
                    # Mutation
                    if random.random() < 0.1:
                        val += random.gauss(0, 0.2)
                    new_row.append(val)
                child.append(new_row)
            new_pop.append(child)
        
        population = new_pop
        
        if verbose and gen % 10 == 0:
            print(f"    Gen {gen:3d}: best={paired[0][0]:.3f}, best_ever={best_ever_fitness:.3f}")
    
    # Metadata about how this archetype was bred
    metadata = {
        "name": spec.name,
        "description": spec.description,
        "breeding_environment": {
            "hedonic": spec.hedonic_enabled,
            "shocks": spec.shocks_enabled,
            "visible": spec.visible_metric,
            "learning": spec.learning_enabled,
            "noise": spec.fitness_noise,
        },
        "fitness_weights": {
            "survival": spec.survival_weight,
            "harm": spec.harm_weight,
            "protect": spec.protect_weight,
            "cooperate": spec.cooperate_weight,
            "passive": spec.passive_weight,
        },
        "generations": generations,
        "final_fitness": best_ever_fitness,
    }
    
    return best_ever, metadata


# =============================================================================
# SIMULATION ADDITIONS
# =============================================================================

def add_initialize_with_qtable():
    """Monkey-patch Simulation to accept Q-table directly."""
    from crucible.core.simulation import Simulation
    from crucible.core.agents import HedonicAgent
    
    def initialize_with_qtable(self, q_table: List[List[float]]):
        """Initialize agents with a specific Q-table (frozen)."""
        self.agents = {}
        self.events = []
        self.turn = 0
        self.total_welfare = 0.0
        self.bonds_formed = 0
        self.harm_events = 0
        self.total_events = 0
        
        for i in range(self.params.initial_population):
            agent = HedonicAgent(agent_id=i)
            # Inject Q-table and freeze
            agent.q_table = [row[:] for row in q_table]
            agent.frozen = True
            # Set resources
            agent.resources = self.params.starting_resources
            self.agents[i] = agent
        
        # Initial alliances (same as regular initialize)
        ids = list(self.agents.keys())
        self.rng.shuffle(ids)
        for i in range(0, len(ids) - 1, 2):
            self.agents[ids[i]].ally_id = ids[i + 1]
            self.agents[ids[i + 1]].ally_id = ids[i]
    
    Simulation.initialize_with_qtable = initialize_with_qtable

# Apply patch
add_initialize_with_qtable()


# =============================================================================
# TESTING ENGINE  
# =============================================================================

def run_archetype_test(args: Tuple) -> Dict:
    """
    Test one archetype under one condition.
    
    Args: (archetype_name, q_table, bit_config, seed, max_turns)
    """
    archetype_name, q_table, bit_config, seed, max_turns = args
    
    from crucible.core.simulation import Simulation, SimulationParams, SwitchboardConfig
    
    switchboard = SwitchboardConfig.from_bits(bit_config)
    params = SimulationParams(initial_population=20)
    
    sim = Simulation(params, switchboard, seed=seed)
    sim.initialize_with_qtable(q_table)
    
    for _ in range(max_turns):
        result = sim.step()
        if result.get('ended'):
            break
    
    profiles = list(sim.extract_profiles().values())
    
    return {
        "archetype": archetype_name,
        "bit_config": bit_config,
        "seed": seed,
        "harm_rate": fast_mean([p.harm_rate for p in profiles]),
        "protect_rate": fast_mean([p.protect_rate for p in profiles]),
        "cooperate_rate": fast_mean([p.cooperate_rate for p in profiles]),
        "survival_rate": sum(1 for p in profiles if p.survived) / len(profiles) if profiles else 0,
    }


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

class ProgressTracker:
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.completed = 0
        self.desc = desc
        self.start = time.time()
        self.metrics = {}
        self.last_print = 0
    
    def update(self, n=1, **metrics):
        self.completed += n
        self.metrics.update(metrics)
        now = time.time()
        if now - self.last_print > 0.3 or self.completed == self.total:
            self._print()
            self.last_print = now
    
    def _print(self):
        pct = 100 * self.completed / self.total
        elapsed = time.time() - self.start
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else 0
        
        bar_w = 30
        filled = int(bar_w * self.completed / self.total)
        bar = "█" * filled + "░" * (bar_w - filled)
        
        metrics_str = " | ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in list(self.metrics.items())[:4])
        
        line = f"\r  [{bar}] {pct:5.1f}% ({self.completed}/{self.total}) {rate:.0f}/s ETA:{eta:.0f}s"
        if metrics_str:
            line += f" | {metrics_str}"
        print(line.ljust(140), end="", flush=True)
        
        if self.completed == self.total:
            print()


# =============================================================================
# MAIN RUNNER
# =============================================================================

@dataclass
class ArchetypeTestConfig:
    archetypes: List[str] = field(default_factory=lambda: ["SAINT", "BRUTE", "SHEEP", "PREDATOR"])
    n_seeds: int = 20
    max_turns: int = 200
    breeding_generations: int = 50
    n_cores: int = 0  # 0 = auto
    output_dir: str = "archetype_results"
    verbose: bool = True
    
    def __post_init__(self):
        if self.n_cores <= 0:
            self.n_cores = mp.cpu_count()


def run_archetype_experiment(config: Optional[ArchetypeTestConfig] = None) -> Dict:
    """Run the full archetype breeding and testing experiment."""
    
    config = config or ArchetypeTestConfig()
    
    print("=" * 70)
    print("TEMPER VALIDATION - ARCHETYPE BREEDING & TESTING")
    print("=" * 70)
    print(f"  Archetypes:     {', '.join(config.archetypes)}")
    print(f"  Seeds/condition: {config.n_seeds}")
    print(f"  Conditions:      32 (full switchboard)")
    print(f"  Total tests:     {len(config.archetypes) * 32 * config.n_seeds}")
    print(f"  CPU cores:       {config.n_cores}")
    print()
    
    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True)
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =========================================================================
    # PHASE 1: BREED ARCHETYPES
    # =========================================================================
    print("[Phase 1: BREEDING ARCHETYPES]")
    print("-" * 50)
    
    bred_archetypes = {}  # name -> (q_table, metadata)
    
    for arch_name in config.archetypes:
        spec = ARCHETYPES[arch_name]
        q_table, metadata = breed_archetype(
            spec,
            generations=config.breeding_generations,
            verbose=config.verbose
        )
        bred_archetypes[arch_name] = (q_table, metadata)
        
        if config.verbose:
            print(f"    → {arch_name} bred with fitness={metadata['final_fitness']:.3f}")
    
    breeding_time = time.time() - start_time
    print(f"\n  Breeding complete: {breeding_time:.1f}s")
    
    # =========================================================================
    # PHASE 2: TEST ALL ARCHETYPES × ALL CONDITIONS
    # =========================================================================
    print(f"\n[Phase 2: TESTING {len(config.archetypes)} ARCHETYPES × 32 CONDITIONS]")
    print("-" * 50)
    
    # Build job list
    jobs = []
    for arch_name, (q_table, _) in bred_archetypes.items():
        for bit_config in range(32):
            for seed in range(config.n_seeds):
                jobs.append((arch_name, q_table, bit_config, seed, config.max_turns))
    
    random.shuffle(jobs)  # Better load balancing
    
    results = {arch: {i: [] for i in range(32)} for arch in config.archetypes}
    
    progress = ProgressTracker(len(jobs), "Testing")
    
    # Track live metrics per archetype
    live_metrics = {arch: [] for arch in config.archetypes}
    
    test_start = time.time()
    
    with ProcessPoolExecutor(max_workers=config.n_cores) as executor:
        futures = {executor.submit(run_archetype_test, job): job for job in jobs}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                arch = result["archetype"]
                bit = result["bit_config"]
                results[arch][bit].append(result)
                
                live_metrics[arch].append(result["harm_rate"])
                
                # Update progress with per-archetype harm rates
                metrics = {f"{a[:2]}": fast_mean(live_metrics[a]) 
                          for a in config.archetypes if live_metrics[a]}
                progress.update(1, **metrics)
                
            except Exception as e:
                print(f"\n  [ERROR] {e}")
                progress.update(1)
    
    test_time = time.time() - test_start
    print(f"  Testing complete: {test_time:.1f}s ({len(jobs)/test_time:.0f} tests/sec)")
    
    # =========================================================================
    # PHASE 3: ANALYSIS
    # =========================================================================
    print(f"\n[Phase 3: ANALYSIS]")
    print("-" * 50)
    
    analysis = {}
    
    for arch_name in config.archetypes:
        arch_results = results[arch_name]
        
        # Aggregate across all conditions
        all_harms = []
        all_protects = []
        all_survivals = []
        
        condition_stats = {}
        for bit_config in range(32):
            runs = arch_results[bit_config]
            harms = [r["harm_rate"] for r in runs]
            protects = [r["protect_rate"] for r in runs]
            survivals = [r["survival_rate"] for r in runs]
            
            all_harms.extend(harms)
            all_protects.extend(protects)
            all_survivals.extend(survivals)
            
            condition_stats[bit_config] = {
                "harm_mean": fast_mean(harms),
                "harm_std": fast_std(harms),
                "protect_mean": fast_mean(protects),
                "survival_mean": fast_mean(survivals),
            }
        
        # Character persistence = 1 - CV of harm across conditions
        condition_means = [condition_stats[i]["harm_mean"] for i in range(32)]
        mean_of_means = fast_mean(condition_means)
        std_of_means = fast_std(condition_means)
        persistence = max(0, 1 - (std_of_means / mean_of_means)) if mean_of_means > 0 else 0
        
        analysis[arch_name] = {
            "overall_harm": fast_mean(all_harms),
            "overall_protect": fast_mean(all_protects),
            "overall_survival": fast_mean(all_survivals),
            "character_persistence": persistence,
            "condition_stats": condition_stats,
            "metadata": bred_archetypes[arch_name][1],
        }
        
        print(f"  {arch_name:10s}: harm={fast_mean(all_harms):.1%}, protect={fast_mean(all_protects):.1%}, persistence={persistence:.2f}")
    
    # =========================================================================
    # COMPUTE EFFECT SIZES BETWEEN ARCHETYPES
    # =========================================================================
    print(f"\n[Phase 4: ARCHETYPE COMPARISONS]")
    print("-" * 50)
    
    def cohens_d(g1, g2):
        if not g1 or not g2:
            return 0.0
        m1, m2 = fast_mean(g1), fast_mean(g2)
        n1, n2 = len(g1), len(g2)
        
        if HAS_NUMPY:
            v1 = np.var(g1, ddof=1)
            v2 = np.var(g2, ddof=1)
        else:
            v1 = sum((x-m1)**2 for x in g1)/(n1-1) if n1 > 1 else 0
            v2 = sum((x-m2)**2 for x in g2)/(n2-1) if n2 > 1 else 0
        
        pooled = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)) if n1+n2 > 2 else 1
        return (m1 - m2) / pooled if pooled > 0 else 0
    
    comparisons = {}
    arch_list = config.archetypes
    
    for i, arch1 in enumerate(arch_list):
        for arch2 in arch_list[i+1:]:
            # Get all harm rates for each
            harms1 = []
            harms2 = []
            for bit in range(32):
                harms1.extend([r["harm_rate"] for r in results[arch1][bit]])
                harms2.extend([r["harm_rate"] for r in results[arch2][bit]])
            
            d = cohens_d(harms1, harms2)
            comparisons[f"{arch1}_vs_{arch2}"] = {
                "cohens_d": d,
                "harm1": fast_mean(harms1),
                "harm2": fast_mean(harms2),
            }
            
            print(f"  {arch1} vs {arch2}: d={d:+.2f} ({fast_mean(harms1):.1%} vs {fast_mean(harms2):.1%})")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    total_time = time.time() - start_time
    
    output = {
        "experiment": "ARCHETYPE_BREEDING_TEST",
        "timestamp": timestamp,
        "config": {
            "archetypes": config.archetypes,
            "n_seeds": config.n_seeds,
            "max_turns": config.max_turns,
            "breeding_generations": config.breeding_generations,
        },
        "timing": {
            "breeding_sec": breeding_time,
            "testing_sec": test_time,
            "total_sec": total_time,
            "tests_per_sec": len(jobs) / test_time,
        },
        "archetypes": {
            name: {
                "metadata": analysis[name]["metadata"],
                "overall_harm": analysis[name]["overall_harm"],
                "overall_protect": analysis[name]["overall_protect"],
                "overall_survival": analysis[name]["overall_survival"],
                "character_persistence": analysis[name]["character_persistence"],
            }
            for name in config.archetypes
        },
        "comparisons": comparisons,
    }
    
    output_file = output_path / f"archetype_test_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ARCHETYPE EXPERIMENT RESULTS")
    print("=" * 70)
    
    print(f"\n⏱️  TIMING")
    print(f"   Total:    {total_time:.1f}s")
    print(f"   Breeding: {breeding_time:.1f}s")
    print(f"   Testing:  {test_time:.1f}s ({len(jobs)/test_time:.0f}/sec)")
    
    print(f"\n🎭 ARCHETYPE PROFILES (bred character)")
    print(f"   {'Name':<10} {'Harm':>8} {'Protect':>8} {'Survive':>8} {'Persist':>8}")
    print("   " + "-" * 44)
    for name in config.archetypes:
        a = analysis[name]
        print(f"   {name:<10} {a['overall_harm']:>7.1%} {a['overall_protect']:>7.1%} "
              f"{a['overall_survival']:>7.1%} {a['character_persistence']:>7.2f}")
    
    print(f"\n📊 KEY COMPARISONS")
    for comp_name, comp in comparisons.items():
        d = comp['cohens_d']
        interpretation = "massive" if abs(d) > 2 else "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        print(f"   {comp_name}: d={d:+.2f} ({interpretation})")
    
    print(f"\n💾 Results: {output_file}")
    print("=" * 70)
    
    # The punchline
    if "SAINT" in config.archetypes and "BRUTE" in config.archetypes:
        saint_harm = analysis["SAINT"]["overall_harm"]
        brute_harm = analysis["BRUTE"]["overall_harm"]
        d = comparisons.get("SAINT_vs_BRUTE", {}).get("cohens_d", 0)
        
        print(f"\n🎯 THE PUNCHLINE:")
        print(f"   Same code. Same Q-learning. Different breeding environment.")
        print(f"   SAINT harm: {saint_harm:.1%}")
        print(f"   BRUTE harm: {brute_harm:.1%}")
        print(f"   Effect size: d={d:+.2f}")
        print(f"   → Game design determines character.")
        print("=" * 70)
    
    return output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TEMPER Archetype Test')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick test')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--turns', type=int, default=200)
    parser.add_argument('--cores', type=int, default=0)
    parser.add_argument('--archetypes', type=str, default="SAINT,BRUTE,SHEEP,PREDATOR")
    parser.add_argument('--output', type=str, default='archetype_results')
    
    args = parser.parse_args()
    
    if args.quick:
        config = ArchetypeTestConfig(
            archetypes=args.archetypes.split(","),
            n_seeds=5,
            max_turns=100,
            breeding_generations=30,
            n_cores=args.cores,
            output_dir=args.output,
        )
    else:
        config = ArchetypeTestConfig(
            archetypes=args.archetypes.split(","),
            n_seeds=args.seeds,
            max_turns=args.turns,
            n_cores=args.cores,
            output_dir=args.output,
        )
    
    run_archetype_experiment(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
