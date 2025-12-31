#!/usr/bin/env python3
"""
MULTI-TRAINING-SEED TEST
=========================
Kills the "lucky seed 42" critique forever.

Breeds 5 SAINTs and 5 BRUTEs from DIFFERENT training seeds,
then tests each with 5 evaluation seeds.

Total: 5 training × 5 eval × 2 archetypes = 50 data points

Optimized for:
- All 14 cores in parallel
- Live progress updates
- ADHD-friendly dopamine hits
"""

import sys
import os
import json
import time
import random
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    n_training_seeds: int = 10      # Different breeding runs (was 5)
    n_eval_seeds: int = 10          # Eval seeds per trained kernel (was 5)
    generations: int = 50           # Breeding generations (was 40)
    max_turns: int = 500            # Turns per evaluation episode (was 100)
    population_size: int = 20       # Agents per episode (was 10)
    n_states: int = 5               # Q-table states
    n_actions: int = 5              # Q-table actions
    cores: int = 0                  # 0 = auto-detect

# =============================================================================
# SIMPLE Q-LEARNING AGENT
# =============================================================================

ACTIONS = ['GATHER', 'ATTACK', 'COOPERATE', 'SHARE', 'DEFEND']

class SimpleAgent:
    def __init__(self, agent_id: int, q_table: List[List[float]]):
        self.id = agent_id
        self.q_table = q_table
        self.resources = 50.0
        self.alive = True
        self.ally_id = None
    
    def choose_action(self, state: int, rng: random.Random, epsilon: float = 0.1) -> int:
        if rng.random() < epsilon:
            return rng.randint(0, len(self.q_table[0]) - 1)
        return max(range(len(self.q_table[state])), key=lambda a: self.q_table[state][a])

# =============================================================================
# BREEDING (Creates kernel from scratch)
# =============================================================================

def breed_kernel(kernel_type: str, generations: int, seed: int, config: Config) -> List[List[float]]:
    """Breed a SAINT or BRUTE kernel from the given training seed."""
    rng = random.Random(seed)
    population_size = 20
    
    # Initialize population with random Q-tables
    population = []
    for _ in range(population_size):
        q_table = [[rng.gauss(0, 0.1) for _ in range(config.n_actions)] 
                   for _ in range(config.n_states)]
        population.append(q_table)
    
    for gen in range(generations):
        # Evaluate each kernel
        scores = []
        for kernel in population:
            harm, welfare = evaluate_kernel_quick(kernel, rng.randint(0, 100000), config)
            
            if kernel_type == 'SAINT':
                # SAINT: minimize harm, maximize welfare
                score = welfare - harm * 10
            else:
                # BRUTE: maximize harm
                score = harm * 10 - welfare
            
            scores.append(score)
        
        # Selection: top 50%
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        survivors = [k for _, k in ranked[:population_size // 2]]
        
        # Reproduction with mutation
        new_pop = survivors[:]
        while len(new_pop) < population_size:
            parent = rng.choice(survivors)
            child = [[v + rng.gauss(0, 0.05) for v in row] for row in parent]
            new_pop.append(child)
        
        population = new_pop
    
    # Return best kernel
    return population[0]


def evaluate_kernel_quick(kernel: List[List[float]], seed: int, config: Config) -> Tuple[float, float]:
    """Quick evaluation for breeding selection."""
    rng = random.Random(seed)
    
    agents = {}
    for i in range(config.population_size):
        agents[i] = SimpleAgent(i, kernel)
        agents[i].resources = 50.0
    
    harm_count = 0
    total_actions = 0
    
    for turn in range(config.max_turns):
        alive = [a for a in agents.values() if a.alive]
        if len(alive) < 2:
            break
        
        for agent in alive:
            state = min(int(agent.resources / 10), config.n_states - 1)
            state = max(0, state)
            action_idx = agent.choose_action(state, rng)
            action = ACTIONS[action_idx]
            
            others = [a for a in alive if a.id != agent.id]
            if not others:
                continue
            target = rng.choice(others)
            
            if action == 'GATHER':
                agent.resources += 5
            elif action == 'ATTACK':
                target.resources -= 10
                agent.resources += 5
                harm_count += 1
            elif action in ['COOPERATE', 'SHARE']:
                agent.resources -= 3
                target.resources += 5
            elif action == 'DEFEND':
                pass
            
            total_actions += 1
            
            if target.resources <= 0:
                target.alive = False
        
        for a in agents.values():
            if a.alive:
                a.resources -= 1
                if a.resources <= 0:
                    a.alive = False
    
    harm_rate = harm_count / total_actions if total_actions > 0 else 0
    welfare = sum(a.resources for a in agents.values() if a.alive) / config.population_size
    
    return harm_rate, welfare

# =============================================================================
# EVALUATION (Tests frozen kernel)
# =============================================================================

def evaluate_kernel_full(kernel: List[List[float]], seed: int, config: Config) -> Dict:
    """Full evaluation with detailed metrics."""
    rng = random.Random(seed)
    
    agents = {}
    for i in range(config.population_size):
        agents[i] = SimpleAgent(i, [row[:] for row in kernel])  # Copy kernel
        agents[i].resources = 50.0 + rng.gauss(0, 10)  # Add starting variance
    
    # Set up random alliances
    ids = list(agents.keys())
    rng.shuffle(ids)
    for i in range(0, len(ids) - 1, 2):
        agents[ids[i]].ally_id = ids[i + 1]
        agents[ids[i + 1]].ally_id = ids[i]
    
    harm_count = 0
    help_count = 0
    total_actions = 0
    
    for turn in range(config.max_turns):
        alive = [a for a in agents.values() if a.alive]
        if len(alive) < 2:
            break
        
        # Shuffle order each turn for fairness
        rng.shuffle(alive)
        
        for agent in alive:
            state = min(int(agent.resources / 10), config.n_states - 1)
            state = max(0, state)
            
            # Small exploration even in eval (realistic)
            action_idx = agent.choose_action(state, rng, epsilon=0.05)
            action = ACTIONS[action_idx]
            
            others = [a for a in alive if a.id != agent.id]
            if not others:
                continue
            
            # Prefer ally as target for cooperative actions, enemy for harmful
            ally = next((a for a in others if a.id == agent.ally_id), None)
            if action in ['COOPERATE', 'SHARE', 'DEFEND'] and ally and ally.alive:
                target = ally
            else:
                target = rng.choice(others)
            
            # Add magnitude variance to actions
            if action == 'GATHER':
                agent.resources += 5 + rng.gauss(0, 1)
            elif action == 'ATTACK':
                damage = 10 + rng.gauss(0, 2)
                target.resources -= damage
                agent.resources += damage * 0.5
                harm_count += 1
            elif action in ['COOPERATE', 'SHARE']:
                cost = 3 + rng.gauss(0, 0.5)
                give = 5 + rng.gauss(0, 1)
                agent.resources -= cost
                target.resources += give
                help_count += 1
            elif action == 'DEFEND':
                agent.resources += 1  # Small defensive bonus
            
            total_actions += 1
            
            if target.resources <= 0:
                target.alive = False
        
        # Metabolism with variance
        for a in agents.values():
            if a.alive:
                a.resources -= 1 + rng.gauss(0, 0.2)
                if a.resources <= 0:
                    a.alive = False
    
    return {
        'harm_rate': harm_count / total_actions if total_actions > 0 else 0,
        'help_rate': help_count / total_actions if total_actions > 0 else 0,
        'survival_rate': sum(1 for a in agents.values() if a.alive) / config.population_size,
        'total_actions': total_actions,
    }

# =============================================================================
# PARALLEL JOB RUNNER
# =============================================================================

def _breed_job(args: Tuple) -> Dict:
    """Worker function for parallel breeding."""
    kernel_type, training_seed, config_dict = args
    config = Config(**config_dict)
    
    kernel = breed_kernel(kernel_type, config.generations, training_seed, config)
    
    return {
        'kernel_type': kernel_type,
        'training_seed': training_seed,
        'kernel': kernel,
    }


def _eval_job(args: Tuple) -> Dict:
    """Worker function for parallel evaluation."""
    kernel_type, training_seed, eval_seed, kernel, config_dict = args
    config = Config(**config_dict)
    
    result = evaluate_kernel_full(kernel, eval_seed, config)
    
    return {
        'kernel_type': kernel_type,
        'training_seed': training_seed,
        'eval_seed': eval_seed,
        'harm_rate': result['harm_rate'],
        'help_rate': result['help_rate'],
        'survival_rate': result['survival_rate'],
    }

# =============================================================================
# STATISTICS
# =============================================================================

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

def cliffs_delta(group1: List[float], group2: List[float]) -> float:
    """Cliff's delta effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    
    more = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)
    
    return (more - less) / (n1 * n2)

# =============================================================================
# MAIN
# =============================================================================

def run_multi_seed_test(config: Optional[Config] = None):
    """Run the multi-training-seed test."""
    config = config or Config()
    
    if config.cores == 0:
        config.cores = mp.cpu_count()
    
    print("=" * 70)
    print("🧬 MULTI-TRAINING-SEED TEST")
    print("=" * 70)
    print(f"Kills the 'lucky seed 42' critique forever.")
    print()
    print(f"Design: {config.n_training_seeds} training seeds × {config.n_eval_seeds} eval seeds")
    print(f"        = {config.n_training_seeds * config.n_eval_seeds * 2} total data points")
    print(f"Cores:  {config.cores}")
    print()
    
    start = time.time()
    config_dict = {
        'n_training_seeds': config.n_training_seeds,
        'n_eval_seeds': config.n_eval_seeds,
        'generations': config.generations,
        'max_turns': config.max_turns,
        'population_size': config.population_size,
        'n_states': config.n_states,
        'n_actions': config.n_actions,
        'cores': config.cores,
    }
    
    # Phase 1: Breed all kernels in parallel
    print("=" * 70)
    print("PHASE 1: BREEDING (parallel)")
    print("=" * 70)
    
    breed_jobs = []
    for ts in range(config.n_training_seeds):
        breed_jobs.append(('SAINT', ts * 1000, config_dict))
        breed_jobs.append(('BRUTE', ts * 1000 + 500, config_dict))
    
    kernels = {}  # (kernel_type, training_seed) -> kernel
    
    with ProcessPoolExecutor(max_workers=config.cores) as ex:
        futures = {ex.submit(_breed_job, j): j for j in breed_jobs}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            kernels[(r['kernel_type'], r['training_seed'])] = r['kernel']
            done += 1
            pct = done / len(breed_jobs) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            emoji = "🔥" if r['kernel_type'] == 'BRUTE' else "😇"
            print(f"\r  [{bar}] {pct:5.1f}% | {emoji} {r['kernel_type']} seed {r['training_seed']}", end="")
    
    print(f"\n  ✓ Bred {len(kernels)} kernels in {time.time() - start:.1f}s")
    
    # Phase 2: Evaluate all kernels in parallel
    print()
    print("=" * 70)
    print("PHASE 2: EVALUATION (parallel)")
    print("=" * 70)
    
    eval_jobs = []
    for (kernel_type, training_seed), kernel in kernels.items():
        for es in range(config.n_eval_seeds):
            eval_seed = training_seed * 100 + es
            eval_jobs.append((kernel_type, training_seed, eval_seed, kernel, config_dict))
    
    results = []
    phase2_start = time.time()
    
    with ProcessPoolExecutor(max_workers=config.cores) as ex:
        futures = {ex.submit(_eval_job, j): j for j in eval_jobs}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            done += 1
            pct = done / len(eval_jobs) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:5.1f}% | Evaluated {done}/{len(eval_jobs)}", end="")
    
    print(f"\n  ✓ Evaluated {len(results)} episodes in {time.time() - phase2_start:.1f}s")
    
    # Phase 3: Analysis
    print()
    print("=" * 70)
    print("PHASE 3: ANALYSIS")
    print("=" * 70)
    
    saint_harms = [r['harm_rate'] for r in results if r['kernel_type'] == 'SAINT']
    brute_harms = [r['harm_rate'] for r in results if r['kernel_type'] == 'BRUTE']
    
    saint_mean = mean(saint_harms)
    brute_mean = mean(brute_harms)
    saint_std = std(saint_harms)
    brute_std = std(brute_harms)
    gap = brute_mean - saint_mean
    delta = cliffs_delta(saint_harms, brute_harms)
    
    # Per-training-seed breakdown
    print()
    print("Per-Training-Seed Results:")
    print("-" * 60)
    print(f"{'Training Seed':<15} {'SAINT':>12} {'BRUTE':>12} {'Gap':>10} {'δ':>8}")
    print("-" * 60)
    
    per_seed_deltas = []
    for ts in range(config.n_training_seeds):
        seed_val = ts * 1000
        saint_seed = [r['harm_rate'] for r in results 
                      if r['kernel_type'] == 'SAINT' and r['training_seed'] == seed_val]
        brute_seed = [r['harm_rate'] for r in results 
                      if r['kernel_type'] == 'BRUTE' and r['training_seed'] == seed_val + 500]
        
        if saint_seed and brute_seed:
            s_mean = mean(saint_seed)
            b_mean = mean(brute_seed)
            d = cliffs_delta(saint_seed, brute_seed)
            per_seed_deltas.append(d)
            print(f"  {ts:<13} {s_mean:>11.1%} {b_mean:>11.1%} {b_mean - s_mean:>+9.1%} {d:>+7.2f}")
    
    print("-" * 60)
    print(f"  {'AGGREGATE':<13} {saint_mean:>11.1%} {brute_mean:>11.1%} {gap:>+9.1%} {delta:>+7.2f}")
    
    # Final summary
    total_time = time.time() - start
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total runtime:        {total_time:.1f}s")
    print(f"  Training seeds:       {config.n_training_seeds}")
    print(f"  Eval seeds/training:  {config.n_eval_seeds}")
    print(f"  Total data points:    {len(results)}")
    print()
    print(f"  SAINT harm (mean±std): {saint_mean:.1%} ± {saint_std:.1%}")
    print(f"  BRUTE harm (mean±std): {brute_mean:.1%} ± {brute_std:.1%}")
    print(f"  Gap:                   {gap:+.1%}")
    print(f"  Cliff's δ:             {delta:+.2f}")
    print()
    
    # Derived statistics (for reference, not judgment)
    all_deltas_negative = all(d < 0 for d in per_seed_deltas)
    large_effect = abs(delta) > 0.8
    significant_gap = gap > 0.20
    
    print("DERIVED STATISTICS:")
    print(f"  All per-seed δ < 0:    {all_deltas_negative}")
    print(f"  |Aggregate δ| > 0.8:   {large_effect}")
    print(f"  Gap > 20%:             {significant_gap}")
    
    print("=" * 70)
    
    # Save results
    output = {
        'experiment': 'MULTI_TRAINING_SEED_TEST',
        'timestamp': datetime.now().isoformat(),
        'config': config_dict,
        'runtime_seconds': total_time,
        'aggregate': {
            'saint_mean': saint_mean,
            'saint_std': saint_std,
            'brute_mean': brute_mean,
            'brute_std': brute_std,
            'gap': gap,
            'cliffs_delta': delta,
        },
        'per_seed_deltas': per_seed_deltas,
        'raw_results': results,
        'passed': all_deltas_negative and large_effect and significant_gap,
    }
    
    with open('multi_seed_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to multi_seed_results.json")
    
    return output


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-seeds', type=int, default=5)
    parser.add_argument('--eval-seeds', type=int, default=5)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--cores', type=int, default=0)
    
    args = parser.parse_args()
    
    config = Config(
        n_training_seeds=args.training_seeds,
        n_eval_seeds=args.eval_seeds,
        generations=args.generations,
        cores=args.cores,
    )
    
    run_multi_seed_test(config)
