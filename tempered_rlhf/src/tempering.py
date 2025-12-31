"""
Tempering: Fitness-Hidden Evolutionary Selection of Reward Models.

The core mechanism being tested: breed RMs via selection on true outcomes,
but RMs never see their fitness scores. This disrupts the TSC-IR
(train/select/check inner/outer loop corruption) attack vector.

Key insight: RMs cannot optimize for selection pressure they cannot observe.
"""

import numpy as np
import torch
from typing import List, Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from env_civicgrid import CivicGrid, Step, deterministic_seed_mix
from models import RewardNet, create_policy, create_reward_model
from train_policy import train_policy, collect_rollouts
from metrics import true_score


@dataclass
class TemperingConfig:
    """Configuration for tempering experiment."""
    pop_size: int = 12          # Population size
    num_gens: int = 8           # Number of generations
    elite_k: int = 4            # Top-k survive
    train_steps: int = 2000     # Policy training steps per RM
    eval_episodes: int = 20     # Episodes for fitness evaluation
    mutation_sigma: float = 0.02  # Mutation noise
    hidden: List[int] = None    # RM hidden layers
    
    def __post_init__(self):
        if self.hidden is None:
            self.hidden = [64, 64]


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.array(x) / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def temper_reward_models(
    env: CivicGrid,
    ref_trajectories: List[List[Step]],
    config: TemperingConfig = None,
    seed: int = 42,
    on_generation: Optional[Callable] = None,
    on_rm_eval: Optional[Callable] = None,
    verbose: bool = False,
    ablation_visible_fitness: bool = False,
) -> Tuple[RewardNet, Dict[str, Any]]:
    """
    Breed RMs via fitness-hidden selection.
    
    RMs never see their fitness scores. Selection happens based on
    true_score of policies trained on each RM, but this information
    is used only for selection - never fed back to the RM.
    
    Parameters:
    -----------
    env : CivicGrid
        Environment
    ref_trajectories : list
        Fixed reference trajectories for z-normalization
    config : TemperingConfig
        Experiment configuration
    seed : int
        Random seed
    on_generation : callable
        Callback after each generation: on_generation(gen, stats)
    on_rm_eval : callable
        Callback after each RM evaluation: on_rm_eval(gen, rm_idx, fitness)
    verbose : bool
        Print progress
    ablation_visible_fitness : bool
        If True, store fitness on RM (breaks tempering - for ablation)
        
    Returns:
    --------
    (best_rm, history) : tuple
        Best RM from final generation and training history
    """
    if config is None:
        config = TemperingConfig()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    # Initialize population with random RMs
    population = []
    for i in range(config.pop_size):
        rm = create_reward_model(hidden=config.hidden)
        # Cache reference stats immediately
        rm.cache_reference_stats(ref_trajectories)
        population.append(rm)
    
    # Training history
    history = {
        'generations': [],
        'best_fitness': [],
        'mean_fitness': [],
        'fitness_std': [],
    }
    
    for gen in range(config.num_gens):
        fitnesses = []
        
        for rm_idx, rm in enumerate(population):
            # Derive deterministic seed for this RM's training
            rm_seed = deterministic_seed_mix(seed, gen * config.pop_size + rm_idx)
            
            # Train a policy using this RM (RM is frozen during this)
            policy = train_policy(
                env, rm, 
                steps=config.train_steps,
                seed=rm_seed,
                verbose=False
            )
            
            # Evaluate policy on TRUE metric (HIDDEN from RM)
            eval_seed = deterministic_seed_mix(rm_seed, 999)
            trajectories = collect_rollouts(
                policy, env, 
                episodes=config.eval_episodes,
                seed=eval_seed
            )
            
            # Compute fitness on true evaluator
            fitness = np.mean([true_score(traj) for traj in trajectories])
            fitnesses.append(fitness)
            
            # CRITICAL: RM never receives this fitness value
            # No gradients, no feedback, just... existence and selection
            
            # ABLATION: If visible fitness mode, store it (breaks tempering)
            if ablation_visible_fitness:
                rm.last_fitness = fitness  # This defeats the mechanism!
            
            # Callback
            if on_rm_eval is not None:
                on_rm_eval(gen, rm_idx, fitness)
            
            if verbose:
                print(f"  Gen {gen} | RM {rm_idx}/{config.pop_size} | "
                      f"Fitness: {fitness:.4f}")
        
        # Record stats
        history['generations'].append(gen)
        history['best_fitness'].append(max(fitnesses))
        history['mean_fitness'].append(np.mean(fitnesses))
        history['fitness_std'].append(np.std(fitnesses))
        
        if verbose:
            print(f"Gen {gen}: best={max(fitnesses):.4f}, "
                  f"mean={np.mean(fitnesses):.4f}, "
                  f"std={np.std(fitnesses):.4f}")
        
        # Callback
        if on_generation is not None:
            on_generation(gen, {
                'best_fitness': max(fitnesses),
                'mean_fitness': np.mean(fitnesses),
                'fitness_std': np.std(fitnesses),
                'fitnesses': fitnesses.copy(),
            })
        
        # Selection: top-k elites survive
        elite_indices = np.argsort(fitnesses)[-config.elite_k:]
        elites = [population[i] for i in elite_indices]
        elite_fitnesses = [fitnesses[i] for i in elite_indices]
        
        # Skip reproduction on last generation
        if gen == config.num_gens - 1:
            break
        
        # Reproduction: sample from elites, mutate to fill population
        new_population = []
        
        # Fitness-proportional parent selection (softmax over elite fitnesses)
        probs = softmax(elite_fitnesses)
        
        for _ in range(config.pop_size):
            # Select parent
            parent_idx = rng.choice(len(elites), p=probs)
            parent = elites[parent_idx]
            
            # Clone and mutate
            child = parent.clone()
            child.mutate(sigma=config.mutation_sigma)
            
            # CRITICAL: Cache reference stats for new RM
            child.cache_reference_stats(ref_trajectories)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best RM from final generation
    best_idx = np.argmax(fitnesses)
    best_rm = population[best_idx]
    
    # Ensure stats are cached
    best_rm.cache_reference_stats(ref_trajectories)
    
    return best_rm, history


def run_tempering_with_ablation(
    env: CivicGrid,
    ref_trajectories: List[List[Step]],
    config: TemperingConfig = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run tempering with and without visible fitness (ablation study).
    
    Returns results for both conditions to validate that visible fitness
    breaks the tempering mechanism.
    
    Expected: visible_fitness condition should perform closer to proxy
    than to hidden_fitness (tempered) condition.
    """
    results = {}
    
    # Hidden fitness (normal tempering)
    if verbose:
        print("\n" + "="*50)
        print("Running HIDDEN FITNESS (normal tempering)")
        print("="*50)
    
    rm_hidden, history_hidden = temper_reward_models(
        env, ref_trajectories, config, seed,
        ablation_visible_fitness=False,
        verbose=verbose
    )
    results['hidden'] = {
        'rm': rm_hidden,
        'history': history_hidden,
    }
    
    # Visible fitness (ablation - should break tempering)
    if verbose:
        print("\n" + "="*50)
        print("Running VISIBLE FITNESS (ablation)")
        print("="*50)
    
    rm_visible, history_visible = temper_reward_models(
        env, ref_trajectories, config, seed + 1000,  # Different seed
        ablation_visible_fitness=True,
        verbose=verbose
    )
    results['visible'] = {
        'rm': rm_visible,
        'history': history_visible,
    }
    
    return results


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tempering Module")
    print("=" * 60)
    print("\nThis module provides:")
    print("- TemperingConfig: Configuration dataclass")
    print("- temper_reward_models(): Core evolutionary selection")
    print("- run_tempering_with_ablation(): Ablation study")
    print("\nKey mechanism: RMs are selected on true outcomes,")
    print("but never see their fitness scores.")
    print("=" * 60)
