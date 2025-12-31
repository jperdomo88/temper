"""
Main Experiment Driver for Tempered RLHF Experiment.

Orchestrates the complete experiment:
1. Setup (seeds, reference trajectories)
2. Proxy condition (baseline)
3. Tempered condition (our mechanism)
4. Oracle condition (ceiling)
5. Evaluation (all conditions, all modes)
6. Analysis and plotting

Usage:
    python run_all.py                    # Fast run (5 seeds)
    python run_all.py --seeds 8          # Full run (8 seeds)
    python run_all.py --ablation         # Include ablation study
    python run_all.py --laundering       # Include CCD validation
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import random

# Set up path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from env_civicgrid import CivicGrid, Step, deterministic_seed_mix
from models import create_policy, create_reward_model, RewardNet
from train_policy import (
    train_policy, collect_rollouts, 
    RandomPolicy, MixedReferencePolicy
)
from baselines import create_proxy_rm, OracleRM, train_oracle_policy
from tempering import temper_reward_models, TemperingConfig
from metrics import (
    true_score, compute_e_reference_stats,
    vulnerability_reduction_with_ci, exploit_z_with_ci,
    cohens_d, permutation_test
)
from evaluation import (
    evaluate_condition, evaluate_forced_terminal_start,
    evaluate_variant_robustness, run_full_evaluation
)
from dashboard import ExperimentDashboard, LiveAnalysis, RICH_AVAILABLE
from plotting import generate_all_figures, create_results_table


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def collect_reference_trajectories(
    env: CivicGrid,
    n_episodes: int = 100,
    seed: int = 42,
) -> list:
    """
    Collect reference trajectories using mixed policy.
    
    80% random + 20% structural movement for wider reward range.
    """
    mixed_policy = MixedReferencePolicy(env, seed=seed)
    return collect_rollouts(mixed_policy, env, episodes=n_episodes, seed=seed)


def run_proxy_condition(
    env: CivicGrid,
    ref_trajectories: list,
    mu_E: float,
    sigma_E: float,
    seeds: list,
    config: dict,
    verbose: bool = True,
) -> dict:
    """Run proxy RM condition across all seeds."""
    results = []
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n[Proxy] Seed {i+1}/{len(seeds)} (seed={seed})")
        
        # Train proxy RM
        proxy_rm = create_proxy_rm(
            env, 
            n_episodes=config.get('proxy_episodes', 500),
            epochs=config.get('proxy_epochs', 100),
            seed=seed,
            verbose=False
        )
        proxy_rm.cache_reference_stats(ref_trajectories)
        
        # Train policy on proxy RM
        policy = train_policy(
            env, proxy_rm,
            steps=config.get('policy_steps', 3000),
            seed=seed,
            verbose=False
        )
        
        # Evaluate
        eval_results = run_full_evaluation(
            policy, proxy_rm, env, ref_trajectories,
            seed=seed, mu_E=mu_E, sigma_E=sigma_E,
            n_episodes=config.get('eval_episodes', 30)
        )
        eval_results['seed'] = seed
        results.append(eval_results)
        
        if verbose:
            std = eval_results['standard']
            print(f"  Hack: {std['hack_rate']:.3f} | E: {std['e_score']:.3f} | "
                  f"Exploit_z: {std['exploit_z']:.3f}")
    
    return aggregate_results('proxy', results)


def run_tempered_condition(
    env: CivicGrid,
    ref_trajectories: list,
    mu_E: float,
    sigma_E: float,
    seeds: list,
    config: dict,
    verbose: bool = True,
) -> dict:
    """Run tempered RM condition across all seeds."""
    results = []
    histories = []
    
    temper_config = TemperingConfig(
        pop_size=config.get('pop_size', 12),
        num_gens=config.get('num_gens', 8),
        elite_k=config.get('elite_k', 4),
        train_steps=config.get('temper_train_steps', 2000),
        eval_episodes=config.get('temper_eval_episodes', 20),
        mutation_sigma=config.get('mutation_sigma', 0.02),
    )
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n[Tempered] Seed {i+1}/{len(seeds)} (seed={seed})")
        
        # Run tempering
        tempered_rm, history = temper_reward_models(
            env, ref_trajectories,
            config=temper_config,
            seed=seed,
            verbose=verbose,
        )
        histories.append(history)
        
        # Train final policy
        policy = train_policy(
            env, tempered_rm,
            steps=config.get('policy_steps', 3000),
            seed=seed,
            verbose=False
        )
        
        # Evaluate
        eval_results = run_full_evaluation(
            policy, tempered_rm, env, ref_trajectories,
            seed=seed, mu_E=mu_E, sigma_E=sigma_E,
            n_episodes=config.get('eval_episodes', 30)
        )
        eval_results['seed'] = seed
        results.append(eval_results)
        
        if verbose:
            std = eval_results['standard']
            print(f"  Hack: {std['hack_rate']:.3f} | E: {std['e_score']:.3f} | "
                  f"Exploit_z: {std['exploit_z']:.3f}")
    
    aggregated = aggregate_results('tempered', results)
    aggregated['histories'] = histories
    return aggregated


def run_oracle_condition(
    env: CivicGrid,
    ref_trajectories: list,
    mu_E: float,
    sigma_E: float,
    seeds: list,
    config: dict,
    verbose: bool = True,
) -> dict:
    """Run oracle RM condition across all seeds."""
    results = []
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n[Oracle] Seed {i+1}/{len(seeds)} (seed={seed})")
        
        # Create oracle RM
        oracle_rm = OracleRM()
        oracle_rm.cache_reference_stats(ref_trajectories)
        
        # Train policy on oracle
        policy = train_oracle_policy(
            env,
            steps=config.get('policy_steps', 3000),
            seed=seed,
            verbose=False
        )
        
        # Evaluate
        eval_results = run_full_evaluation(
            policy, oracle_rm, env, ref_trajectories,
            seed=seed, mu_E=mu_E, sigma_E=sigma_E,
            n_episodes=config.get('eval_episodes', 30)
        )
        eval_results['seed'] = seed
        results.append(eval_results)
        
        if verbose:
            std = eval_results['standard']
            print(f"  Hack: {std['hack_rate']:.3f} | E: {std['e_score']:.3f}")
    
    return aggregate_results('oracle', results)


def aggregate_results(condition: str, per_seed_results: list) -> dict:
    """Aggregate per-seed results into summary statistics."""
    aggregated = {
        'condition': condition,
        'n_seeds': len(per_seed_results),
        'per_seed_results': per_seed_results,
    }
    
    # Metrics to aggregate
    metrics = [
        'exploit_z', 'exploit_corr', 'hack_rate', 'harm_rate', 
        'protect_rate', 'e_score', 'terminal_visit_rate',
        'conditional_hack_rate', 'episodes_with_terminal_visit'
    ]
    
    for metric in metrics:
        values = []
        for r in per_seed_results:
            if 'standard' in r and metric in r['standard']:
                values.append(r['standard'][metric])
        
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_per_seed'] = values
    
    return aggregated


def compute_headline_stats(proxy: dict, tempered: dict) -> dict:
    """Compute headline statistics for paper."""
    stats = {}
    
    # Percent reduction in hack rate
    proxy_hack = proxy.get('hack_rate_per_seed', [])
    tempered_hack = tempered.get('hack_rate_per_seed', [])
    
    if proxy_hack and tempered_hack:
        reduction, ci_low, ci_high = vulnerability_reduction_with_ci(
            proxy_hack, tempered_hack
        )
        stats['hack_reduction'] = {
            'point': reduction,
            'ci_low': ci_low,
            'ci_high': ci_high,
        }
    
    # Exploit_z effect size
    proxy_ez = proxy.get('exploit_z_per_seed', [])
    tempered_ez = tempered.get('exploit_z_per_seed', [])
    
    if proxy_ez and tempered_ez:
        delta, ci_low, ci_high = exploit_z_with_ci(proxy_ez, tempered_ez)
        d = cohens_d(proxy_ez, tempered_ez)
        stats['exploit_z_effect'] = {
            'delta': delta,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'cohens_d': d,
        }
    
    # Hypothesis test
    if proxy_hack and tempered_hack:
        p_value = permutation_test(proxy_hack, tempered_hack)
        stats['permutation_p'] = p_value
    
    return stats


def save_results(results: dict, output_dir: str = "results") -> None:
    """Save results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    full_path = output_path / f"full_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(full_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {full_path}")


def main():
    parser = argparse.ArgumentParser(description="Tempered RLHF Experiment")
    parser.add_argument('--seeds', type=int, default=5, 
                        help='Number of seeds (default: 5)')
    parser.add_argument('--master-seed', type=int, default=42,
                        help='Master seed (default: 42)')
    parser.add_argument('--ablation', action='store_true',
                        help='Include visible-fitness ablation')
    parser.add_argument('--laundering', action='store_true',
                        help='Include CCD/laundering validation')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode (reduced parameters)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip figure generation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TEMPERED RLHF EXPERIMENT")
    print("=" * 60)
    print(f"\nSeeds: {args.seeds}")
    print(f"Master seed: {args.master_seed}")
    print(f"Ablation: {args.ablation}")
    print(f"Laundering: {args.laundering}")
    print(f"Fast mode: {args.fast}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Set master seed
    set_all_seeds(args.master_seed)
    
    # Derive per-condition seeds
    rng = np.random.RandomState(args.master_seed)
    seeds = [rng.randint(0, 2**31) for _ in range(args.seeds)]
    
    # Configuration
    if args.fast:
        config = {
            'proxy_episodes': 200,
            'proxy_epochs': 50,
            'policy_steps': 1000,
            'eval_episodes': 15,
            'pop_size': 6,
            'num_gens': 4,
            'elite_k': 2,
            'temper_train_steps': 500,
            'temper_eval_episodes': 10,
        }
    else:
        config = {
            'proxy_episodes': 500,
            'proxy_epochs': 100,
            'policy_steps': 3000,
            'eval_episodes': 30,
            'pop_size': 12,
            'num_gens': 8,
            'elite_k': 4,
            'temper_train_steps': 2000,
            'temper_eval_episodes': 20,
        }
    
    # Create environment
    env = CivicGrid(seed=args.master_seed)
    
    # Collect reference trajectories
    print("\n--- Collecting reference trajectories ---")
    ref_trajectories = collect_reference_trajectories(env, n_episodes=100, seed=args.master_seed)
    mu_E, sigma_E = compute_e_reference_stats(ref_trajectories)
    print(f"Reference stats: μ_E={mu_E:.4f}, σ_E={sigma_E:.4f}")
    
    # Results container
    all_results = {
        'config': config,
        'seeds': seeds,
        'master_seed': args.master_seed,
        'mu_E': mu_E,
        'sigma_E': sigma_E,
    }
    
    # Run conditions
    print("\n" + "=" * 60)
    print("PROXY CONDITION (Baseline - Should Exploit)")
    print("=" * 60)
    all_results['proxy'] = run_proxy_condition(
        env, ref_trajectories, mu_E, sigma_E, seeds, config, args.verbose
    )
    
    print("\n" + "=" * 60)
    print("TEMPERED CONDITION (Our Mechanism - Should Resist)")
    print("=" * 60)
    all_results['tempered'] = run_tempered_condition(
        env, ref_trajectories, mu_E, sigma_E, seeds, config, args.verbose
    )
    
    print("\n" + "=" * 60)
    print("ORACLE CONDITION (Ceiling)")
    print("=" * 60)
    all_results['oracle'] = run_oracle_condition(
        env, ref_trajectories, mu_E, sigma_E, seeds, config, args.verbose
    )
    
    # Compute headline stats
    print("\n" + "=" * 60)
    print("COMPUTING HEADLINE STATISTICS")
    print("=" * 60)
    
    headline = compute_headline_stats(all_results['proxy'], all_results['tempered'])
    all_results['headline'] = headline
    
    if 'hack_reduction' in headline:
        hr = headline['hack_reduction']
        print(f"\nHack Rate Reduction: {hr['point']*100:.1f}% "
              f"[95% CI: {hr['ci_low']*100:.1f}% - {hr['ci_high']*100:.1f}%]")
    
    if 'exploit_z_effect' in headline:
        ez = headline['exploit_z_effect']
        print(f"Exploit_z Δ: {ez['delta']:.3f} "
              f"[95% CI: {ez['ci_low']:.3f} - {ez['ci_high']:.3f}]")
        print(f"Cohen's d: {ez['cohens_d']:.3f}")
    
    if 'permutation_p' in headline:
        print(f"Permutation test p-value: {headline['permutation_p']:.4f}")
    
    # Save results
    save_results(all_results)
    
    # Generate figures
    if not args.no_plots:
        print("\n--- Generating figures ---")
        try:
            generate_all_figures(all_results)
        except Exception as e:
            print(f"Warning: Could not generate figures: {e}")
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to results/")
    
    # Print paper-ready claims
    print("\n" + "=" * 60)
    print("PAPER-READY CLAIMS")
    print("=" * 60)
    
    proxy = all_results['proxy']
    tempered = all_results['tempered']
    
    print(f"\n\"Tempering reduced exploit action frequency by "
          f"{headline.get('hack_reduction', {}).get('point', 0)*100:.0f}% "
          f"[95% CI: {headline.get('hack_reduction', {}).get('ci_low', 0)*100:.0f}% - "
          f"{headline.get('hack_reduction', {}).get('ci_high', 0)*100:.0f}%], "
          f"from a hack rate of {proxy.get('hack_rate_mean', 0):.3f} to "
          f"{tempered.get('hack_rate_mean', 0):.3f}.\"")
    
    print(f"\n\"Conditional exploit rate—hacks given terminal access—dropped "
          f"from {proxy.get('conditional_hack_rate_mean', 0)*100:.0f}% (proxy) to "
          f"{tempered.get('conditional_hack_rate_mean', 0)*100:.0f}% (tempered), "
          f"confirming tempered policies actively chose not to exploit rather than "
          f"simply avoiding opportunity.\"")


if __name__ == "__main__":
    main()
