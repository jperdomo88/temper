#!/usr/bin/env python3
"""
TEMPER Kernel Breeding for Machiavelli Transfer Test
=====================================================

Breeds SAINT and BRUTE kernels in the Crucible, then extracts their
behavioral profiles for use in the Machiavelli transfer test.

The behavioral profile captures HOW the kernel behaves - what percentage
of its actions fall into each TAO family. This is what transfers across
domains, not raw Q-values.

Usage:
    python breed_kernels.py --output kernels/

Author: TEMPER Project
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crucible.core.simulation import (
    breed_population, saint_fitness, brute_fitness,
    Simulation, SimulationParams, SwitchboardConfig, Profile
)
from crucible.core.agents import (
    AgentType, N_ACTIONS, N_STATES, Action, ACTION_TO_FAMILY
)


def extract_behavioral_profile(q_table: list, n_runs: int = 20, seed: int = 42) -> dict:
    """
    Run the kernel through the Crucible and extract its behavioral profile.
    
    The behavioral profile is the empirical action distribution - what
    percentage of actions fall into each TAO family. This is what transfers
    across domains.
    """
    params = SimulationParams()
    switchboard = SwitchboardConfig.temper_full()
    
    family_counts = {f: 0 for f in ['HARM', 'PROTECT', 'COOPERATE', 'COMMUNICATE', 'BOND', 'TRANSFORM', 'EXIST']}
    total_harm_rate = 0
    total_protect_rate = 0
    total_coop_rate = 0
    
    for run_seed in range(n_runs):
        sim = Simulation(params, switchboard, seed=seed * 1000 + run_seed)
        sim.initialize(AgentType.FROZEN, kernel=q_table)
        sim.run(max_turns=100)
        
        # Extract profile
        profile = Profile.from_events(sim.events, 0, sim.agents[0].alive)
        total_harm_rate += profile.harm_rate
        total_protect_rate += profile.protect_rate
        total_coop_rate += profile.cooperate_rate
        
        # Count actions by family
        for event in sim.events:
            if event.agent_id == 0:
                family = ACTION_TO_FAMILY.get(event.action, 'OTHER')
                if family in family_counts:
                    family_counts[family] += 1
    
    # Normalize to frequencies
    total = sum(family_counts.values())
    family_frequencies = {
        family: count / total if total > 0 else 0
        for family, count in family_counts.items()
    }
    
    return {
        'family_frequencies': family_frequencies,
        'harm_rate': total_harm_rate / n_runs,
        'protect_rate': total_protect_rate / n_runs,
        'cooperate_rate': total_coop_rate / n_runs,
        'total_actions': total,
        'n_runs': n_runs
    }


def breed_and_save_kernel(
    name: str,
    fitness_fn,
    output_dir: Path,
    generations: int = 50,
    pop_size: int = 20,
    eval_seeds: int = 5,
    breeding_seed: int = 42,
    profile_runs: int = 20
):
    """Breed a kernel and save with its behavioral profile."""
    
    print(f"\n{'='*60}")
    print(f"BREEDING {name} KERNEL")
    print(f"{'='*60}")
    print(f"Generations: {generations}, Population: {pop_size}")
    print(f"Breeding seed: {breeding_seed}")
    
    params = SimulationParams()
    switchboard = SwitchboardConfig.temper_full()
    
    # Breed the kernel
    q_table = breed_population(
        fitness_fn=fitness_fn,
        params=params,
        switchboard=switchboard,
        pop_size=pop_size,
        generations=generations,
        eval_seeds=eval_seeds,
        verbose=True,
        breeding_seed=breeding_seed
    )
    
    # Extract behavioral profile
    print(f"\nExtracting behavioral profile ({profile_runs} runs)...")
    profile = extract_behavioral_profile(q_table, n_runs=profile_runs, seed=breeding_seed)
    
    # Display profile
    print(f"\n{name} Behavioral Profile:")
    print(f"  Harm Rate:    {profile['harm_rate']:.1%}")
    print(f"  Protect Rate: {profile['protect_rate']:.1%}")
    print(f"  Cooperate Rate: {profile['cooperate_rate']:.1%}")
    print(f"\n  Family Frequencies:")
    for family, freq in sorted(profile['family_frequencies'].items(), key=lambda x: -x[1]):
        print(f"    {family:12s}: {freq:5.1%}")
    
    # Save
    kernel_data = {
        'name': name,
        'breeding_config': {
            'generations': generations,
            'pop_size': pop_size,
            'eval_seeds': eval_seeds,
            'breeding_seed': breeding_seed
        },
        'q_table': q_table,
        'behavioral_profile': profile
    }
    
    output_path = output_dir / f"{name.lower()}_kernel.json"
    with open(output_path, 'w') as f:
        json.dump(kernel_data, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    return kernel_data


def main():
    parser = argparse.ArgumentParser(description='Breed TEMPER kernels')
    parser.add_argument('--output', '-o', type=str, default='kernels/',
                        help='Output directory for kernel files')
    parser.add_argument('--generations', '-g', type=int, default=50,
                        help='Generations for breeding')
    parser.add_argument('--pop-size', '-p', type=int, default=20,
                        help='Population size')
    parser.add_argument('--profile-runs', '-r', type=int, default=20,
                        help='Number of runs for behavioral profile')
    parser.add_argument('--saint-only', action='store_true',
                        help='Only breed SAINT kernel')
    parser.add_argument('--brute-only', action='store_true',
                        help='Only breed BRUTE kernel')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.brute_only:
        breed_and_save_kernel(
            name='SAINT',
            fitness_fn=saint_fitness,
            output_dir=output_dir,
            generations=args.generations,
            pop_size=args.pop_size,
            profile_runs=args.profile_runs,
            breeding_seed=42
        )
    
    if not args.saint_only:
        breed_and_save_kernel(
            name='BRUTE',
            fitness_fn=brute_fitness,
            output_dir=output_dir,
            generations=args.generations,
            pop_size=args.pop_size,
            profile_runs=args.profile_runs,
            breeding_seed=43
        )
    
    print("\n" + "="*60)
    print("BREEDING COMPLETE")
    print("="*60)
    print(f"Kernels saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Run blind classification with your LLM (see prompts/classifier_prompt.md)")
    print("2. Run transfer test: python run_transfer_test.py")


if __name__ == '__main__':
    main()
