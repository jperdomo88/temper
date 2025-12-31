#!/usr/bin/env python3
"""
TEMPER VALIDATION SUITE
=======================

Runs all validation experiments and outputs raw data.

Usage:
    python3 run_validation.py          # Full run
    python3 run_validation.py --quick  # Quick test
"""

import sys
import os

# Set up path so imports work regardless of how script is called
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Add both the script dir and parent to path
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Now we can do imports
import json
import time
from datetime import datetime
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TEMPER VALIDATION SUITE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Script dir: {SCRIPT_DIR}")
    print()
    
    # Configuration
    if args.quick:
        n_training = 3
        n_eval = 3
        h_seeds = 5
        mode = "quick"
    else:
        n_training = 10
        n_eval = 10
        h_seeds = 20
        mode = "full"
    
    print(f"Mode: {mode}")
    print(f"Training seeds: {n_training}")
    print(f"Eval seeds per training: {n_eval}")
    print(f"Exp H seeds: {h_seeds}")
    print()
    
    # Create results directory
    results_dir = Path(SCRIPT_DIR) / args.output
    results_dir.mkdir(exist_ok=True)
    
    overall_start = time.time()
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'config': {
            'n_training_seeds': n_training,
            'n_eval_seeds': n_eval,
            'h_seeds': h_seeds,
        }
    }
    
    # =========================================================================
    # EXPERIMENT: MULTI-TRAINING-SEED TEST
    # =========================================================================
    print("=" * 70)
    print("EXPERIMENT: MULTI-TRAINING-SEED")
    print(f"Design: {n_training} training seeds × {n_eval} eval seeds × 2 archetypes")
    print(f"Total data points: {n_training * n_eval * 2}")
    print("=" * 70)
    print()
    
    try:
        from experiments.exp_multi_seed import run_multi_seed_test, Config
        
        config = Config(
            n_training_seeds=n_training,
            n_eval_seeds=n_eval,
            generations=50,
            max_turns=500,
            population_size=20
        )
        result = run_multi_seed_test(config)
        all_results['multi_seed'] = result
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        all_results['multi_seed'] = {'error': str(e)}
    
    # =========================================================================
    # EXPERIMENT: CROSS-DOMAIN × CROSS-ADAPTER (Exp H)
    # =========================================================================
    print()
    print("=" * 70)
    print("EXPERIMENT: CROSS-DOMAIN × CROSS-ADAPTER")
    print(f"Design: 2 domains × 2 adapters × {h_seeds} seeds")
    print(f"Total conditions: 4")
    print("=" * 70)
    print()
    
    try:
        from experiments.exp_h_combined import run_exp_h, ExpHConfig
        
        config = ExpHConfig(n_seeds=h_seeds)
        result = run_exp_h(config)
        all_results['exp_h'] = result.to_dict()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        all_results['exp_h'] = {'error': str(e)}
    
    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    total_time = time.time() - overall_start
    all_results['total_runtime_seconds'] = total_time
    
    # Save full results
    output_file = results_dir / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary report
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_time:.1f}s")
    print()
    
    # Multi-seed results
    if 'multi_seed' in all_results and 'error' not in all_results['multi_seed']:
        ms = all_results['multi_seed']
        print("MULTI-TRAINING-SEED TEST")
        print("-" * 70)
        print(f"  Training seeds tested: {ms['config']['n_training_seeds']}")
        print(f"  Eval seeds per training: {ms['config']['n_eval_seeds']}")
        print(f"  Total data points: {len(ms['raw_results'])}")
        print()
        print(f"  SAINT harm rate: {ms['aggregate']['saint_mean']:.1%} ± {ms['aggregate']['saint_std']:.1%}")
        print(f"  BRUTE harm rate: {ms['aggregate']['brute_mean']:.1%} ± {ms['aggregate']['brute_std']:.1%}")
        print(f"  Gap: {ms['aggregate']['gap']:+.1%}")
        print(f"  Cliff's δ: {ms['aggregate']['cliffs_delta']:.2f}")
        print()
        print("  Per-training-seed Cliff's δ:")
        for i, d in enumerate(ms['per_seed_deltas']):
            print(f"    Seed {i}: δ = {d:.2f}")
        print()
    
    # Exp H results
    if 'exp_h' in all_results and 'error' not in all_results['exp_h']:
        eh = all_results['exp_h']
        print("CROSS-DOMAIN × CROSS-ADAPTER (EXP H)")
        print("-" * 70)
        print(f"  {'Domain':<12} {'Adapter':<12} {'SAINT':>8} {'BRUTE':>8} {'Gap':>10} {'δ':>8}")
        print("  " + "-" * 58)
        for cell in eh['cells']:
            print(f"  {cell['domain']:<12} {cell['adapter']:<12} "
                  f"{cell['saint_harm']:>7.1%} {cell['brute_harm']:>7.1%} "
                  f"{cell['gap']:>+9.1%} {cell['cliffs_delta']:>+7.2f}")
        print()
        print(f"  Minimum gap: {eh['summary']['min_gap']:.1%}")
        print(f"  All |δ| > 0.8: {eh['summary']['all_effects_large']}")
        print()
    
    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
