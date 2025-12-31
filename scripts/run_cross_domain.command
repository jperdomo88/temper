#!/bin/bash
# Exp H: Cross-Domain Robustness Test
# Paper Section 9.4
# NOT included in Gold Master - run separately
#
# WHAT IT PROVES:
# SAINT/BRUTE behavioral separation is a genuine KERNEL property, not:
#   - A gridworld quirk (separation holds in INFO and COMMITMENT domains)
#   - An adapter artifact (separation holds with independent Adapter B)
#
# Tests: 2 domains × 2 adapters = 4 conditions
# Pass criteria: δ ≤ -0.8 in ALL FOUR conditions

cd "$(dirname "$0")/../validation"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=============================================="
echo "EXP H: CROSS-DOMAIN ROBUSTNESS TEST"
echo "Paper Section 9.4"
echo "=============================================="
echo ""
echo "Proves SAINT/BRUTE separation is a genuine kernel property:"
echo "  - Not a gridworld quirk (tests INFO + COMMITMENT domains)"
echo "  - Not an adapter artifact (tests Adapter A + Adapter B)"
echo "  - Must pass ALL FOUR conditions"
echo ""
python3 -c "
from crucible.experiments.exp_h_combined import run_exp_h, ExpHConfig
config = ExpHConfig(n_seeds=10)
results = run_exp_h(config)
"

echo ""
echo "Press any key to close..."
read -n 1
