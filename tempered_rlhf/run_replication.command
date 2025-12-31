#!/bin/bash
# Registered Replication - Tempered RLHF Experiment
# 
# PURPOSE: Run identical protocol with N=15 seeds (vs prereg N=5)
# This provides stronger statistical power while maintaining methodological integrity.
# Report BOTH results in paper: "Prereg (N=5)" and "Replication (N=15)"
#
# GPT's guidance: "Add more seeds, same everything, same analysis"
# - Same training steps, population size, generations
# - Same master seed derivation
# - Only difference: more independent seeds

echo "========================================"
echo "TEMPERED RLHF - REGISTERED REPLICATION"
echo "========================================"
echo ""
echo "Running N=15 seeds (vs prereg N=5)"
echo "Same protocol, stronger inference"
echo ""
echo "Expected runtime: ~45-60 min on 14-core machine"
echo ""

cd "$(dirname "$0")"

# Detect CPU cores for parallelization
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$((CORES / 2))
if [ $WORKERS -lt 2 ]; then WORKERS=2; fi
if [ $WORKERS -gt 8 ]; then WORKERS=8; fi

echo "Detected $CORES cores, using $WORKERS parallel workers"
echo ""

# Run with 15 seeds - everything else identical to prereg
python3 src/run_all_v2.py --seeds 15 --workers $WORKERS "$@"

echo ""
echo "========================================"
echo "Replication complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/  (JSON data with N=15)"
echo "  - figures/  (updated plots)"
echo ""
echo "For paper, report:"
echo "  - Prereg (N=5): original run_experiment.command results"
echo "  - Replication (N=15): these results"
echo ""
read -p "Press Enter to close..."
