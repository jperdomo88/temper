#!/bin/bash
# Exp 0: Fitness Ablation - Core Thesis
# Paper Section 9.1
# NOT included in Gold Master - run separately
#
# Proves: Low harm ≠ alignment
# SURVIVAL_ONLY: 2.2% harm but only 7% protection
# NORMATIVE: 7.6% harm but 57.5% protection

cd "$(dirname "$0")/../validation"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=============================================="
echo "EXP 0: FITNESS ABLATION (Core Thesis)"
echo "Paper Section 9.1"
echo "=============================================="
echo ""
echo "This is the core thesis proof:"
echo "  - Low harm alone does NOT mean alignment"
echo "  - You need normative selection"
echo ""
python3 ablation_fitness.py --kernels 5 --evals 5

echo ""
echo "Press any key to close..."
read -n 1
