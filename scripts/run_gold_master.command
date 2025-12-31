#!/bin/bash
# TEMPER Gold Master - Comprehensive Test Suite
# Runs 6 phases covering most paper experiments
#
# INCLUDED IN THIS TEST:
#   Phase 1: Kernel breeding (SAINT, BRUTE)
#   Phase 2: Separation experiment (Exp 1 - Defense-in-Depth)
#   Phase 3: Noise robustness (Exp A)
#   Phase 4: CCD detection (Exp 3)
#   Phase 5: Zeno detection (Exp 4)
#   Phase 6: Linkage detection (Exp D)
#
# NOT INCLUDED (run separately):
#   - Ablation (run_ablation) - Core thesis
#   - Cross-Domain Robustness (run_cross_domain) - Exp H
#   - Machiavelli (run_machiavelli) - Exp M
#   - RLHF (tempered_rlhf/) - Requires PyTorch

cd "$(dirname "$0")/../validation"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=============================================="
echo "TEMPER GOLD MASTER - COMPREHENSIVE TEST"
echo "=============================================="
echo ""
echo "This runs 6 phases (~5-10 minutes):"
echo "  1. Kernel breeding"
echo "  2. Separation (Exp 1)"
echo "  3. Noise robustness (Exp A)"
echo "  4. CCD detection (Exp 3)"
echo "  5. Zeno detection (Exp 4)"
echo "  6. Linkage detection (Exp D)"
echo ""
python3 gold_master_hostile.py --seeds 10

echo ""
echo "Press any key to close..."
read -n 1
