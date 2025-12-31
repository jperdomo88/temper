#!/bin/bash
# Exp M: Machiavelli Behavioral Transfer
# Paper Section 9.6
#
# Tests: Do bred behaviors transfer to text games?
# Expect: ~28% harm rate (21+ pp reduction from baseline)

cd "$(dirname "$0")/../validation"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=============================================="
echo "EXP M: MACHIAVELLI TRANSFER"
echo "Paper Section 9.6"
echo "=============================================="
echo ""
echo "Tests behavioral transfer to text-based games"
echo "Expect: ~28% harm rate (vs 50% baseline)"
echo ""
cd machiavelli
python3 run_transfer_test.py --kernel kernels/saint_kernel.json

echo ""
echo "Press any key to close..."
read -n 1
