#!/bin/bash
# Quick sanity check (~1 minute)
# Verifies imports and basic functionality

cd "$(dirname "$0")/../validation"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=============================================="
echo "TEMPER QUICK CHECK"
echo "=============================================="

echo ""
echo "1. Import test..."
python3 -c "from crucible import simulation; print('✅ Imports OK')"

echo ""
echo "2. Determinism check..."
python3 verify_local.py

echo ""
echo "=============================================="
echo "QUICK CHECK COMPLETE"
echo "=============================================="
echo ""
echo "Press any key to close..."
read -n 1
