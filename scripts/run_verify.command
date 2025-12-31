#!/bin/bash
# TEMPER Verification Script (Mac/Linux)
# Double-click to run

cd "$(dirname "$0")/../validation"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=============================================="
echo "TEMPER LOCAL VERIFICATION"
echo "=============================================="
python3 verify_local.py

echo ""
echo "Press any key to close..."
read -n 1
