#!/bin/bash
# Run Tempered RLHF Experiment v2 - Complete Package

echo "========================================"
echo "TEMPERED RLHF EXPERIMENT v2"
echo "========================================"
echo ""
echo "This runs ALL conditions:"
echo "  - Proxy (baseline)"
echo "  - Tempered (our mechanism)"
echo "  - Oracle (ceiling)"
echo "  - Ablation (visible fitness)"
echo "  - CCD/Laundering validation"
echo "  - Born-gamed analysis"
echo ""

cd "$(dirname "$0")"

# Detect CPU cores for parallelization
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$((CORES / 2))
if [ $WORKERS -lt 2 ]; then WORKERS=2; fi
if [ $WORKERS -gt 8 ]; then WORKERS=8; fi

echo "Detected $CORES cores, using $WORKERS parallel workers"
echo ""

# Default: 5 seeds with auto-detected workers
# Add --fast for quick test (~5 min)
# Add --seeds 8 for full run

python3 src/run_all_v2.py --workers $WORKERS "$@"

echo ""
echo "========================================"
echo "Experiment complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/  (JSON data)"
echo "  - figures/  (plots)"
echo ""
read -p "Press Enter to close..."
