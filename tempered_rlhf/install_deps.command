#!/bin/bash
# Install dependencies for Tempered RLHF Experiment

echo "========================================"
echo "Installing dependencies..."
echo "========================================"

cd "$(dirname "$0")"

pip3 install -r requirements.txt

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "You can now run the experiment with:"
echo "  Double-click run_experiment.command"
echo ""
read -p "Press Enter to close..."
