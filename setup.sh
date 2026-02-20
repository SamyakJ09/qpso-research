#!/usr/bin/env bash
# One-command project setup: creates venv, installs package + dev deps
set -e

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate: rm -rf $VENV_DIR && bash setup.sh"
    exit 0
fi

echo "Creating virtual environment..."
python -m venv "$VENV_DIR"

echo "Activating..."
# Detect OS for activation path
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing project in editable mode with dev dependencies..."
pip install -e ".[dev]"

echo ""
echo "Setup complete! Activate with:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "Then run:"
echo "  pytest                                                    # tests"
echo "  python experiments/run_comparison.py --benchmark sphere   # quick demo"
echo ""
echo "For quantum modes (full/hybrid), also run:"
echo "  pip install -e \".[quantum]\""
