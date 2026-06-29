#!/usr/bin/env bash
# One-command project setup: creates venv, installs package + dev deps
set -e

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate: rm -rf $VENV_DIR && bash setup.sh"
    exit 0
fi

# Detect Python — on Windows (MSYS/Git Bash) try 'py' launcher first
# (immune to Microsoft Store alias), then 'python'. On Unix try 'python3' first.
PYTHON_CMD=""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    if command -v py &>/dev/null; then
        PYTHON_CMD="py"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    fi
else
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "ERROR: Python was not found."
    echo ""
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        echo "If Python is installed but you see a Microsoft Store prompt, fix it:"
        echo "  1. Open Settings > Apps > Advanced app settings > App execution aliases"
        echo "  2. Turn OFF 'python.exe' and 'python3.exe'"
    fi
    echo ""
    echo "If Python is not installed:"
    echo "  1. Download from https://www.python.org/downloads/"
    echo "  2. IMPORTANT: Check 'Add Python to PATH' during installation"
    echo "  3. Re-open this terminal and run setup.sh again"
    exit 1
fi

echo "Using: $PYTHON_CMD"

echo "Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_DIR"

echo "Activating..."
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
echo "  qpso-compare --benchmark sphere --dims 2 --mode math     # quick demo"
echo "  qpso-study --config configs/default.yaml                 # full study"
echo ""
echo "For quantum modes (full/hybrid), also run:"
echo "  pip install -e \".[quantum]\""
