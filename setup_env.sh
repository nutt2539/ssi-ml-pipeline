#!/bin/bash
# ==============================================================================
# One-Click Setup Script for SSI Machine Learning Research Pipeline
# Somdech Phra Pinklao Hospital & Department of Surgery
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "🏥 Setting up Python Virtual Environment for SSI Machine Learning Pipeline"
echo "=============================================================================="

# 1. Create virtualenv if not exists
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Activate virtual environment
echo "[*] Activating virtual environment..."
source .venv/bin/activate

# 3. Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip

# 4. Install dependencies
echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "=============================================================================="
echo "✅ Environment setup complete!"
echo "   To run the pipeline:   python run_pipeline.py"
echo "   To launch web app:     streamlit run app/streamlit_app.py"
echo "=============================================================================="
