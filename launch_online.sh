#!/bin/bash
# 1-Click Online Launcher for SSI Risk Predictor
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "🩺 Starting End-of-Surgery SSI Risk Calculator Online..."
echo "=========================================================="

# Check if Streamlit is running, if not start it
if ! lsof -i :8501 > /dev/null 2>&1; then
    echo "🚀 Starting Streamlit Web App on port 8501..."
    env HOME="$DIR" MPLCONFIGDIR="$DIR/.cache" STREAMLIT_GATHER_USAGE_STATS=false \
        .venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 &
    sleep 3
fi

echo "🌐 Creating Secure Public HTTPS Tunnel via Cloudflare..."
"$DIR/bin/cloudflared" tunnel --url http://localhost:8501
