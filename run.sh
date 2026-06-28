#!/usr/bin/env bash
# PolyWeather — Startup Script
# Activates the virtual environment, launches the Streamlit UI,
# and starts the autonomous background trading daemon.

set -e

echo "=================================================="
echo "🌤️  Starting PolyWeather Autonomous Trading System"
echo "=================================================="

# 1. Activate virtual environment
VENV_PATH="/home/sunilrajd/disk/code/venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "→ Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 2. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠ Warning: .env file not found. Copying from template..."
    cp .env.template .env || touch .env
fi

# 3. Launch Streamlit dashboard in the background
echo "→ Launching Streamlit dashboard..."
streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "  ✓ Dashboard running on PID $STREAMLIT_PID"

# 4. Wait a moment for Streamlit to initialize
sleep 2

echo "=================================================="
echo "📊 PolyWeather UI is live at: http://localhost:8501"
echo "=================================================="
echo ""
echo "→ Starting Hermes Agent Daemon (Press Ctrl+C to stop)..."
echo ""

# 5. Run the background daemon in the foreground
python main.py daemon

# Cleanup when daemon exits
kill $STREAMLIT_PID 2>/dev/null || true
echo "PolyWeather shutdown complete."
