#!/usr/bin/env bash
# One-click launcher for MedicalAI on Linux/Mac
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check Python ──
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1)
        if echo "$ver" | grep -qE "Python 3\.(1[0-9]|[0-9]+)"; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.10+ is required but not found."
    echo "Install from: https://www.python.org/downloads/"
    read -rp "Press Enter to exit"
    exit 1
fi
echo "Using: $($PYTHON --version)"

# ── Virtual Environment ──
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

source .venv/bin/activate

# ── Install Dependencies ──
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt 2>/dev/null || pip install -r requirements.txt
fi

# ── Check / Generate Default Models ──
if [ ! -f "checkpoints/fusion_model.pth" ] && \
   [ ! -f "checkpoints/onnx_full/fusion_full.onnx" ] && \
   [ ! -f "models/default/fusion_classifier.onnx" ]; then
    echo "No models found. Generating default models..."
    python setup_default.py
    echo "Default models generated. The app will work immediately."
fi

if [ -f "checkpoints/fusion_model.pth" ]; then
    echo "Trained model found."
elif [ -f "checkpoints/onnx_full/fusion_full.onnx" ]; then
    echo "ONNX pipeline found."
elif [ -f "models/default/fusion_classifier.onnx" ]; then
    echo "Default model found. Train for accurate results."
fi

# ── Launch ──
echo ""
echo "MedicalAI - Light Weight"
echo "========================"
echo "1) Web UI (recommended - opens in browser)"
echo "2) Command-line interface (CLI)"
echo "3) API Server (for website/app integration)"
echo ""
read -rp "Select (1, 2, or 3): " choice

if [ "$choice" = "2" ]; then
    echo "Launching CLI..."
    python run.py
elif [ "$choice" = "3" ]; then
    echo "Launching API server on http://127.0.0.1:8000 ..."
    echo "Your website can connect to: http://127.0.0.1:8000/api"
    pip install -q "fastapi[standard]" uvicorn 2>/dev/null || pip install "fastapi[standard]" uvicorn
    python quantization.py --mode serve-api
else
    if python -c "import gradio" 2>/dev/null; then
        python web_ui.py
    else
        echo "Installing gradio..."
        pip install gradio
        python web_ui.py
    fi
fi
