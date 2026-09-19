#!/bin/bash
set -e

echo "=== PTAI Setup - Local Autonomous Trading Agent ==="

# Check python
python3 --version || { echo "Python3 required"; exit 1; }

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing Playwright..."
playwright install chromium
playwright install-deps chromium || true

echo "Creating directories..."
mkdir -p data logs browser/profiles/default config

# Copy env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from example - EDIT IT!"
fi

if [ ! -f "config/config.yaml" ]; then
    cp config/config.yaml.example config/config.yaml
    echo "Created config.yaml from example"
fi

echo ""
echo "=== Checking Ollama (local LLM) ==="
if command -v ollama &> /dev/null; then
    echo "Ollama found"
    echo "Pulling model llama3.1:8b (this may take a while)..."
    ollama pull llama3.1:8b || echo "Failed to pull, try manually: ollama pull llama3.1:8b"
else
    echo "Ollama NOT found - install from https://ollama.com"
    echo "PTAI will work with heuristic fallback, but LLM is recommended"
fi

echo ""
echo "=== Setup Complete ==="
echo "1. Edit .env with your keys (POLYMARKET_PRIVATE_KEY etc) - or keep DRY_RUN=true for testing"
echo "2. Run: source .venv/bin/activate"
echo "3. Run: python -m ptai.cli init --bankroll 50"
echo "4. Run: python -m ptai.cli scan --count 100"
echo "5. Run autonomous: python -m ptai.cli pay-for-yourself 50 --daily-cost 5"
echo ""
echo "For live trading, set DRY_RUN=false and provide POLYMARKET_PRIVATE_KEY"
