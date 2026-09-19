#!/bin/bash
# Install Ollama locally - fully local LLM, no cloud
echo "Installing Ollama for local LLM..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "On Mac, install via: brew install ollama"
    echo "Or download from https://ollama.com"
else
    echo "Windows: download from https://ollama.com"
fi

echo "Starting Ollama..."
ollama serve &
sleep 3

echo "Pulling models..."
ollama pull llama3.1:8b
ollama pull mistral:7b || true

echo "Ollama ready at http://localhost:11434"
echo "Test: ollama run llama3.1:8b 'What is fair value in prediction markets?'"
