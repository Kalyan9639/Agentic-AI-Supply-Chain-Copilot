#!/bin/bash

# Agentic AI Supply Chain Copilot - Start Script

# Activate virtual environment
source .venv/Scripts/activate

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

# Check if HF_API_KEY is set
if grep -q "your-huggingface-api-key-here" .env; then
    echo "ERROR: HF_API_KEY not configured in .env"
    echo "Please update .env with your Hugging Face API key."
    exit 1
fi

# Create directories if needed
mkdir -p data/chromadb data/config data/cache

echo "Starting Agentic AI Supply Chain Copilot..."
echo ""

# Run the agent
python src/main.py "$@"
