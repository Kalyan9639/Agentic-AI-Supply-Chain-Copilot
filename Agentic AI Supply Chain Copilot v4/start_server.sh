#!/bin/bash

# Agentic AI Supply Chain Copilot - Start API Server

# Activate virtual environment
source .venv/Scripts/activate

echo Starting FastAPI Server...
echo.

# Run the API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
