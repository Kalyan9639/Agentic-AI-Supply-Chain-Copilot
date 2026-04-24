@echo off

REM Agentic AI Supply Chain Copilot - Start API Server (Windows)

REM Activate virtual environment
call .venv\Scripts\activate

echo Starting FastAPI Server...
echo.

REM Run the API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
