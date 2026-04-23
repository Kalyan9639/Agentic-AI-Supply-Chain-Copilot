@echo off

REM Agentic AI Supply Chain Copilot - Start Script (Windows)

REM Activate virtual environment
call .venv\Scripts\activate

REM Check if .env exists
if not exist .env (
    echo ERROR: .env file not found!
    echo Please copy .env.example to .env and configure it.
    exit /b 1
)

REM Check if HF_API_KEY is set
findstr /c:"your-huggingface-api-key-here" .env >nul
if %errorlevel% equ 0 (
    echo ERROR: HF_API_KEY not configured in .env
    echo Please update .env with your Hugging Face API key.
    exit /b 1
)

REM Create directories if needed
if not exist data\chromadb mkdir data\chromadb
if not exist data\config mkdir data\config
if not exist data\cache mkdir data\cache

echo Starting Agentic AI Supply Chain Copilot...
echo.

REM Run the agent
python src\main.py %*
