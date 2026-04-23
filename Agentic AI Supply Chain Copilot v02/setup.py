#!/usr/bin/env python3
"""
Setup script for Agentic AI Supply Chain Copilot.
Initializes the project structure and validates configuration.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set."""
    errors = []

    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key or hf_api_key == "your-huggingface-api-key-here":
        errors.append(
            "HF_API_KEY is not set. Please add your Hugging Face API key to .env"
        )

    if errors:
        print("Configuration issues found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these issues before running the agent.")
        return False

    print("Environment configuration: OK")
    return True


def setup_directories():
    """Create required directories."""
    base = Path(__file__).parent
    directories = [
        "data/chromadb",
        "data/config",
        "data/cache",
    ]

    print("Setting up directories...")
    for dir_path in directories:
        full_path = base / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")


def create_readme():
    """Create a README if it doesn't exist."""
    readme_path = Path(__file__).parent / "README.md"
    if not readme_path.exists():
        readme_content = """# Agentic AI Supply Chain Copilot

An AI-powered agent that monitors Indian logistics news, assesses risks to your business, and proposes actionable solutions.

## Quick Start

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure `.env` with your Hugging Face API key

3. Run the agent:
   ```bash
   python src/main.py --mode scheduler
   ```

4. Start the web API:
   ```bash
   python src/run_server.py
   ```

## How It Works

1. **Watch**: Crawl4AI scrapes ITLN news every 15 minutes
2. **Store**: Embeddings saved to ChromaDB vector store
3. **Think**: Gemma model analyzes news for business risks
4. **Act**: Proposes actions for human approval

## Tech Stack

- **LLM**: google/gemma-4-26b-a4b-it (Hugging Face)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (local)
- **Scraping**: Crawl4AI
- **Backend**: FastAPI
"""
        readme_path.write_text(readme_content)
        print("Created README.md")


def main():
    """Main setup function."""
    print("=" * 50)
    print("Agentic AI Supply Chain Copilot Setup")
    print("=" * 50)
    print()

    # Check environment first
    if not check_environment():
        print("\nSetup incomplete. Please fix configuration issues.")
        return 1

    # Setup directories
    setup_directories()

    # Create README
    create_readme()

    print()
    print("=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Run agent:   python src/main.py --mode scheduler")
    print("  2. Start API:   python src/run_server.py")
    print("  3. View UI:     Open http://localhost:8000")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
