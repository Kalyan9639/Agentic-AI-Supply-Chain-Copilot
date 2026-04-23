# Agentic AI Supply Chain Copilot

An AI-powered agent that monitors Indian logistics news, checks it against your business profile and past incidents, and proposes practical actions using Gemma via Hugging Face.

## What This Project Does

The copilot runs in a loop:

1. Scrape logistics news from the configured source pages.
2. Convert each article into embeddings for duplicate detection and memory search.
3. Store the article in ChromaDB if it is new.
4. Send the article, your business profile, and similar incidents to Gemma.
5. Parse Gemma's answer into a structured risk assessment.
6. Save the assessment and expose it through the API and frontend.

## Data Flow

```text
ITLN news pages
    -> Crawl4AI scraper
    -> NewsArticle object
    -> SentenceTransformer embeddings
    -> ChromaDB news memory
    -> Similar incident lookup
    -> Gemma on Hugging Face
    -> RiskAssessment object
    -> ChromaDB assessment memory
    -> FastAPI endpoints
    -> Frontend dashboard
```

### Storage and Config Locations

- `data/config/business_profile.json` is the primary business profile used by the agent.
- `config.json` is also supported as a fallback business profile source.
- `data/cache/chromadb` stores the local ChromaDB persistence files.
- `data/cache` is also used as the local cache base for Crawl4AI.

## Project Structure

```text
src/
  api/       FastAPI backend
  agent/     Agent orchestration and Gemma integration
  scraper/   Crawl4AI news scraping
  memory/    ChromaDB vector store
  models/    Pydantic schemas
  utils/     Logging and embeddings
  main.py    CLI entry point
frontend/    Web dashboard
data/        Business profile and local storage
```

## Requirements

- Python 3.11+
- Hugging Face API key in `.env`
- A virtual environment with the project dependencies installed

## Environment Setup

Copy `.env.example` to `.env` and set your Hugging Face key:

```env
HF_API_KEY=your-huggingface-api-key
SCRAPE_INTERVAL_MINUTES=15
LOG_LEVEL=INFO
DEBUG=False
```

## Install Dependencies

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -r requirements.txt
```

## Ways to Run the Project

### 1. Run one agent cycle

Use this for a quick test of scraping, memory lookup, and Gemma inference:

```bash
python src/main.py --mode single
```

### 2. Run the continuous scheduler

Use this to keep the agent running on a loop every `SCRAPE_INTERVAL_MINUTES` minutes:

```bash
python src/main.py --mode scheduler
```

### 3. Run the FastAPI server

Use this when you want the API and frontend dashboard:

```bash
python src/run_server.py
```

Then open:

- `http://localhost:8000/` for the dashboard
- `http://localhost:8000/health` for a health check
- `http://localhost:8000/run-cycle` to trigger a cycle manually

### 4. Use the helper scripts

Windows:

```bat
start.bat
start_server.bat
```

Linux/macOS:

```bash
./start.sh
./start_server.sh
```

The start scripts:

- activate the virtual environment
- verify that `.env` exists
- verify that `HF_API_KEY` is configured
- create the required local data directories
- launch the agent or the API server

## API Endpoints

- `GET /health` - service health check
- `POST /run-cycle` - run one full agent cycle
- `GET /assessments` - list recent risk assessments
- `GET /assessments/{news_id}` - fetch a specific assessment
- `GET /news` - placeholder endpoint for recent news output

## How the Agent Thinks

The agent uses:

- Gemma as the reasoning model
- local embeddings for semantic search
- ChromaDB for short-term and historical memory
- a business profile file so decisions stay specific to your operation

For each article, the prompt includes:

- your business name
- your commodity
- your region
- your business rules
- the news article content
- similar incidents from memory

Gemma returns a structured result:

- `risk_level`
- `reasoning`
- `proposed_action`
- `confidence`

## Tech Stack

- **LLM**: `google/gemma-4-26B-A4B-it` via `huggingface_hub`
- **Embeddings**: `sentence-transformers`
- **Vector DB**: ChromaDB
- **Scraping**: Crawl4AI
- **Backend**: FastAPI
- **Validation**: Pydantic

## Notes

- The app defaults to the business profile at `data/config/business_profile.json`.
- ChromaDB and Crawl4AI both use the `data/cache` subtree so local state stays inside the project.
- If Hugging Face requests fail, check that the API key in `.env` is valid and that your account can access the chosen Gemma model.

## License

MIT License
