"""API - FastAPI Backend."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent.orchestrator import AgenticCopilot
from models.schema import NewsArticle, RiskAssessment, UserConfig

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI(
    title="Agentic AI Supply Chain Copilot API",
    description="API for the agentic AI copilot that monitors logistics news and proposes actions",
    version="0.1.0",
)

copilot: AgenticCopilot | None = None
business_profile_cache: UserConfig | None = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BUSINESS_PROFILE_PATH = DATA_DIR / "config" / "business_profile.json"
LEGACY_CONFIG_PATH = PROJECT_ROOT / "config.json"

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


def _load_business_profile() -> UserConfig:
    """Load the business profile from disk."""

    global business_profile_cache
    if business_profile_cache is not None:
        return business_profile_cache

    for path in (BUSINESS_PROFILE_PATH, LEGACY_CONFIG_PATH):
        if path.exists():
            business_profile_cache = UserConfig.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            return business_profile_cache

    raise HTTPException(status_code=404, detail="Business profile not found")


def _save_business_profile(profile: UserConfig) -> UserConfig:
    """Persist the business profile to both supported config files."""

    global business_profile_cache
    business_profile_cache = profile
    payload = profile.model_dump_json(indent=2)
    write_errors: list[str] = []

    try:
        BUSINESS_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        write_errors.append(f"{BUSINESS_PROFILE_PATH.parent}: {exc}")

    for path, writer in (
        (LEGACY_CONFIG_PATH, lambda p: p.write_text(json.dumps(profile.model_dump(mode="json"), indent=2), encoding="utf-8")),
        (BUSINESS_PROFILE_PATH, lambda p: p.write_text(payload, encoding="utf-8")),
    ):
        try:
            writer(path)
        except Exception as exc:
            write_errors.append(f"{path}: {exc}")

    if len(write_errors) == 2:
        raise HTTPException(
            status_code=500,
            detail="Could not save business profile to disk: " + "; ".join(write_errors),
        )

    return profile


def _get_copilot() -> AgenticCopilot:
    """Create the copilot lazily so config endpoints can run without the vector store."""

    global copilot
    if copilot is None:
        copilot = AgenticCopilot()
    return copilot

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/run-cycle")
async def run_agent_cycle():
    """Manually trigger one agent cycle."""
    try:
        assessments = await _get_copilot().run_cycle()
        return {
            "status": "success",
            "assessments_count": len(assessments),
            "assessments": [a.model_dump() for a in assessments],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/assessments")
@app.get("/assessments")
async def get_assessments(limit: int = 10):
    """Get the most recent risk assessments."""
    assessments = await _get_copilot().get_latest_assessments(limit)
    return [a.model_dump() for a in assessments]


@app.get("/api/assessments/{news_id}")
@app.get("/assessments/{news_id}")
async def get_assessment(news_id: str):
    """Get risk assessment for a specific news article."""
    assessment = await _get_copilot().get_assessment_for_news(news_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment.model_dump()


@app.get("/api/news")
@app.get("/news")
async def get_news(limit: int = 20):
    """Get recent news articles."""
    return []


@app.get("/api/config")
async def get_config():
    """Return the current business profile."""
    return _load_business_profile().model_dump(mode="json")


@app.put("/api/config")
async def update_config(profile: UserConfig):
    """Update the business profile used by the agent."""
    updated = _save_business_profile(profile)

    # Refresh the in-memory profile so the next cycle uses the new rules immediately.
    if copilot is not None:
        copilot.business_profile = updated
    return updated.model_dump(mode="json")


# Serve frontend
@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "frontend", "index.html"
    )
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"error": "Frontend not found. Please run from project root."}
