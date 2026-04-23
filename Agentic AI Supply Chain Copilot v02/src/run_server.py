"""Run the FastAPI server."""
import asyncio
import os

import uvicorn

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        reload=False,
        log_level="info",
    )
