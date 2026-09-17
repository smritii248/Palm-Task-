"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.ingestion import router as ingestion_router
from app.core.logging import configure_logging
from app.db.database import init_db

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup/shutdown tasks: initialize the SQL schema."""
    logger.info("Starting up: initializing database schema")
    init_db()
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="PalmMind RAG Backend",
        description="Document ingestion + conversational RAG with interview booking.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(ingestion_router)
    app.include_router(chat_router)

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
