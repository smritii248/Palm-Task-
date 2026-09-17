"""Centralized, typed application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings. See .env.example for all supported keys."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM / embeddings
    openai_api_key: str = "sk-placeholder"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chat_model: str = "gpt-4o-mini"

    # SQL database
    database_url: str = "sqlite:///./palmmind.db"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "palmmind_documents"

    # Redis chat memory
    redis_url: str = "redis://localhost:6379/0"
    chat_memory_ttl_seconds: int = 86400
    chat_memory_max_turns: int = 20

    # Chunking
    default_chunk_size: int = 800
    default_chunk_overlap: int = 120

    # RAG
    rag_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
