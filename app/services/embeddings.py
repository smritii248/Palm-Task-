"""Embedding generation, wrapped behind a small provider-agnostic interface.

Only this module needs to change to swap embedding providers.
"""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()
_client = OpenAI(api_key=settings.openai_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed_text(text: str) -> list[float]:
    """Return the embedding vector for a single string."""
    response = _client.embeddings.create(model=settings.embedding_model, input=text)
    return response.data[0].embedding


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a batch of strings, preserving input order."""
    if not texts:
        return []
    response = _client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]
