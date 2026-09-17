"""Thin wrapper around the Qdrant client: create collection, upsert, similarity search.

This is the only module that talks to Qdrant directly. Swapping to Weaviate/Milvus/
Pinecone means reimplementing this module's public functions; nothing else in the
codebase imports qdrant_client directly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import get_settings

settings = get_settings()
_client = QdrantClient(url=settings.qdrant_url)


@dataclass(frozen=True)
class VectorSearchResult:
    """A single similarity-search hit."""

    vector_id: str
    score: float
    payload: dict[str, str | int]


def ensure_collection(collection_name: str = settings.qdrant_collection) -> None:
    """Create the collection if it doesn't already exist. Idempotent."""
    existing = {c.name for c in _client.get_collections().collections}
    if collection_name not in existing:
        _client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )


def upsert_chunks(
    collection_name: str,
    vectors: list[list[float]],
    payloads: list[dict[str, str | int]],
) -> list[str]:
    """Upsert a batch of chunk vectors with metadata payloads.

    Returns:
        The generated vector point IDs, in the same order as the inputs.
    """
    if len(vectors) != len(payloads):
        raise ValueError("vectors and payloads must be the same length")

    ensure_collection(collection_name)
    point_ids = [str(uuid.uuid4()) for _ in vectors]
    points = [
        PointStruct(id=point_id, vector=vector, payload=payload)
        for point_id, vector, payload in zip(point_ids, vectors, payloads, strict=True)
    ]
    _client.upsert(collection_name=collection_name, points=points)
    return point_ids


def search(
    collection_name: str,
    query_vector: list[float],
    top_k: int = 4,
) -> list[VectorSearchResult]:
    """Return the top_k most similar vectors to `query_vector`."""
    ensure_collection(collection_name)
    hits = _client.search(collection_name=collection_name, query_vector=query_vector, limit=top_k)
    return [
        VectorSearchResult(vector_id=str(hit.id), score=hit.score, payload=hit.payload or {})
        for hit in hits
    ]
