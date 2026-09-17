"""Request/response models for the document ingestion API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategyName(str, Enum):
    """Selectable chunking strategies."""

    FIXED = "fixed"
    SENTENCE_WINDOW = "sentence_window"


class DocumentUploadResponse(BaseModel):
    """Response returned after a document has been ingested."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    chunking_strategy: ChunkingStrategyName
    chunk_count: int = Field(..., description="Number of chunks created and embedded")
    vector_collection: str
    created_at: datetime


class ChunkPreview(BaseModel):
    """A lightweight preview of a stored chunk, used in retrieval responses."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float | None = Field(default=None, description="Similarity score, if from a search result")
