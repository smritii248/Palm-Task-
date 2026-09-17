"""Request/response models for the conversational RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.booking import BookingResponse
from app.schemas.document import ChunkPreview


class ChatRequest(BaseModel):
    """A single user turn in a multi-turn conversation."""

    session_id: str = Field(..., description="Stable identifier for the conversation/user")
    message: str = Field(..., min_length=1, description="The user's message")


class ChatResponse(BaseModel):
    """The assistant's reply to a chat turn, plus retrieval and booking context."""

    session_id: str
    reply: str
    sources: list[ChunkPreview] = Field(default_factory=list)
    booking: BookingResponse | None = Field(
        default=None, description="Populated only if a booking was completed on this turn"
    )
