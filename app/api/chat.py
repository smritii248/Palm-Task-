"""Conversational RAG API: multi-turn chat with retrieval and interview booking."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.db.models import Booking
from app.schemas.booking import BookingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag import handle_chat_turn

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Handle a single conversational turn: retrieval-augmented answer or booking flow."""
    logger.info("Chat turn for session=%s", request.session_id)
    return handle_chat_turn(
        db=db,
        session_id=request.session_id,
        user_message=request.message,
        vector_collection=settings.qdrant_collection,
    )


@router.get("/bookings/{session_id}", response_model=list[BookingResponse])
def get_bookings(session_id: str, db: Session = Depends(get_db)) -> list[BookingResponse]:
    """Return all bookings captured for a given session."""
    bookings = db.scalars(select(Booking).where(Booking.session_id == session_id)).all()
    if not bookings:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No bookings found for this session")
    return [BookingResponse.model_validate(b) for b in bookings]
