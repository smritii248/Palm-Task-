"""Custom retrieval-augmented generation pipeline.

Deliberately hand-rolled (no RetrievalQAChain / LangChain chain abstractions): each
step - embed query, retrieve, load memory, build prompt, generate, persist memory -
is an explicit function call the caller can see and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Booking
from app.schemas.booking import BookingResponse, BookingSlots
from app.schemas.chat import ChatResponse
from app.schemas.document import ChunkPreview
from app.services import booking_agent, memory
from app.services.embeddings import embed_text
from app.services.vector_store import search as vector_search

settings = get_settings()
_client = OpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = (
    "You are PalmMind's assistant. Answer the user's question using ONLY the provided "
    "context chunks when they are relevant. If the context doesn't contain the answer, "
    "say you don't have that information rather than guessing. Be concise. If the user "
    "wants to book an interview, guide them conversationally to provide their name, "
    "email, preferred date, and preferred time."
)


@dataclass(frozen=True)
class RetrievedContext:
    chunks: list[ChunkPreview]

    def as_prompt_block(self) -> str:
        if not self.chunks:
            return "(no relevant context retrieved)"
        return "\n\n".join(f"[Source {i + 1}]\n{c.text}" for i, c in enumerate(self.chunks))


def retrieve(query: str, collection_name: str, top_k: int | None = None) -> RetrievedContext:
    """Embed the query and retrieve the top-k most similar chunks from the vector store."""
    query_vector = embed_text(query)
    hits = vector_search(collection_name, query_vector, top_k=top_k or settings.rag_top_k)
    chunks = [
        ChunkPreview(
            chunk_id=hit.vector_id,
            document_id=str(hit.payload.get("document_id", "")),
            chunk_index=int(hit.payload.get("chunk_index", 0)),
            text=str(hit.payload.get("text", "")),
            score=hit.score,
        )
        for hit in hits
    ]
    return RetrievedContext(chunks=chunks)


def _build_messages(
    history: list[memory.Turn], context: RetrievedContext, user_message: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context.as_prompt_block()}\n\nQuestion: {user_message}",
        }
    )
    return messages


def _generate_reply(messages: list[dict[str, str]]) -> str:
    response = _client.chat.completions.create(model=settings.chat_model, messages=messages)
    return response.choices[0].message.content or ""


def _persist_booking(db: Session, session_id: str, slots: BookingSlots) -> BookingResponse:
    booking = Booking(
        session_id=session_id,
        name=slots.name or "",
        email=slots.email or "",
        interview_date=slots.date or "",
        interview_time=slots.time or "",
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    memory.clear_booking_slots(session_id)
    return BookingResponse.model_validate(booking)


def handle_chat_turn(
    db: Session,
    session_id: str,
    user_message: str,
    vector_collection: str,
) -> ChatResponse:
    """Run one full conversational RAG turn: booking check, retrieval, generation, memory."""
    history = memory.get_history(session_id)

    # 1. Check booking intent / extract any new slot values from this message.
    wants_to_book, new_slots = booking_agent.extract_booking_intent(user_message)
    existing_slots = memory.get_booking_slots(session_id)
    merged_slots = booking_agent.merge_slots(existing_slots, new_slots)

    booking_response: BookingResponse | None = None

    if wants_to_book or any([existing_slots.name, existing_slots.email, existing_slots.date, existing_slots.time]):
        if merged_slots.is_complete():
            booking_response = _persist_booking(db, session_id, merged_slots)
            reply = (
                f"You're all set, {merged_slots.name}! Your interview is booked for "
                f"{merged_slots.date} at {merged_slots.time}. A confirmation will go to "
                f"{merged_slots.email}."
            )
            memory.append_turn(session_id, "user", user_message)
            memory.append_turn(session_id, "assistant", reply)
            return ChatResponse(session_id=session_id, reply=reply, sources=[], booking=booking_response)

        memory.save_booking_slots(session_id, merged_slots)
        reply = booking_agent.missing_fields_prompt(merged_slots)
        memory.append_turn(session_id, "user", user_message)
        memory.append_turn(session_id, "assistant", reply)
        return ChatResponse(session_id=session_id, reply=reply, sources=[], booking=None)

    # 2. Normal RAG path: retrieve context, build prompt, generate, persist memory.
    context = retrieve(user_message, vector_collection)
    messages = _build_messages(history, context, user_message)
    reply = _generate_reply(messages)

    memory.append_turn(session_id, "user", user_message)
    memory.append_turn(session_id, "assistant", reply)

    return ChatResponse(session_id=session_id, reply=reply, sources=context.chunks, booking=None)
