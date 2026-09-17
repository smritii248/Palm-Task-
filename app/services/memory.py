"""Redis-backed conversational memory: stores turn history and in-progress booking
slots per session, both with a TTL so idle sessions clean themselves up.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

import redis

from app.config import get_settings
from app.schemas.booking import BookingSlots

settings = get_settings()
_redis = redis.from_url(settings.redis_url, decode_responses=True)

_HISTORY_KEY_TMPL = "chat:history:{session_id}"
_BOOKING_KEY_TMPL = "chat:booking_slots:{session_id}"


@dataclass(frozen=True)
class Turn:
    """A single message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


def append_turn(session_id: str, role: Literal["user", "assistant"], content: str) -> None:
    """Append a turn to the session's history, trimming to the configured max length."""
    key = _HISTORY_KEY_TMPL.format(session_id=session_id)
    turn = Turn(role=role, content=content)
    _redis.rpush(key, json.dumps(asdict(turn)))
    _redis.ltrim(key, -settings.chat_memory_max_turns, -1)
    _redis.expire(key, settings.chat_memory_ttl_seconds)


def get_history(session_id: str) -> list[Turn]:
    """Return the full stored turn history for a session, oldest first."""
    key = _HISTORY_KEY_TMPL.format(session_id=session_id)
    raw_turns = _redis.lrange(key, 0, -1)
    return [Turn(**json.loads(raw)) for raw in raw_turns]


def get_booking_slots(session_id: str) -> BookingSlots:
    """Return any partially-filled booking slots collected so far for this session."""
    key = _BOOKING_KEY_TMPL.format(session_id=session_id)
    raw = _redis.get(key)
    if raw is None:
        return BookingSlots()
    return BookingSlots(**json.loads(raw))


def save_booking_slots(session_id: str, slots: BookingSlots) -> None:
    """Persist partially-filled booking slots for this session."""
    key = _BOOKING_KEY_TMPL.format(session_id=session_id)
    _redis.set(key, slots.model_dump_json(), ex=settings.chat_memory_ttl_seconds)


def clear_booking_slots(session_id: str) -> None:
    """Clear booking slots once a booking has been completed and persisted."""
    key = _BOOKING_KEY_TMPL.format(session_id=session_id)
    _redis.delete(key)
