"""Uses LLM function/tool calling to detect booking intent and extract structured
interview-booking slots (name, email, date, time) from free-form conversation,
merging newly extracted fields with any already collected for the session.
"""

from __future__ import annotations

import json

from openai import OpenAI

from app.config import get_settings
from app.schemas.booking import BookingSlots

settings = get_settings()
_client = OpenAI(api_key=settings.openai_api_key)

_EXTRACT_BOOKING_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_booking_slots",
        "description": (
            "Extract any interview-booking details (full name, email, date, time) the user "
            "has provided in their latest message. Only include fields explicitly stated or "
            "clearly inferable from this message; omit fields that were not mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The user's full name"},
                "email": {"type": "string", "description": "The user's email address"},
                "date": {"type": "string", "description": "Requested interview date, ISO format if possible"},
                "time": {"type": "string", "description": "Requested interview time, 24h format if possible"},
                "wants_to_book": {
                    "type": "boolean",
                    "description": "True if the user is trying to schedule/book an interview",
                },
            },
            "required": ["wants_to_book"],
        },
    },
}


def extract_booking_intent(user_message: str) -> tuple[bool, BookingSlots]:
    """Ask the LLM whether the message expresses booking intent and to extract slots.

    Returns:
        A tuple of (wants_to_book, extracted_slots). extracted_slots only contains
        fields explicitly present in this message; merge with prior session state
        in the caller.
    """
    response = _client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You detect interview-booking intent in user messages and extract any "
                    "booking details present. Call the extract_booking_slots tool exactly once."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        tools=[_EXTRACT_BOOKING_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_booking_slots"}},
    )

    message = response.choices[0].message
    if not message.tool_calls:
        return False, BookingSlots()

    args = json.loads(message.tool_calls[0].function.arguments)
    wants_to_book = bool(args.get("wants_to_book", False))
    slots = BookingSlots(
        name=args.get("name"),
        email=args.get("email"),
        date=args.get("date"),
        time=args.get("time"),
    )
    return wants_to_book, slots


def merge_slots(existing: BookingSlots, new: BookingSlots) -> BookingSlots:
    """Merge newly extracted fields on top of previously collected ones."""
    return BookingSlots(
        name=new.name or existing.name,
        email=new.email or existing.email,
        date=new.date or existing.date,
        time=new.time or existing.time,
    )


def missing_fields_prompt(slots: BookingSlots) -> str:
    """Return a natural-language prompt asking the user for whatever is still missing."""
    missing = []
    if not slots.name:
        missing.append("your full name")
    if not slots.email:
        missing.append("your email address")
    if not slots.date:
        missing.append("a preferred interview date")
    if not slots.time:
        missing.append("a preferred interview time")
    return "Sure — could you share " + ", ".join(missing) + "?"
