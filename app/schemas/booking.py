"""Request/response models for interview booking."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class BookingSlots(BaseModel):
    """Partially or fully filled booking fields, as extracted by the LLM tool call."""

    name: str | None = None
    email: str | None = None
    date: str | None = Field(default=None, description="Requested interview date, e.g. '2026-09-20'")
    time: str | None = Field(default=None, description="Requested interview time, e.g. '15:00'")

    def is_complete(self) -> bool:
        return all([self.name, self.email, self.date, self.time])


class BookingResponse(BaseModel):
    """A persisted booking record."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: EmailStr
    interview_date: str
    interview_time: str
    created_at: datetime
