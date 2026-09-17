"""API tests exercising the FastAPI app with external services (OpenAI, Qdrant,
Redis) monkey-patched, so the suite runs without any live infrastructure.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.booking import BookingSlots

client = TestClient(app)


@pytest.fixture(autouse=True)
def _patch_db(tmp_path, monkeypatch):
    """Point the app at a throwaway SQLite file for each test."""
    import app.db.database as database_module

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_db_path = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(database_module, "engine", test_engine)
    monkeypatch.setattr(database_module, "SessionLocal", TestSessionLocal)
    database_module.init_db()

    def _override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[database_module.get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document_txt(monkeypatch) -> None:
    monkeypatch.setattr("app.api.ingestion.embed_batch", lambda texts: [[0.1] * 1536 for _ in texts])
    monkeypatch.setattr(
        "app.api.ingestion.upsert_chunks",
        lambda collection, vectors, payloads: [f"vec-{i}" for i in range(len(payloads))],
    )

    file_content = b"This is a test document. " * 50
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", file_content, "text/plain")},
        data={"chunking_strategy": "fixed"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "test.txt"
    assert body["chunk_count"] > 0
    assert body["chunking_strategy"] == "fixed"


def test_upload_document_rejects_unsupported_type() -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.docx", b"binary content", "application/msword")},
    )
    assert response.status_code == 400


def test_upload_document_rejects_empty_file() -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_chat_normal_rag_turn(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.booking_agent.extract_booking_intent",
        lambda message: (False, BookingSlots()),
    )
    monkeypatch.setattr(
        "app.services.rag.retrieve",
        lambda query, collection_name, top_k=None: __import__(
            "app.services.rag", fromlist=["RetrievedContext"]
        ).RetrievedContext(chunks=[]),
    )
    monkeypatch.setattr("app.services.rag._generate_reply", lambda messages: "This is a test answer.")
    monkeypatch.setattr("app.services.memory.get_history", lambda session_id: [])
    monkeypatch.setattr("app.services.memory.append_turn", lambda *a, **k: None)
    monkeypatch.setattr("app.services.memory.get_booking_slots", lambda session_id: BookingSlots())

    response = client.post("/api/v1/chat", json={"session_id": "s1", "message": "What is the leave policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "This is a test answer."
    assert body["booking"] is None


def test_chat_booking_flow_completes(monkeypatch) -> None:
    complete_slots = BookingSlots(name="Smriti", email="smriti@example.com", date="2026-09-20", time="15:00")

    monkeypatch.setattr(
        "app.services.booking_agent.extract_booking_intent",
        lambda message: (True, complete_slots),
    )
    monkeypatch.setattr("app.services.memory.get_history", lambda session_id: [])
    monkeypatch.setattr("app.services.memory.get_booking_slots", lambda session_id: BookingSlots())
    monkeypatch.setattr("app.services.memory.save_booking_slots", lambda *a, **k: None)
    monkeypatch.setattr("app.services.memory.clear_booking_slots", lambda *a, **k: None)
    monkeypatch.setattr("app.services.memory.append_turn", lambda *a, **k: None)

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "s2", "message": "Book me for Sept 20 at 3pm, I'm Smriti, smriti@example.com"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["booking"] is not None
    assert body["booking"]["name"] == "Smriti"
    assert body["booking"]["email"] == "smriti@example.com"
