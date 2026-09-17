# RAG Backend

A modular FastAPI backend implementing:

1. **Document Ingestion API** — upload PDF/TXT files, extract text, chunk it using one
   of two selectable strategies, embed the chunks, store vectors in **Qdrant**, and
   persist file/chunk metadata in a SQL database (SQLite by default, Postgres-ready).
2. **Conversational RAG API** — a hand-rolled retrieval-augmented generation pipeline
   (no `RetrievalQAChain`, no LangChain chain abstractions) with **Redis**-backed
   multi-turn chat memory, and an LLM-driven interview **booking** flow that extracts
   structured booking details (name, email, date, time) from natural conversation and
   persists them to the database.


---

## Architecture

```
app/
├── main.py                 # FastAPI app factory, router registration, lifespan
├── config.py                # Pydantic-settings based configuration (env driven)
├── core/
│   └── logging.py           # Centralized logging setup
├── db/
│   ├── database.py          # SQLAlchemy engine/session
│   └── models.py            # ORM models: Document, Chunk, Booking, ChatSession
├── schemas/
│   ├── document.py          # Pydantic request/response models for ingestion
│   ├── chat.py               # Pydantic request/response models for chat
│   └── booking.py            # Pydantic booking models
├── services/
│   ├── text_extraction.py   # PDF/TXT -> raw text
│   ├── chunking.py          # Strategy pattern: fixed-size & sentence-window chunking
│   ├── embeddings.py        # Embedding provider abstraction (OpenAI by default)
│   ├── vector_store.py      # Qdrant client wrapper (create/upsert/search)
│   ├── memory.py            # Redis-backed conversation memory
│   ├── rag.py                # Custom retrieval + prompt assembly + generation
│   └── booking_agent.py     # LLM tool-call based booking-slot extraction
└── api/
    ├── ingestion.py          # POST /api/v1/documents/upload
    └── chat.py                # POST /api/v1/chat
```

### Design choices

- **Strategy pattern for chunking** — `ChunkingStrategy` is an abstract base; two
  concrete strategies (`FixedSizeChunking`, `SentenceWindowChunking`) are selectable
  per-request via a `strategy` field, so adding a third strategy later requires no
  changes to calling code.
- **No LangChain chain abstractions.** `rag.py` manually: (1) embeds the user query,
  (2) retrieves top-k chunks from Qdrant, (3) pulls prior turns from Redis, (4) builds
  a prompt, (5) calls the LLM directly, (6) writes the new turn back to Redis. This
  satisfies the "custom RAG, no RetrievalQAChain" constraint explicitly.
- **Booking is LLM-tool-call driven, not regex.** On every turn, the RAG service asks
  the LLM (via function/tool calling) whether the user is trying to book an interview
  and to extract `name`, `email`, `date`, `time` if present. Partial info is tracked
  per session in Redis until all fields are collected, then persisted to SQL.
- **Everything is typed.** All functions use Python type hints; all API I/O is defined
  via Pydantic models; ORM models use SQLAlchemy 2.0 `Mapped`/`mapped_column` typed
  declarative style.

---

## Setup

### 1. Infrastructure (Qdrant + Redis + Postgres) via Docker Compose

```bash
docker compose up -d
```

This starts Qdrant (`:6333`), Redis (`:6379`), and Postgres (`:5432`). SQLite is used
automatically instead if `DATABASE_URL` is left as the default in `.env`.

### 2. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# then set OPENAI_API_KEY (or swap the embedding/LLM provider in services/)
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

Swagger UI: `http://localhost:8000/docs`

---

## API Overview

### `POST /api/v1/documents/upload`
`multipart/form-data`: `file` (.pdf or .txt), `chunking_strategy` (`fixed` | `sentence_window`).

Returns the document id, number of chunks created, and the vector-store collection used.

### `POST /api/v1/chat`
```json
{
  "session_id": "user-123",
  "message": "What does the onboarding policy say about laptops?"
}
```
Returns the assistant's reply, the source chunks used, and — if a booking was detected
and completed in this turn — the persisted booking record.

### `GET /api/v1/bookings/{session_id}`
Returns any booking captured for that session.

---

## Testing

```bash
pytest
```

`tests/test_chunking.py` covers both chunking strategies deterministically (no network
calls). `tests/test_api.py` uses FastAPI's `TestClient` with the embedding/LLM/vector
services monkey-patched, so the suite runs without live external services.

## Notes on swapping providers

- Vector DB: `services/vector_store.py` only talks to Qdrant's client; swapping to
  Weaviate/Milvus/Pinecone means replacing that one module — nothing else depends on
  Qdrant directly.
- Embeddings/LLM: `services/embeddings.py` and the LLM call in `services/rag.py` and
  `services/booking_agent.py` wrap the OpenAI SDK behind small functions, so swapping
  providers is localized.
