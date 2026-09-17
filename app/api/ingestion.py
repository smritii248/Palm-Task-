"""Document ingestion API: upload, extract, chunk, embed, store."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.db.models import Chunk, Document
from app.schemas.document import ChunkingStrategyName, DocumentUploadResponse
from app.services.chunking import get_chunking_strategy
from app.services.embeddings import embed_batch
from app.services.text_extraction import UnsupportedFileTypeError, extract_text
from app.services.vector_store import upsert_chunks

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/documents", tags=["ingestion"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="A .pdf or .txt file"),
    chunking_strategy: ChunkingStrategyName = Form(default=ChunkingStrategyName.FIXED),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    """Upload a document, chunk it, embed the chunks, and persist everything.

    Steps: extract text -> chunk (selected strategy) -> embed -> upsert to Qdrant
    -> persist Document/Chunk metadata rows in the SQL database.
    """
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    try:
        text = extract_text(raw_bytes, file.content_type or "", file.filename or "upload")
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No extractable text found")

    strategy = get_chunking_strategy(
        chunking_strategy,
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
    )
    chunk_texts = strategy.chunk(text)
    if not chunk_texts:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Chunking produced no chunks")

    document = Document(
        filename=file.filename or "upload",
        content_type=file.content_type or "unknown",
        chunking_strategy=chunking_strategy.value,
        chunk_count=len(chunk_texts),
        vector_collection=settings.qdrant_collection,
    )
    db.add(document)
    db.flush()  # obtain document.id before building chunk payloads

    vectors = embed_batch(chunk_texts)
    payloads = [
        {"document_id": document.id, "chunk_index": idx, "text": chunk_text}
        for idx, chunk_text in enumerate(chunk_texts)
    ]
    vector_ids = upsert_chunks(settings.qdrant_collection, vectors, payloads)

    for idx, (chunk_text, vector_id) in enumerate(zip(chunk_texts, vector_ids, strict=True)):
        db.add(Chunk(document_id=document.id, chunk_index=idx, text=chunk_text, vector_id=vector_id))

    db.commit()
    db.refresh(document)

    logger.info("Ingested document %s (%d chunks, strategy=%s)", document.id, len(chunk_texts), chunking_strategy)
    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        chunking_strategy=ChunkingStrategyName(document.chunking_strategy),
        chunk_count=document.chunk_count,
        vector_collection=document.vector_collection,
        created_at=document.created_at,
    )
