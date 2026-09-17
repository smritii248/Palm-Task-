"""Deterministic unit tests for chunking strategies (no network calls)."""

from __future__ import annotations

from app.schemas.document import ChunkingStrategyName
from app.services.chunking import FixedSizeChunking, SentenceWindowChunking, get_chunking_strategy


def test_fixed_size_chunking_respects_size_and_overlap() -> None:
    text = "a" * 2000
    strategy = FixedSizeChunking(chunk_size=800, chunk_overlap=100)
    chunks = strategy.chunk(text)

    assert len(chunks) >= 2
    assert all(len(c) <= 800 for c in chunks)
    # Reconstructed length (accounting for overlap) should cover the whole text.
    assert sum(len(c) for c in chunks) >= len(text)


def test_fixed_size_chunking_empty_text_returns_no_chunks() -> None:
    assert FixedSizeChunking().chunk("   ") == []


def test_fixed_size_chunking_rejects_invalid_overlap() -> None:
    try:
        FixedSizeChunking(chunk_size=100, chunk_overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_sentence_window_chunking_preserves_sentence_boundaries() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    strategy = SentenceWindowChunking(target_chunk_size=40, sentence_overlap=1)
    chunks = strategy.chunk(text)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.strip().endswith(".")


def test_sentence_window_chunking_empty_text_returns_no_chunks() -> None:
    assert SentenceWindowChunking().chunk("") == []


def test_get_chunking_strategy_factory_returns_correct_type() -> None:
    fixed = get_chunking_strategy(ChunkingStrategyName.FIXED)
    sentence = get_chunking_strategy(ChunkingStrategyName.SENTENCE_WINDOW)

    assert isinstance(fixed, FixedSizeChunking)
    assert isinstance(sentence, SentenceWindowChunking)
