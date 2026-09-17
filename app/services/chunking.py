"""Two selectable text-chunking strategies, exposed behind a common interface.

Adding a new strategy only requires subclassing ChunkingStrategy and registering it
in STRATEGY_REGISTRY - no changes needed anywhere else in the codebase.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.schemas.document import ChunkingStrategyName


class ChunkingStrategy(ABC):
    """Common interface for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split `text` into a list of chunk strings, in order."""
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """Splits text into fixed-size, overlapping character windows.

    Simple and predictable; good baseline for dense/technical text where sentence
    boundaries are less meaningful (tables, code, logs).
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == text_len:
                break
            start += step

        return chunks


class SentenceWindowChunking(ChunkingStrategy):
    """Groups whole sentences into windows up to a target size.

    Preserves sentence boundaries so chunks read naturally and don't cut a sentence
    mid-way, which tends to help retrieval quality and answer coherence for prose-like
    content (policies, articles, reports).
    """

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, target_chunk_size: int = 800, sentence_overlap: int = 1) -> None:
        self.target_chunk_size = target_chunk_size
        self.sentence_overlap = max(sentence_overlap, 0)

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        i = 0

        while i < len(sentences):
            sentence = sentences[i]
            if current and current_len + len(sentence) > self.target_chunk_size:
                chunks.append(" ".join(current))
                # start next window with overlap from the tail of the previous one
                overlap_sentences = current[-self.sentence_overlap :] if self.sentence_overlap else []
                current = list(overlap_sentences)
                current_len = sum(len(s) for s in current)
            current.append(sentence)
            current_len += len(sentence)
            i += 1

        if current:
            chunks.append(" ".join(current))

        return chunks


STRATEGY_REGISTRY: dict[ChunkingStrategyName, type[ChunkingStrategy]] = {
    ChunkingStrategyName.FIXED: FixedSizeChunking,
    ChunkingStrategyName.SENTENCE_WINDOW: SentenceWindowChunking,
}


def get_chunking_strategy(
    name: ChunkingStrategyName,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> ChunkingStrategy:
    """Factory returning a configured ChunkingStrategy instance for `name`."""
    strategy_cls = STRATEGY_REGISTRY[name]
    if strategy_cls is FixedSizeChunking:
        return FixedSizeChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return SentenceWindowChunking(target_chunk_size=chunk_size)
