"""Artem owns semantic Markdown splitting with constraints preserved."""

from __future__ import annotations

from contracts import KnowledgeChunk, KnowledgeDocument


def chunk_documents(documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
    raise NotImplementedError("Artem: implement chunking; see docs/tasks/ARTEM.md")
