"""Shared domain interfaces; stdlib only, usable in a flat competition ZIP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    text: str
    score: float = 0.0


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        """Return unique relevant chunks, ranked best first."""
        ...


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ClientBrief:
    object: str | None = None
    problem: str | None = None
    known_parameters: list[str] = field(default_factory=list)
    recommended_products: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    manager_summary: str | None = None


@dataclass(frozen=True)
class Timings:
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class ConsultationResult:
    answer: str
    sources: list[KnowledgeChunk]
    brief: ClientBrief
    lead_offer: bool = False
    timings: Timings = field(default_factory=Timings)
