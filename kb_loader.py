"""Artem owns loading the private KB. No data is bundled with the scaffold."""

from __future__ import annotations

from pathlib import Path

from contracts import KnowledgeDocument


def load_documents(kb_path: Path) -> list[KnowledgeDocument]:
    """Read Markdown documents from a KB ZIP or directory without side effects."""
    raise NotImplementedError("Artem: implement KB loading; see docs/tasks/ARTEM.md")
