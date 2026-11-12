"""Artem owns the retrieval implementation; Platon consumes this factory."""

from __future__ import annotations

from pathlib import Path

from contracts import Retriever


def build_retriever(kb_path: Path, cache_dir: Path | None = None) -> Retriever:
    """Build/load an index once. Importing this module must not build it."""
    raise NotImplementedError("Artem: implement retrieval; see docs/tasks/ARTEM.md")
