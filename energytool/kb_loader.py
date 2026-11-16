"""Bounded UTF-8 Markdown loading. ZIP entries are never extracted."""

from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
import hashlib
import re
import stat
from zipfile import BadZipFile, ZipFile

from energytool.contracts import KnowledgeDocument

MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 10_000


class KnowledgeBaseError(ValueError):
    """Missing, malformed, unsafe or unsupported knowledge base."""


def _source(name: str) -> str:
    # Treat Windows separators as separators too, even on Linux/macOS.
    name = name.replace("\\", "/")
    parts = name.split("/")
    if (name.startswith("/") or ".." in parts or "\x00" in name
            or any(":" in part for part in parts)):
        raise KnowledgeBaseError("Unsafe path in knowledge base")
    source = PurePosixPath(name).as_posix()
    if source in {"", "."}:
        raise KnowledgeBaseError("Empty document path")
    return source


def _document(source: str, raw: bytes) -> KnowledgeDocument:
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise KnowledgeBaseError(f"Markdown is not UTF-8: {source}") from exc
    if not text.strip() or "\x00" in text:
        raise KnowledgeBaseError(f"Empty or binary Markdown: {source}")
    heading = re.search(r"^ {0,3}#\s+(.+?)\s*#*\s*$", text, re.MULTILINE)
    title = heading.group(1) if heading else PurePosixPath(source).stem
    digest = hashlib.sha256((source + "\0" + text).encode()).hexdigest()
    return KnowledgeDocument(digest, source, title, text)


def load_documents(kb_path: Path) -> list[KnowledgeDocument]:
    """Read every .md (case insensitive), deterministically, without extraction.

    ZIP and directory inputs produce identical IDs for identical relative paths
    and content. Directory symlinks are rejected; non-Markdown files are ignored.
    Limits apply before and during reading, including ZIP expansion.
    """
    path = Path(kb_path)
    if path.is_symlink():
        raise KnowledgeBaseError("Symlinks are not supported in knowledge base")
    documents: dict[str, KnowledgeDocument] = {}
    total = 0

    def add(name: str, raw: bytes) -> None:
        nonlocal total
        source = _source(name)
        if source in documents:
            raise KnowledgeBaseError(f"Duplicate Markdown path: {source}")
        total += len(raw)
        if len(raw) > MAX_DOCUMENT_BYTES or total > MAX_TOTAL_BYTES:
            raise KnowledgeBaseError("Knowledge base exceeds reading limits")
        documents[source] = _document(source, raw)

    if path.is_dir():
        entries = 0
        for item in path.rglob("*"):
            entries += 1
            if entries > MAX_ENTRIES:
                raise KnowledgeBaseError("Too many knowledge base entries")
            if item.is_symlink():
                raise KnowledgeBaseError("Symlinks are not supported in knowledge base")
            if not item.is_file() or item.suffix.lower() != ".md":
                continue
            if item.stat().st_size > MAX_DOCUMENT_BYTES:
                raise KnowledgeBaseError("Markdown exceeds reading limit")
            with item.open("rb") as stream:
                add(item.relative_to(path).as_posix(), stream.read(MAX_DOCUMENT_BYTES + 1))
    elif path.is_file():
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise KnowledgeBaseError("ZIP exceeds archive size limit")
        try:
            with ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_ENTRIES:
                    raise KnowledgeBaseError("Too many knowledge base entries")
                for entry in sorted(entries, key=lambda e: e.filename):
                    if entry.is_dir():
                        continue
                    name = _source(entry.filename)
                    if PurePosixPath(name).suffix.lower() != ".md":
                        continue
                    if stat.S_ISLNK(entry.external_attr >> 16):
                        raise KnowledgeBaseError("ZIP symlinks are not supported")
                    if (entry.file_size > MAX_DOCUMENT_BYTES
                            or total + entry.file_size > MAX_TOTAL_BYTES):
                        raise KnowledgeBaseError("ZIP expansion exceeds reading limits")
                    with archive.open(entry) as stream:
                        add(name, stream.read(MAX_DOCUMENT_BYTES + 1))
        except (BadZipFile, RuntimeError, NotImplementedError, EOFError) as exc:
            raise KnowledgeBaseError("Cannot read knowledge base ZIP") from exc
    else:
        raise FileNotFoundError(f"Knowledge base not found: {path}")
    if not documents:
        raise KnowledgeBaseError("Knowledge base contains no Markdown documents")
    return [documents[source] for source in sorted(documents)]
