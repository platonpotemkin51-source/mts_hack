"""Heading-based Markdown chunks; complete sections are preferred to size cuts."""

from __future__ import annotations

from contracts import KnowledgeChunk, KnowledgeDocument
import hashlib
import re

CHUNKER_VERSION = "markdown-sections-v1"
# Soft bound: indivisible tables/paragraphs may be larger. Never truncate facts.
MAX_SECTION_CHARS = 5000
OVERLAP_CHARS = 500
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _sections(document: KnowledgeDocument):
    headings: list[tuple[int, str]] = []
    body: list[str] = []
    fence: str | None = None
    for line in document.text.splitlines():
        marker = _FENCE.match(line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            body.append(line)
            continue
        match = _HEADING.match(line) if fence is None else None
        if match:
            if any(part.strip() for part in body):
                yield tuple(title for _, title in headings), "\n".join(body).strip("\n")
            body = []
            level, title = len(match.group(1)), match.group(2)
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
        else:
            body.append(line)
    if any(part.strip() for part in body):
        yield tuple(title for _, title in headings), "\n".join(body).strip("\n")


def _parts(body: str) -> list[str]:
    """Only split oversized sections at blank lines outside fenced code.

    Indented Pandoc tables, pipe tables and lists often contain blank lines.
    Keep these whole rather than assume a blank line ends a table or condition.
    """
    if len(body) <= MAX_SECTION_CHARS:
        return [body]
    lines = body.splitlines()
    # Conservative handling of structured sections: retain full section.
    if any(re.match(r"^(?: {2,}\S|\s*\||\s*[-*+]\s|\s*\d+[.)]\s|```|~~~)", line)
           for line in lines):
        return [body]
    blocks = re.split(r"\n\s*\n", body)
    result: list[str] = []
    current: list[str] = []
    for block in blocks:
        if current and len("\n\n".join([*current, block])) > MAX_SECTION_CHARS:
            result.append("\n\n".join(current))
            previous = current[-1]
            current = [previous] if len(previous) <= OVERLAP_CHARS else []
        current.append(block)
    if current:
        result.append("\n\n".join(current))
    return result


def chunk_documents(documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
    """Retain heading ancestry, every body block, table and document tail.

    Soft size bound deliberately yields to intact tables and caveats. Consumers
    must budget complete chunks, not slice their text. IDs include source,
    heading ancestry, part ordinal, text and chunker version.
    """
    chunks: list[KnowledgeChunk] = []
    seen: set[str] = set()
    for document in sorted(documents, key=lambda d: d.source):
        for headings, body in _sections(document):
            titles = headings if headings and headings[0] == document.title else (document.title, *headings)
            title = " / ".join(titles)
            for ordinal, part in enumerate(_parts(body)):
                text = "\n".join("#" * min(i + 1, 6) + " " + h for i, h in enumerate(titles))
                text += "\n\n" + part
                digest = hashlib.sha256(
                    f"{CHUNKER_VERSION}\0{document.source}\0{ordinal}\0{text}".encode()
                ).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    chunks.append(KnowledgeChunk(digest, document.source, title, text))
    return chunks
