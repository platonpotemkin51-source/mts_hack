"""Offline BM25 baseline with Russian normalization. Standard library only.

No network or indexing on import. A built retriever is read-only and may be
shared between threads. This is lexical retrieval, not embeddings/hybrid.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import unicodedata

from contracts import KnowledgeChunk, Retriever
from chunking import CHUNKER_VERSION, MAX_SECTION_CHARS, OVERLAP_CHARS, chunk_documents
from kb_loader import load_documents

INDEX_VERSION = "bm25-ru-v1"
_LOG = logging.getLogger(__name__)
_STOPWORDS = frozenset("""а без бы был была были было быть в вам вас ваш ваша ваши во вот все всего вы где да даже для до его ее если есть еще же за зачем здесь и из или им их к как какая какие какой какому ко когда кто ли либо мне может можно мы на над надо нам нас наш наша не него нее нет ни но о об он она они оно от по под после почему при про с со так также такой там то того тоже только том тут ты у уже чем через что чтобы это этой этот эту я меня мой очень пожалуйста здравствуйте привет спасибо пользователь ассистент хочу хотим нужно нужен нужна нужны стоит вообще сможете можете сколько""".split())
# shortcut: noun/adjective endings only; use a full stemmer if measured recall needs it.
_SUFFIXES = tuple(sorted(set("иями ями ами ией иям ием иях его ого ему ому ими ыми ее ие ые ое ей ий ый ой ем им ым ом их ых ую юю ая яя ою ею ев ов ье еи ии й ям ам ах ях ию ью ия ья а е и о у ы ь ю я".split()), key=lambda s: (-len(s), s)))
_ALIASES = {
    "электричеств": "электроэнерг", "крыш": "кровл",
    "фото": "фотограф", "снимок": "фотограф", "снимк": "фотограф",
    "симк": "сим", "sim": "сим", "орэм": "оптов", "askue": "аскуэ",
}


def tokenize(text: str, *, expand: bool = True) -> list[str]:
    """Normalize ё, hyphens, units, grouped/decimal numbers and simple endings.

    Curated aliases express vocabulary only; they never add product claims.
    Short unknown words/numbers/abbreviations are retained exactly.
    """
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    text = re.sub(r"квт\s*[·*⋅-]?\s*[чh]|kwh", " квтч ", text)
    text = re.sub(r"k[wв]", " квт ", text)
    text = text.replace("₽", " руб ").replace("%", " процент ")
    tokens: list[str] = []
    for word in re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+", text, re.UNICODE):
        if word in _STOPWORDS or (len(word) == 1 and not word.isdigit()):
            continue
        if re.fullmatch(r"[а-я]{5,}", word):
            for suffix in _SUFFIXES:
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    word = word[:-len(suffix)]
                    break
        if expand:
            word = _ALIASES.get(word, word)
        tokens.append(word)
    return tokens


class BM25Retriever:
    """Immutable after construction; retrieve uses only request-local state.

    Positive Robertson IDF, k1=1.5, b=0.75. score is ranking only, never a
    probability. A lexical match does not establish that the KB answers a query.
    """

    def __init__(self, chunks: list[KnowledgeChunk], *, expand: bool = True):
        if not chunks:
            raise ValueError("Cannot index an empty knowledge base")
        self._chunks = tuple(chunks)
        self._expand = expand
        self._lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            terms = tokenize(chunk.text, expand=expand)
            self._lengths.append(len(terms))
            for token, count in Counter(terms).items():
                postings[token].append((index, count))
        self._average = sum(self._lengths) / len(chunks) or 1.0
        self._postings = {key: tuple(values) for key, values in postings.items()}
        self._idf = {key: math.log1p((len(chunks) - len(values) + 0.5) / (len(values) + 0.5))
                     for key, values in postings.items()}

    def retrieve(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        scores: dict[int, float] = defaultdict(float)
        for token in sorted(set(tokenize(query, expand=self._expand))):
            for index, frequency in self._postings.get(token, ()):
                norm = 1.5 * (0.25 + 0.75 * self._lengths[index] / self._average)
                scores[index] += self._idf[token] * frequency * 2.5 / (frequency + norm)
        ranked = sorted(scores, key=lambda i: (-scores[i], self._chunks[i].chunk_id))
        result: list[KnowledgeChunk] = []
        seen: set[str] = set()
        for index in ranked:
            chunk = self._chunks[index]
            if scores[index] > 0 and chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                result.append(replace(chunk, score=scores[index]))
                if len(result) == top_k:
                    break
        return result


def _cache_key(documents) -> str:
    configuration = [INDEX_VERSION, CHUNKER_VERSION, MAX_SECTION_CHARS, OVERLAP_CHARS,
                     sorted(_STOPWORDS), _SUFFIXES, sorted(_ALIASES.items()),
                     [(d.source, d.document_id) for d in documents]]
    return hashlib.sha256(json.dumps(configuration, ensure_ascii=False).encode()).hexdigest()


def _read_cache(path: Path, key: str) -> list[KnowledgeChunk] | None:
    try:
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("Cache exceeds size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["key"] != key or payload["version"] != INDEX_VERSION:
            return None
        chunks = [KnowledgeChunk(**item) for item in payload["chunks"]]
        if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Invalid cached chunks")
        if any(not all(isinstance(v, str) and v for v in (c.chunk_id, c.source, c.title, c.text))
               or c.score != 0.0 for c in chunks):
            raise ValueError("Invalid cached chunk fields")
        return chunks
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        _LOG.warning("Ignoring unreadable RAG cache; rebuilding from knowledge base")
        return None


def _write_cache(path: Path, key: str, chunks: list[KnowledgeChunk]) -> None:
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".rag-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump({"key": key, "version": INDEX_VERSION,
                       "chunks": [asdict(c) for c in chunks]}, stream, ensure_ascii=False)
        os.replace(temporary, path)
    except OSError:
        _LOG.warning("Cannot write RAG cache; using in-memory index")
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                _LOG.warning("Cannot remove temporary RAG cache")


def build_retriever(kb_path: Path, cache_dir: Path | None = None) -> Retriever:
    """Read the full KB and build BM25 once; reuse the returned object.

    Optional JSON cache stores chunks (private data), keyed by all document
    hashes and algorithm configuration. Postings are rebuilt in memory; no
    pickle or network. Use an ignored runtime directory for the cache.
    """
    documents = load_documents(Path(kb_path))
    key = _cache_key(documents)
    path = Path(cache_dir) / f"bm25-{key}.json" if cache_dir is not None else None
    chunks = _read_cache(path, key) if path is not None else None
    if chunks is None:
        chunks = chunk_documents(documents)
        if path is not None:
            _write_cache(path, key, chunks)
    return BM25Retriever(chunks)
