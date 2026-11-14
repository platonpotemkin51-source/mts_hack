"""BM25, organizer embeddings and RRF hybrid behind one retrieval contract.

No network or indexing on import. A built retriever is read-only and may be
shared between threads. HTTPX is imported only when embeddings are requested.
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
import time
import unicodedata
from urllib.parse import urlsplit

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


def _validate_query(query: str, top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(query, str):
        raise TypeError("query must be a string")


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
        _validate_query(query, top_k)
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


class EmbeddingError(RuntimeError):
    """Safe diagnostic without credentials, response body or document content."""


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def _normalize_vectors(value, count: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count or not count:
        raise EmbeddingError("Embedding response has incorrect vector count")
    vectors = []
    dimension = None
    for vector in value:
        if (not isinstance(vector, (list, tuple)) or not 1 <= len(vector) <= 8192
                or (dimension is not None and len(vector) != dimension)):
            raise EmbeddingError("Embedding vectors have invalid dimensions")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vector):
            raise EmbeddingError("Embedding vector contains non-finite or non-numeric values")
        try:
            vector = tuple(float(x) for x in vector)
        except OverflowError:
            raise EmbeddingError("Embedding vector value exceeds float range") from None
        if not all(math.isfinite(x) for x in vector):
            raise EmbeddingError("Embedding vector contains non-finite or non-numeric values")
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise EmbeddingError("Embedding vector has invalid norm")
        dimension = len(vector)
        vectors.append(tuple(x / norm for x in vector))
    return tuple(vectors)


class OrganizerEmbedder:
    """POST /embeddings, as in the organizer's sample_submission.zip.

    Each call owns its HTTP client, so calls can run in different threads.
    No network in construction; at most two attempts per batch, timeout 30s.
    """

    def __init__(self):
        self.base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
        self.model = _env("EMB_MODEL", default="baai/bge-m3")
        self._key = _env("EMBEDDER_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
        try:
            url = urlsplit(self.base_url)
            valid = (url.scheme in {"http", "https"} and url.hostname and url.port != 0
                     and not url.username and not url.password and not url.query and not url.fragment)
        except ValueError:
            valid = False
        if not valid or not self._key:
            raise EmbeddingError("Set proxy BASE_URL and API_KEY via LLM_* or OPENAI_* environment variables")

    def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty strings")
        import httpx

        vectors = []
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            for start in range(0, len(texts), 16):
                batch = texts[start:start + 16]
                for attempt in range(2):
                    try:
                        response = client.post(
                            self.base_url + "/embeddings",
                            headers={"Authorization": "Bearer " + self._key},
                            json={"model": self.model, "input": batch},
                        )
                    except httpx.TransportError:
                        if attempt == 0:
                            time.sleep(0.5)
                            continue
                        raise EmbeddingError("Embedding proxy transport failure after 2 attempts") from None
                    status = response.status_code
                    if status == 200:
                        break
                    if (status == 429 or 500 <= status < 600) and attempt == 0:
                        time.sleep(0.5)
                        continue
                    raise EmbeddingError(f"Embedding proxy HTTP {status}; no lexical fallback")
                try:
                    items = response.json()["data"]
                    if not isinstance(items, list) or len(items) != len(batch):
                        raise ValueError("count")
                    ordered = [None] * len(batch)
                    for ordinal, item in enumerate(items):
                        index = item.get("index", ordinal)
                        if (isinstance(index, bool) or not isinstance(index, int)
                                or not 0 <= index < len(batch) or ordered[index] is not None):
                            raise ValueError("index")
                        ordered[index] = item["embedding"]
                    vectors.extend(_normalize_vectors(ordered, len(batch)))
                except (ValueError, KeyError, TypeError, AttributeError):
                    raise EmbeddingError("Malformed embedding response: expected data[index, embedding]") from None
        # Validate cross-batch dimensions too.
        return _normalize_vectors(vectors, len(texts))


class EmbeddingRetriever:
    def __init__(self, chunks: list[KnowledgeChunk], embedder: OrganizerEmbedder, vectors):
        self._chunks = tuple(chunks)
        self._embedder = embedder
        self._vectors = _normalize_vectors(vectors, len(chunks))

    def retrieve(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        _validate_query(query, top_k)
        if not tokenize(query):
            return []
        vector = self._embedder.embed([query])[0]
        if len(vector) != len(self._vectors[0]):
            raise EmbeddingError("Query embedding dimension differs from document index; rebuild/check model")
        scores = [sum(a * b for a, b in zip(vector, doc)) for doc in self._vectors]
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], self._chunks[i].chunk_id))
        return [replace(self._chunks[i], score=scores[i]) for i in ranked[:top_k]]


class HybridRetriever:
    """Equal-weight reciprocal rank fusion; never add BM25 and cosine scores."""

    def __init__(self, lexical: Retriever, semantic: Retriever):
        self._lexical = lexical
        self._semantic = semantic

    def retrieve(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        _validate_query(query, top_k)
        if not tokenize(query):
            return []
        scores: dict[str, float] = defaultdict(float)
        chunks = {}
        for retriever in (self._lexical, self._semantic):
            seen = set()
            for rank, chunk in enumerate(retriever.retrieve(query, max(20, top_k)), 1):
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    chunks[chunk.chunk_id] = chunk
                    scores[chunk.chunk_id] += 1.0 / (60 + rank)
        ranked = sorted(chunks, key=lambda key: (-scores[key], key))
        return [replace(chunks[key], score=scores[key]) for key in ranked[:top_k]]


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


def _write_cache(path: Path, payload: dict) -> None:
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".rag-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    except OSError:
        _LOG.warning("Cannot write RAG cache; using in-memory index")
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                _LOG.warning("Cannot remove temporary RAG cache")


def _embedding_retriever(chunks: list[KnowledgeChunk], cache_dir: Path | None) -> EmbeddingRetriever:
    embedder = OrganizerEmbedder()
    key = hashlib.sha256(json.dumps([
        "dense-v1", embedder.base_url, embedder.model, [c.chunk_id for c in chunks],
    ]).encode()).hexdigest()
    path = Path(cache_dir) / f"embeddings-{key}.json" if cache_dir is not None else None
    vectors = None
    if path is not None:
        try:
            if path.stat().st_size > 128 * 1024 * 1024:
                raise ValueError("size")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["key"] == key:
                vectors = _normalize_vectors(payload["vectors"], len(chunks))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError, EmbeddingError):
            _LOG.warning("Ignoring unreadable embedding cache; rebuilding via proxy")
    if vectors is None:
        vectors = embedder.embed([c.text for c in chunks])
        if path is not None:
            _write_cache(path, {"key": key, "vectors": vectors})
    return EmbeddingRetriever(chunks, embedder, vectors)


def build_retriever(kb_path: Path, cache_dir: Path | None = None) -> Retriever:
    """Build once; RETRIEVAL_MODE=bm25 (default), embeddings or hybrid.

    BM25 needs no network/dependencies. Other modes use organizer HTTP embeddings,
    cache document vectors and embed each query. Errors never silently switch mode.
    Use an ignored runtime directory for caches containing private KB material.
    """
    mode = _env("RETRIEVAL_MODE", default="bm25").lower()
    if mode not in {"bm25", "embeddings", "hybrid"}:
        raise ValueError("RETRIEVAL_MODE must be bm25, embeddings or hybrid")
    documents = load_documents(Path(kb_path))
    key = _cache_key(documents)
    path = Path(cache_dir) / f"bm25-{key}.json" if cache_dir is not None else None
    chunks = _read_cache(path, key) if path is not None else None
    if chunks is None:
        chunks = chunk_documents(documents)
        if path is not None:
            _write_cache(path, {"key": key, "version": INDEX_VERSION,
                                "chunks": [asdict(c) for c in chunks]})
    if mode == "bm25":
        return BM25Retriever(chunks)
    semantic = _embedding_retriever(chunks, cache_dir)
    return semantic if mode == "embeddings" else HybridRetriever(BM25Retriever(chunks), semantic)
