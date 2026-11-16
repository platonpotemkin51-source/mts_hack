"""Organizer-compatible HTTP fixtures; no live service or private KB required."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from energytool.contracts import KnowledgeChunk
from energytool.retrieval import (
    BM25Retriever, EmbeddingError, EmbeddingRetriever, HybridRetriever,
    OrganizerEmbedder, _normalize_vectors, build_retriever,
)
from evaluation.rag.compare import compare


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"LLM_BASE_URL": "http://proxy.test/api/",
                               "LLM_API_KEY": "invented-key",
                               "EMB_MODEL": "test-embedder",
                               "RETRIEVAL_MODE": "embeddings"}, clear=True).start()
        patch("energytool.retrieval.time.sleep").start()
        self.requests = []
        self.respond = self.good_response
        original_client = httpx.Client

        def handle(request):
            self.requests.append(request)
            return self.respond(request)

        patch("httpx.Client", side_effect=lambda **kw: original_client(
            transport=httpx.MockTransport(handle), **kw)).start()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.kb = self.root / "kb"
        self.kb.mkdir()
        (self.kb / "invented.md").write_text(
            "# Вымышленный каталог\n\n## Кровля\n\nКровля макета.\n\n"
            "## Фотография\n\nФотография макета.\n", encoding="utf-8")

    def good_response(self, request):
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1, 0] if "кровл" in text.lower() else [0, 1]}
            for i, text in reversed(list(enumerate(texts)))
        ]})

    def test_http_contract_order_and_normalization(self):
        vectors = OrganizerEmbedder().embed(["Кровля", "Фотография"])
        self.assertEqual(vectors, ((1.0, 0.0), (0.0, 1.0)))
        request = self.requests[0]
        self.assertEqual(str(request.url), "http://proxy.test/api/embeddings")
        self.assertEqual(request.headers["Authorization"], "Bearer invented-key")
        self.assertEqual(json.loads(request.content), {
            "model": "test-embedder", "input": ["Кровля", "Фотография"]})
        self.assertEqual(_normalize_vectors([[3, 4]], 1), ((0.6, 0.8),))

    def test_batches_are_bounded(self):
        self.assertEqual(len(OrganizerEmbedder().embed(["Фотография"] * 35)), 35)
        self.assertEqual([len(json.loads(r.content)["input"]) for r in self.requests], [16, 16, 3])

    def test_missing_indices_use_response_order(self):
        self.respond = lambda _: httpx.Response(200, json={"data": [{"embedding": [0, 2]}]})
        self.assertEqual(OrganizerEmbedder().embed(["test"]), ((0.0, 1.0),))

    def test_invalid_responses_are_errors(self):
        payloads = [None, {}, {"data": []}, {"data": "bad"},
                    {"data": [{"index": -1, "embedding": [1, 0]}]},
                    {"data": [{"index": True, "embedding": [1, 0]}]},
                    {"data": [{"embedding": [0, 0]}]},
                    {"data": [{"embedding": ["1", 0]}]},
                    {"data": [{"embedding": [True, 0]}]}]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.respond = lambda _, p=payload: httpx.Response(200, json=p)
                with self.assertRaises(EmbeddingError):
                    OrganizerEmbedder().embed(["test"])
        self.respond = lambda _: httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}]})
        with self.assertRaises(EmbeddingError):
            OrganizerEmbedder().embed(["one", "two"])

    def test_nonfinite_zero_and_dimension_errors(self):
        for vectors in ([[float("nan")]], [[float("inf")]], [[10**400]], [[0]], [[1], [1, 2]], [[]], [[1] * 8193]):
            with self.subTest(dimensions=[len(v) for v in vectors]), self.assertRaises(EmbeddingError):
                _normalize_vectors(vectors, len(vectors))

    def test_cross_batch_dimension_change_is_rejected(self):
        def different_dimensions(request):
            batch = json.loads(request.content)["input"]
            return httpx.Response(200, json={"data": [
                {"embedding": [1, 0] if len(batch) == 16 else [1, 0, 0]} for _ in batch]})
        self.respond = different_dimensions
        with self.assertRaises(EmbeddingError):
            OrganizerEmbedder().embed(["test"] * 17)

    def test_authorization_and_client_errors_do_not_retry_or_leak(self):
        for status in (400, 401, 403, 404, 302):
            self.requests.clear()
            self.respond = lambda _, s=status: httpx.Response(s, text="invented-key PRIVATE_TEXT")
            with self.assertRaises(EmbeddingError) as caught:
                OrganizerEmbedder().embed(["test"])
            self.assertEqual(len(self.requests), 1)
            self.assertIn(str(status), str(caught.exception))
            self.assertNotIn("invented-key", str(caught.exception))
            self.assertNotIn("PRIVATE_TEXT", str(caught.exception))

    def test_transient_errors_retry_once(self):
        for status in (429, 500, 503):
            self.requests.clear()
            self.respond = lambda request, s=status: (
                httpx.Response(s) if len(self.requests) == 1 else self.good_response(request))
            self.assertTrue(OrganizerEmbedder().embed(["test"]))
            self.assertEqual(len(self.requests), 2)
        self.requests.clear()
        self.respond = lambda _: httpx.Response(503)
        with self.assertRaises(EmbeddingError):
            OrganizerEmbedder().embed(["test"])
        self.assertEqual(len(self.requests), 2)

    def test_transport_errors_are_bounded(self):
        def fail(request):
            raise httpx.ConnectError("PRIVATE_TEXT invented-key", request=request)
        self.respond = fail
        with self.assertRaises(EmbeddingError) as caught:
            OrganizerEmbedder().embed(["test"])
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn("invented-key", str(caught.exception))

    def test_missing_settings_and_bad_urls_fail_before_network(self):
        for url in ("", "file:///tmp/base", "http://user:password@proxy.test", "https://proxy.test?token=key"):
            with patch.dict(os.environ, {"LLM_BASE_URL": url}):
                with self.assertRaises(EmbeddingError):
                    OrganizerEmbedder()
        with patch.dict(os.environ, {"LLM_API_KEY": "LLM_API_KEY"}):
            with self.assertRaises(EmbeddingError):
                OrganizerEmbedder()
        self.assertEqual(len(self.requests), 0)

    def test_environment_aliases(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://proxy.test/v1",
                                    "OPENAI_API_KEY": "alternate"}, clear=True):
            embedder = OrganizerEmbedder()
            self.assertEqual(embedder.model, "baai/bge-m3")
            embedder.embed(["test"])
        self.assertEqual(str(self.requests[0].url), "https://proxy.test/v1/embeddings")
        self.assertEqual(self.requests[0].headers["Authorization"], "Bearer alternate")

    def test_modes_document_cache_and_query_embedding(self):
        cache = self.root / "cache"
        dense = build_retriever(self.kb, cache)
        self.assertIsInstance(dense, EmbeddingRetriever)
        self.assertEqual(len(self.requests), 1)
        result = dense.retrieve("Фотография", 1)
        self.assertIn("Фотография", result[0].title)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(dense.retrieve("  "), [])
        self.assertEqual(len(self.requests), 2)
        build_retriever(self.kb, cache)
        self.assertEqual(len(self.requests), 2)
        os.environ["RETRIEVAL_MODE"] = "hybrid"
        hybrid = build_retriever(self.kb, cache)
        self.assertIsInstance(hybrid, HybridRetriever)
        self.assertEqual(len(self.requests), 2)
        self.assertIn("Фотография", hybrid.retrieve("Фотография", 1)[0].title)
        os.environ["RETRIEVAL_MODE"] = "bm25"
        count = len(self.requests)
        self.assertIsInstance(build_retriever(self.kb, cache), BM25Retriever)
        self.assertEqual(count, len(self.requests))

    def test_vector_cache_invalidates_for_model_proxy_and_content(self):
        cache = self.root / "cache"
        build_retriever(self.kb, cache)
        os.environ["EMB_MODEL"] = "another-test-model"
        build_retriever(self.kb, cache)
        os.environ["LLM_BASE_URL"] = "http://another-proxy.test"
        build_retriever(self.kb, cache)
        with (self.kb / "invented.md").open("a", encoding="utf-8") as stream:
            stream.write("\nНовый хвост.")
        build_retriever(self.kb, cache)
        self.assertEqual(len(self.requests), 4)
        for path in cache.glob("*.json"):
            self.assertNotIn("invented-key", path.read_text(encoding="utf-8"))

    def test_corrupt_vector_cache_rebuilds(self):
        cache = self.root / "cache"
        build_retriever(self.kb, cache)
        next(cache.glob("embeddings-*.json")).write_text("{broken", encoding="utf-8")
        with self.assertLogs("energytool.retrieval", level="WARNING"):
            build_retriever(self.kb, cache)
        self.assertEqual(len(self.requests), 2)

    def test_query_dimension_change_fails(self):
        retriever = build_retriever(self.kb)
        self.respond = lambda _: httpx.Response(200, json={"data": [{"embedding": [1, 0, 0]}]})
        with self.assertRaises(EmbeddingError):
            retriever.retrieve("test")

    def test_hybrid_uses_ranks_and_deduplicates(self):
        a = KnowledgeChunk("a", "a.md", "A", "Invented A", score=9999)
        b = KnowledgeChunk("b", "b.md", "B", "Invented B", score=0.01)
        c = KnowledgeChunk("c", "c.md", "C", "Invented C", score=100)

        class Fixed:
            def __init__(self, results):
                self.results = results

            def retrieve(self, query, top_k=6):
                return self.results[:top_k]

        hybrid = HybridRetriever(Fixed([a, b, a]), Fixed([replace(b, score=-1), c]))
        results = hybrid.retrieve("test", 6)
        self.assertEqual([x.chunk_id for x in results], ["b", "a", "c"])
        self.assertAlmostEqual(results[0].score, 1/61 + 1/62)

    def test_failed_semantic_mode_never_becomes_bm25(self):
        self.respond = lambda _: httpx.Response(401)
        for mode in ("embeddings", "hybrid"):
            os.environ["RETRIEVAL_MODE"] = mode
            with self.assertRaises(EmbeddingError):
                build_retriever(self.kb)
        os.environ["RETRIEVAL_MODE"] = "typo"
        with self.assertRaises(ValueError):
            build_retriever(self.kb)

    def test_same_validation_in_all_modes(self):
        for mode in ("embeddings", "hybrid"):
            os.environ["RETRIEVAL_MODE"] = mode
            retriever = build_retriever(self.kb)
            calls = len(self.requests)
            for invalid in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    retriever.retrieve("", invalid)
            with self.assertRaises(TypeError):
                retriever.retrieve(None)
            self.assertEqual(retriever.retrieve("и в на"), [])
            self.assertEqual(calls, len(self.requests))

    def test_comparison_reuses_document_vectors_and_calls_each_query_once(self):
        cases = [{"id": "a", "query": "Кровля"}, {"id": "b", "query": "Фотография"}]
        report = compare(self.kb, cases, ["bm25", "embeddings", "hybrid"], self.root / "cache")
        self.assertEqual(os.environ["RETRIEVAL_MODE"], "embeddings")
        self.assertEqual(len(self.requests), 5)  # one document batch + 2 queries per network mode
        for result in report.values():
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["summary"]["search_samples"], 2)
            self.assertIsNone(result["summary"]["hit_at_6"])
            self.assertTrue(result["details"][0]["results"][0]["text"])

    def test_comparison_reports_api_failure_without_fake_metrics(self):
        self.respond = lambda _: httpx.Response(401)
        cases = [{"id": "a", "query": "Кровля"}]
        report = compare(self.kb, cases, ["bm25", "embeddings", "hybrid"], None)
        self.assertEqual(report["bm25"]["status"], "ok")
        for mode in ("embeddings", "hybrid"):
            self.assertEqual(report[mode]["status"], "error")
            self.assertNotIn("summary", report[mode])

    def test_parallel_semantic_queries(self):
        retriever = build_retriever(self.kb)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(retriever.retrieve, ["Фотография"] * 12))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertIn("Фотография", results[0][0].title)


if __name__ == "__main__":
    unittest.main()
