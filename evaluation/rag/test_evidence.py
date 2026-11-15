"""Evidence completion on invented products; no model or private text."""
import json
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.app.main import create_app
from consultant import Consultant
from contracts import ClientBrief, KnowledgeChunk
from retrieval import BM25Retriever


def corpus():
    source = "20-products/paper.md"
    chunks = [
        KnowledgeChunk("intro", source, "Бумажный макет / Что это", "Бумажный макет: сканирование листов."),
        KnowledgeChunk("answer", source, "Бумажный макет / Возможности", "Сканирование сканирование листов доступно."),
        KnowledgeChunk("scope", source, "Бумажный макет / Целевой диапазон", "Только листы размера макета."),
        KnowledgeChunk("limit", "00-reference/scope.md", "Область применения / Бумажный макет", "Требуется проверка бумаги."),
        KnowledgeChunk("inputs", "00-reference/inputs.md", "Расчёт / Какие сведения нужны / Для бумажного макета", "Нужны плотность и размер бумаги."),
        KnowledgeChunk("other", "00-reference/scope.md", "Область применения / Стеклянный корабль", "Запрещён монтаж в макете."),
    ]
    return [replace(c, text=c.title + "\n\n" + c.text) for c in chunks]


def model(messages):
    supplied = json.loads(messages[1]["content"])
    return json.dumps({"answer": "Вымышленный ответ.", "brief": {
        "object": None, "problem": None, "known_parameters": [],
        "recommended_products": [], "missing_fields": []},
        "lead_offer": False, "used_chunk_ids": [supplied["knowledge_chunks"][0]["chunk_id"]]})


class EvidenceTests(unittest.TestCase):
    def test_missing_inputs_limits_and_scope_added_without_replacing_hits(self):
        r = BM25Retriever(corpus())
        before = r.retrieve_with_overviews("Сканирование: какие данные нужны для оценки?", 1)
        after = r.retrieve_with_evidence("Сканирование: какие данные нужны для оценки?", 1)
        self.assertEqual(after[:len(before)], before)
        self.assertEqual({c.chunk_id for c in after[len(before):]}, {"inputs", "limit", "scope"})
        self.assertNotIn("other", {c.chunk_id for c in after})
        self.assertTrue(all(c.score == 0 for c in after[len(before):]))

    def test_no_duplicates_or_inputs_without_question_about_data(self):
        r = BM25Retriever(corpus())
        results = r.retrieve_with_evidence("сканирование", 2)
        self.assertNotIn("inputs", {c.chunk_id for c in results})
        self.assertEqual(len(results), len({c.chunk_id for c in results}))

    def test_limits_not_added_twice(self):
        r = BM25Retriever(corpus())
        result = r.retrieve_with_evidence("сканирование проверка бумаги", 4)
        self.assertEqual(len(result), len({c.chunk_id for c in result}))

    def test_empty_unrelated_and_validation(self):
        r = BM25Retriever(corpus())
        self.assertEqual(r.retrieve_with_evidence(""), [])
        self.assertEqual(r.retrieve_with_evidence("quuxyz"), [])
        with self.assertRaises(ValueError):
            r.retrieve_with_evidence("сканирование", 0)

    def test_budget_never_truncates_a_condition(self):
        chunks = corpus()
        chunks[3] = KnowledgeChunk("limit", chunks[3].source, chunks[3].title, "x" * 4001)
        r = BM25Retriever(chunks)
        before = r.retrieve_with_overviews("сканирование какие данные нужны", 1)
        after = r.retrieve_with_evidence("сканирование какие данные нужны", 1)
        additions = after[len(before):]
        self.assertLessEqual(len(additions), 3)
        self.assertLessEqual(sum(len(c.text) for c in additions), 4000)
        self.assertNotIn("limit", {c.chunk_id for c in additions})

    def test_ambiguous_reference_heading_is_not_assigned_to_a_product(self):
        chunks = corpus() + [KnowledgeChunk("intro-b", "20-products/b.md", "Бумажный корабль / Что это", "Корабль.")]
        chunks += [KnowledgeChunk("ambiguous", "00-reference/scope.md", "Область применения / Бумажный", "Общее ограничение.")]
        self.assertNotIn("ambiguous", BM25Retriever(chunks)._evidence_products)

    def test_consultant_can_switch_completion_off(self):
        captured = []
        def capture(messages):
            captured.append(json.loads(messages[1]["content"])["knowledge_chunks"])
            return model(messages)
        core = Consultant(BM25Retriever(corpus()), capture, context_expansion=True)
        with patch.dict("os.environ", {"TOP_K": "1", "EVIDENCE_COMPLETION": "1"}):
            core.respond(message="Сканирование: какие данные нужны?", history=[], brief=ClientBrief())
        with patch.dict("os.environ", {"TOP_K": "1", "EVIDENCE_COMPLETION": "0"}):
            core.respond(message="Сканирование: какие данные нужны?", history=[], brief=ClientBrief())
        self.assertEqual(len(captured[0]), 4)
        self.assertEqual(len(captured[1]), 1)

    def test_web_uses_the_same_completed_context(self):
        captured = []
        def capture(messages):
            captured.append(json.loads(messages[1]["content"])["knowledge_chunks"])
            return model(messages)
        with tempfile.TemporaryDirectory() as td, patch.dict("os.environ", {
            "TOP_K": "1", "EVIDENCE_COMPLETION": "1", "CONTEXT_EXPANSION": "1",
        }):
            app = create_app(retriever=BM25Retriever(corpus()), generate_fn=capture,
                             db_path=Path(td) / "invented.sqlite3")
            with TestClient(app) as client:
                response = client.post("/api/chat", json={"message": "Сканирование: какие данные нужны?",
                                                          "request_id": "evidence-test"})
                self.assertEqual(response.status_code, 200)
            self.assertEqual(len(captured[0]), 4)


if __name__ == "__main__":
    unittest.main()
