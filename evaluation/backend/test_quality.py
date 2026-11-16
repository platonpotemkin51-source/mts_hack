"""Focused regression tests for answer recovery and source grounding."""

import asyncio
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from backend.app.main import create_app
from energytool.consultant import Consultant, _search_query
from energytool.contracts import ChatMessage, ClientBrief, KnowledgeChunk
from energytool.llm_client import ModelUnavailable, generate
from energytool.retrieval import BM25Retriever


CHUNK = KnowledgeChunk("real-chunk", "invented.md", "Вымышленный продукт",
                       "Вымышленный продукт собирает показания.", 1.0)


class Retriever:
    def retrieve(self, _query, top_k=6):
        return [CHUNK]


def payload(answer, sources=None, brief=None, lead=False):
    return json.dumps({"answer": answer, "brief": brief if brief is not None else {
        "object": None, "problem": None, "known_parameters": [],
        "recommended_products": [], "missing_fields": []},
        "lead_offer": lead, "used_chunk_ids": sources if sources is not None else []},
        ensure_ascii=False)


class QualityTests(unittest.TestCase):
    def test_two_product_context_keeps_ranked_limits_and_adds_overviews(self):
        chunks = [
            KnowledgeChunk("a-intro", "20-products/a.md", "A / Что это", "Описание А.", 0.0),
            KnowledgeChunk("a-one", "20-products/a.md", "A / Кому подходит", "учёт приборы показания", 0.0),
            KnowledgeChunk("a-two", "20-products/a.md", "A / Ограничение", "учёт приборы только после проверки", 0.0),
            KnowledgeChunk("b-intro", "20-products/b.md", "B / Что это", "Описание Б.", 0.0),
            KnowledgeChunk("b-one", "20-products/b.md", "B / Кому подходит", "показания приборы фото", 0.0),
            KnowledgeChunk("b-two", "20-products/b.md", "B / Ограничение", "показания фото после проверки", 0.0),
        ]
        result = BM25Retriever(chunks).retrieve_with_overviews("учёт приборы показания фото", top_k=4)
        self.assertEqual(len(result), 6)
        self.assertEqual({x.chunk_id for x in result[-2:]}, {"a-intro", "b-intro"})
        self.assertIn("a-two", {x.chunk_id for x in result[:4]})

    def test_dialogue_query_keeps_goal_and_resets_on_topic_change(self):
        history = [ChatMessage("user", "У нас бизнес-центр, нужен учёт по арендаторам"),
                   ChatMessage("assistant", "Как собираете показания?"),
                   ChatMessage("user", "Есть счётчики, часть показаний приходит по фото")]
        query = _search_query("Можете объяснить, с чего нам начать работу с системой?",
                              history, ClientBrief())
        self.assertIn("бизнес-центр", query)
        self.assertIn("арендаторам", query)
        switched = _search_query("А теперь другой вопрос: подходит ли оптовый рынок?",
                                 history, ClientBrief(object="бизнес-центр"))
        self.assertNotIn("арендаторам", switched)
        self.assertNotIn("бизнес-центр", switched)

    def test_unsourced_clarification_survives_but_claim_does_not(self):
        events = []
        question = Consultant(Retriever(), lambda _: payload("Сколько точек учёта у вас?"),
                              competition=True, diagnostic_fn=lambda *x: events.append(x)).respond(
                                  message="Нужна помощь", history=[], brief=ClientBrief())
        self.assertTrue(question.answer.endswith("Сколько точек учёта у вас?"))
        self.assertIn(("safe_unsourced_answer", None), events)
        claim = Consultant(Retriever(), lambda _: payload("Платформа поддерживает любую модель.", ["C99"]),
                           competition=True, diagnostic_fn=lambda *x: events.append(x)).respond(
                               message="Нужна помощь", history=[], brief=ClientBrief())
        self.assertIn("не нашлось", claim.answer)
        self.assertIn(("unknown_source", None), events)
        for unsafe in (
            "В предоставленной базе нет данных о сроках. Наш продукт полностью бесплатен.",
            "Какие объекты вы хотите подключить к полностью бесплатной системе?",
            "В доступной базе нет данных о цене; система работает без ограничений.",
            "Какие объекты вы хотите подключить к системе, которая полностью заменяет персонал?",
        ):
            result = Consultant(Retriever(), lambda _, value=unsafe: payload(value),
                                competition=True).respond(
                                    message="Нужна помощь", history=[], brief=ClientBrief())
            self.assertNotIn("бесплатн", result.answer)
            self.assertNotIn("без ограничений", result.answer)
            self.assertNotIn("заменяет персонал", result.answer)
            self.assertFalse(result.sources)

    def test_auxiliary_error_and_markdown_wrapper_keep_grounded_answer(self):
        wrapped = "Проверка:\n```\n" + payload("Вымышленный продукт собирает показания.",
                                              ["C1"], brief={"object": 99}) + "\n```"
        result = Consultant(Retriever(), lambda _: wrapped, competition=True).respond(
            message="Как собрать показания?", history=[], brief=ClientBrief())
        self.assertEqual(result.answer, "Вымышленный продукт собирает показания.")
        self.assertEqual([c.chunk_id for c in result.sources], ["real-chunk"])

    def test_invalid_json_is_retried_once_for_competition(self):
        calls = []
        def model(_messages):
            calls.append(1)
            return "{broken" if len(calls) == 1 else payload("Сколько точек учёта у вас?")
        result = Consultant(Retriever(), model, competition=True).respond(
            message="Нужна помощь", history=[], brief=ClientBrief())
        self.assertEqual(len(calls), 2)
        self.assertTrue(result.answer.endswith("Сколько точек учёта у вас?"))

    def test_first_direct_request_can_offer_lead(self):
        result = Consultant(Retriever(), lambda _: payload("Свяжемся со специалистом.", ["C1"], lead=False)).respond(
            message="Хочу сразу оставить заявку и связаться со специалистом", history=[], brief=ClientBrief())
        self.assertTrue(result.lead_offer)

    def test_refusal_is_not_treated_as_direct_lead_request(self):
        consultant = Consultant(Retriever(), lambda _: payload("Уточните задачу.", lead=False))
        for message in (
            "Не хочу оставить заявку. Продолжим без контактов.",
            "Я не готов оставить заявку.",
            "Не нужно перезвонить.",
            "Как настроить счетчик? Не хочу связаться со специалистом.",
        ):
            with self.subTest(message=message):
                result = consultant.respond(message=message, history=[], brief=ClientBrief())
                self.assertFalse(result.lead_offer)

    def test_length_finish_reason_retries_with_more_tokens(self):
        requested = []
        def handler(request):
            requested.append(json.loads(request.content)["max_tokens"])
            if len(requested) == 1:
                return httpx.Response(200, json={"choices": [{"message": {"content": "{broken"},
                                                               "finish_reason": "length"}],
                                                 "usage": {"completion_tokens": 2048}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ready"},
                                                           "finish_reason": "stop"}]})
        events = []
        with patch.dict("os.environ", {"BASE_URL": "http://proxy.test", "LLM_API_KEY": "test",
                                    "LLM_MODEL": "test", "MAX_TOKENS": "2048"}, clear=True):
            answer = generate([{"role": "user", "content": "Hi"}],
                              transport=httpx.MockTransport(handler), deadline=time.monotonic() + 100,
                              sleep=lambda _: None, diagnostic_fn=lambda *x: events.append(x))
        self.assertEqual(answer, "ready")
        self.assertEqual(requested, [2048, 4096])
        self.assertIn(("length_truncation", None), events)

    def test_competition_request_obeys_wall_clock_deadline(self):
        async def slow_handler(_request):
            await asyncio.sleep(2.0)
            return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})
        with patch.dict("os.environ", {"BASE_URL": "http://proxy.test", "LLM_API_KEY": "test",
                                    "LLM_MODEL": "test"}, clear=True):
            started = time.monotonic()
            with self.assertRaises(ModelUnavailable):
                generate([{"role": "user", "content": "Hi"}],
                         transport=httpx.MockTransport(slow_handler),
                         deadline=started + 2.4, sleep=lambda _: None)
            self.assertLess(time.monotonic() - started, 1.5)

    def test_malformed_proxy_message_fails_request_without_stuck_pending(self):
        transport = httpx.MockTransport(lambda _request: httpx.Response(
            200, json={"choices": [{"message": "malformed"}]}))
        with tempfile.TemporaryDirectory() as td, patch.dict("os.environ", {
            "BASE_URL": "http://proxy.test", "LLM_API_KEY": "test", "LLM_MODEL": "test",
        }, clear=True):
            app = create_app(retriever=Retriever(),
                             generate_fn=lambda messages: generate(messages, transport=transport),
                             db_path=Path(td) / "sessions.sqlite3")
            with TestClient(app, raise_server_exceptions=False) as client:
                body = {"session_id": None, "request_id": "malformed-proxy-message",
                        "message": "Вопрос про вымышленный продукт"}
                first = client.post("/api/chat", json=body)
                second = client.post("/api/chat", json=body)
                self.assertEqual(first.status_code, 503)
                self.assertEqual(second.status_code, 503)


if __name__ == "__main__":
    unittest.main()
