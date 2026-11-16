"""Behavior checks on invented content and a deterministic local model stub."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from backend.app.main import create_app
from energytool.consultant import Consultant, answer_test_case
from energytool.contracts import ChatMessage, ClientBrief, KnowledgeChunk
from energytool.llm_client import ModelUnavailable, generate


CHUNK = KnowledgeChunk("invented-1", "fiction.md", "Макет", "Вымышленный макет хранится в шкафу.", 1.0)


class FakeRetriever:
    def __init__(self, found=True):
        self.found = found
        self.queries = []

    def retrieve(self, query, top_k=6):
        self.queries.append(query)
        return [CHUNK] if self.found else []


def fake_model(messages):
    data = json.loads(messages[1]["content"])
    message = data["current_message"]
    old = data["current_brief"]
    obj = "офис" if "офис" in message else "склад" if "склад" in message else old["object"]
    return json.dumps({
        "answer": "По вымышленной базе макет хранится в шкафу. Что уточнить?",
        "brief": {"object": obj, "problem": "хранение макета",
                  "known_parameters": ["бумажный макет"],
                  "recommended_products": ["Вымышленный макет"],
                  "missing_fields": ["размер"]},
        "lead_offer": True, "used_chunk_ids": ["invented-1"],
    }, ensure_ascii=False)


class CoreTests(unittest.TestCase):
    def test_history_query_correction_and_source(self):
        retriever = FakeRetriever()
        consultant = Consultant(retriever, fake_model)
        first = consultant.respond(message="Для склада нужен макет", history=[], brief=ClientBrief())
        self.assertEqual(first.brief.object, "склад")
        self.assertFalse(first.lead_offer)
        second = consultant.respond(message="Уточню: это офис", history=[ChatMessage("user", "Для склада нужен макет"),
                                                                     ChatMessage("assistant", first.answer)],
                                    brief=first.brief)
        self.assertEqual(second.brief.object, "офис")
        self.assertIn("Для склада нужен макет", retriever.queries[-1])
        self.assertEqual([x.chunk_id for x in second.sources], ["invented-1"])

    def test_no_sources_cannot_recommend(self):
        result = Consultant(FakeRetriever(False), fake_model).respond(
            message="Вопрос про макет", history=[], brief=ClientBrief())
        self.assertEqual(result.sources, [])
        self.assertEqual(result.brief.recommended_products, [])
        self.assertIn("не нашлось", result.answer)

    def test_technical_setup_is_out_of_scope(self):
        def forbidden(_messages):
            raise AssertionError("model must not be called")
        result = Consultant(FakeRetriever(), forbidden).respond(
            message="Как настроить счетчик?", history=[], brief=ClientBrief())
        self.assertIn("не даю инструкции", result.answer)
        documentation = Consultant(FakeRetriever(), forbidden).respond(
            message="Покажи техническую документацию по прибору", history=[], brief=ClientBrief())
        self.assertIn("не даю инструкции", documentation.answer)

    def test_contact_refusal_is_durable_and_does_not_end_consultation(self):
        consultant = Consultant(FakeRetriever(), fake_model)
        history = [ChatMessage("user", "Расскажите про вымышленный макет"),
                   ChatMessage("assistant", "Что уточнить?")]
        declined = consultant.respond(
            message="Телефон пока оставлять не хочу. Продолжим консультацию.",
            history=history, brief=ClientBrief())
        self.assertFalse(declined.lead_offer)
        self.assertTrue(declined.answer)
        later = consultant.respond(
            message="А как его хранить?",
            history=history + [ChatMessage("user", "Телефон пока оставлять не хочу."),
                               ChatMessage("assistant", declined.answer)] +
                    [ChatMessage("user", "Ещё вопрос"), ChatMessage("assistant", "Ответ")]*3,
            brief=declined.brief)
        self.assertFalse(later.lead_offer)

    def test_dialogue_case_uses_last_user_turn(self):
        seen = []
        class CaptureConsultant:
            def respond(self, **kwargs):
                seen.append(kwargs)
                return type("Result", (), {"answer": "готово"})()
        answer = answer_test_case("Пользователь: Первый вопрос. Ассистент: Уточните. Пользователь: Второй вопрос?",
                                  consultant=CaptureConsultant())
        self.assertEqual(answer, "готово")
        self.assertEqual(seen[0]["message"], "Второй вопрос?")
        self.assertEqual(len(seen[0]["history"]), 2)

    def test_llm_retries_and_hides_secret(self):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) < 3:
                return httpx.Response(503, text="sensitive proxy text")
            return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://proxy.test", "LLM_API_KEY": "secret",
                                    "LLM_MODEL": "local-model"}):
            value = generate([{"role": "user", "content": "Hi"}],
                             transport=httpx.MockTransport(handler), sleep=lambda _: None)
            self.assertEqual(value, "answer")
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0].headers["Authorization"], "Bearer secret")
            with self.assertRaises(ModelUnavailable) as raised:
                generate([], transport=httpx.MockTransport(lambda _r: httpx.Response(401, text="secret")),
                         sleep=lambda _: None)
            self.assertNotIn("secret", str(raised.exception))
            with self.assertRaises(ModelUnavailable):
                generate([], transport=httpx.MockTransport(handler),
                         deadline=time.monotonic() - 1, sleep=lambda _: None)

    def test_competition_proxy_alias_and_token_budget(self):
        captured = []
        def handler(request):
            captured.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})
        with patch.dict("os.environ", {"BASE_URL": "http://proxy.test/v1", "LLM_API_KEY": "test-key",
                                    "LLM_MODEL": "qwen", "MAX_TOKENS": ""}, clear=True):
            self.assertEqual(generate([{"role": "user", "content": "Hi"}],
                                      transport=httpx.MockTransport(handler),
                                      deadline=time.monotonic() + 120), "answer")
        self.assertEqual(str(captured[0].url), "http://proxy.test/v1/chat/completions")
        self.assertEqual(captured[0].extensions["timeout"]["read"], 90.0)
        self.assertEqual(json.loads(captured[0].content)["max_tokens"], 2048)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "sessions.sqlite3"
        self.app = create_app(retriever=FakeRetriever(), generate_fn=fake_model,
                              db_path=self.db_path)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def chat(self, message="Для склада нужен макет", session_id=None, request_id="request-one"):
        return self.client.post("/api/chat", json={"session_id": session_id,
                                                    "message": message,
                                                    "request_id": request_id, "debug": True})

    def test_chat_archive_isolation_and_persistence(self):
        self.assertEqual(self.client.get("/api/ready").status_code, 200)
        first = self.chat().json()
        self.assertEqual(first["brief"]["object"], "склад")
        self.assertEqual(len(first["sources"]), 1)
        self.assertGreaterEqual(first["debug"]["total_ms"], 0)
        self.assertEqual(self.chat().json(), first)
        second = self.chat("Уточню: это офис", first["session_id"], "request-two").json()
        self.assertEqual(second["brief"]["object"], "офис")
        self.assertEqual(len(self.client.get("/api/sessions/" + first["session_id"]).json()["messages"]), 4)
        other = self.chat("Другой объект", request_id="request-three").json()
        self.assertNotEqual(other["session_id"], first["session_id"])
        self.assertEqual(len(self.client.get("/api/sessions/" + other["session_id"]).json()["messages"]), 2)
        with TestClient(create_app(retriever=FakeRetriever(), generate_fn=fake_model,
                                   db_path=self.db_path)) as restarted:
            self.assertEqual(len(restarted.get("/api/sessions").json()), 2)
            self.assertEqual(restarted.get("/api/sessions/" + first["session_id"]).json()["brief"]["object"], "офис")

    def test_lead_consent_validation_duplicate_and_close(self):
        sid = self.chat().json()["session_id"]
        body = {"session_id": sid, "full_name": "Тестовый Клиент", "phone": "+70000000000",
                "company": "Демо-компания", "consent": True}
        self.assertEqual(self.client.post("/api/leads", json={**body, "consent": False}).status_code, 422)
        self.assertEqual(self.client.post("/api/leads", json={**body, "phone": "abc"}).status_code, 422)
        lead = self.client.post("/api/leads", json=body)
        self.assertEqual(lead.status_code, 201)
        self.assertIn("склад", lead.json()["summary"])
        self.assertEqual(self.client.post("/api/leads", json=body).status_code, 409)
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/close").status_code, 200)
        self.assertEqual(self.chat("ещё", sid, "request-four").status_code, 409)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").json()["status"], "closed")

    def test_cancel_before_first_chat_and_request_reuse(self):
        self.assertEqual(self.client.post("/api/requests/request-lost/close").status_code, 204)
        self.assertEqual(self.chat(request_id="request-lost").status_code, 409)
        first = self.chat().json()
        self.assertEqual(self.chat("different", request_id="request-one").status_code, 409)
        self.assertEqual(self.client.post("/api/requests/request-one/close").status_code, 204)
        self.assertEqual(self.client.get(f"/api/sessions/{first['session_id']}").json()["status"], "closed")

    def test_model_failure_preserves_user_message_and_can_retry(self):
        calls = []
        def flaky(messages):
            calls.append(1)
            if len(calls) == 1:
                raise ModelUnavailable("offline")
            return fake_model(messages)
        with TestClient(create_app(retriever=FakeRetriever(), generate_fn=flaky,
                                   db_path=Path(self.temp.name) / "flaky.sqlite3")) as client:
            body = {"session_id": None, "message": "Для склада нужен макет", "request_id": "retry-key"}
            self.assertEqual(client.post("/api/chat", json=body).status_code, 503)
            self.assertEqual(len(client.get("/api/sessions").json()), 1)
            response = client.post("/api/chat", json=body)
            self.assertEqual(response.status_code, 200)
            detail = client.get("/api/sessions/" + response.json()["session_id"]).json()
            self.assertEqual([m["role"] for m in detail["messages"]], ["user", "assistant"])

    def test_second_request_cannot_overtake_first_pending_reply(self):
        entered, release = threading.Event(), threading.Event()
        def slow(messages):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
            return fake_model(messages)
        with TestClient(create_app(retriever=FakeRetriever(), generate_fn=slow,
                                   db_path=Path(self.temp.name) / "parallel.sqlite3")) as client:
            result = []
            thread = threading.Thread(target=lambda: result.append(client.post(
                "/api/chat", json={"session_id": None, "message": "Для склада нужен макет",
                                   "request_id": "parallel-first"})))
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                sid = client.get("/api/sessions").json()[0]["session_id"]
                overtaking = client.post("/api/chat", json={"session_id": sid,
                    "message": "Уточню: это офис", "request_id": "parallel-second"})
                self.assertEqual(overtaking.status_code, 409)
                self.assertEqual(overtaking.json()["error"]["code"], "request_pending")
            finally:
                release.set()
                thread.join(5)
            self.assertEqual(result[0].status_code, 200)
            detail = client.get("/api/sessions/" + sid).json()
            self.assertEqual([item["role"] for item in detail["messages"]], ["user", "assistant"])
            self.assertEqual(detail["brief"]["object"], "склад")

    def test_failed_old_request_cannot_overwrite_newer_brief(self):
        calls = []
        def fail_old_once(messages):
            calls.append(1)
            if len(calls) == 1:
                raise ModelUnavailable("synthetic failure")
            return fake_model(messages)
        with TestClient(create_app(retriever=FakeRetriever(), generate_fn=fail_old_once,
                                   db_path=Path(self.temp.name) / "stale.sqlite3")) as client:
            old = {"session_id": None, "message": "Для склада нужен макет",
                   "request_id": "stale-old"}
            self.assertEqual(client.post("/api/chat", json=old).status_code, 503)
            sid = client.get("/api/sessions").json()[0]["session_id"]
            newer = client.post("/api/chat", json={"session_id": sid,
                "message": "Уточню: это офис", "request_id": "stale-new"})
            self.assertEqual(newer.status_code, 200)
            stale = client.post("/api/chat", json=old)
            self.assertEqual(stale.status_code, 409)
            self.assertEqual(stale.json()["error"]["code"], "request_stale")
            self.assertEqual(client.get("/api/sessions/" + sid).json()["brief"]["object"], "офис")


if __name__ == "__main__":
    unittest.main()
