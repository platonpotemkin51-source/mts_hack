"""Regression checks for manager handoff, including older stored sessions."""
import json
import tempfile
import unittest
from pathlib import Path

from backend.app.storage import Store, StoreError
from consultant import Consultant
from contracts import ChatMessage, ClientBrief, ConsultationResult


class HandoffTests(unittest.TestCase):
    def test_direct_handoff_without_product_source_does_not_become_refusal(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return []
        raw = json.dumps({"answer": "Заявка отправлена, цена нулевая.",
                          "used_chunk_ids": [], "lead_offer": True})
        core = Consultant(Retriever(), lambda _: raw, competition=True)
        result = core.respond(message="Хочу связаться со специалистом. +70000000000",
                              history=[], brief=ClientBrief())
        self.assertIn("номер указан", result.answer)
        self.assertNotIn("не нашлось", result.answer)
        self.assertNotIn("отправлена", result.answer)
        self.assertNotIn("нулевая", result.answer)

    def test_refusal_of_lead_overrides_model_with_phone_in_history(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return []
        raw = json.dumps({"answer": "Оставьте заявку", "used_chunk_ids": [], "lead_offer": True})
        core = Consultant(Retriever(), lambda _: raw, competition=True)
        history = [ChatMessage("user", "Мой вымышленный номер +70000000000")]
        for message in ("Не хочу оставить заявку.", "Не нужно перезвонить.", "Не готов связываться со специалистом."):
            with self.subTest(message=message):
                result = core.respond(message=message, history=history, brief=ClientBrief())
                self.assertFalse(result.lead_offer)
                self.assertNotIn("ФИО", result.answer)

    def test_summary_survives_parser_and_replaces_previous_summary(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return []
        raw = json.dumps({"answer": "Сколько точек учёта у вас?", "used_chunk_ids": [],
                          "manager_evidence": ["12 приборов"],
                          "lead_offer": False, "brief": {
                              "object": "склад", "problem": "учёт", "known_parameters": ["12 приборов"],
                              "recommended_products": [], "missing_fields": [],
                              "manager_summary": "На складе 12 приборов, требуется учёт по арендаторам."}})
        result = Consultant(Retriever(), lambda _: raw).respond(
            message="Уточню: 12 приборов", history=[],
            brief=ClientBrief(manager_summary="На складе 20 приборов."))
        self.assertIn("12", result.brief.manager_summary)
        self.assertNotIn("20", result.brief.manager_summary)
        result = Consultant(Retriever(), lambda _: raw, competition=True).respond(
            message="Хочу оставить заявку: 12 приборов", history=[], brief=ClientBrief())
        self.assertIn("Суть обращения для специалиста", result.answer)
        self.assertIn("12 приборов", result.answer)

    def test_unverified_summary_and_evidence_cannot_bypass_source_guard(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return []
        claim = "Наш продукт бесплатен и гарантирует экономию 50%."
        raw = json.dumps({"answer": "Какую задачу вы хотите решить?", "used_chunk_ids": [],
                          "lead_offer": False, "brief": {"manager_summary": claim},
                          "manager_evidence": [claim]})
        result = Consultant(Retriever(), lambda _: raw, competition=True).respond(
            message="Хочу оставить заявку", history=[], brief=ClientBrief())
        self.assertNotIn("50%", result.answer)
        self.assertIsNone(result.brief.manager_summary)

    def test_old_exact_quote_is_excluded_after_explicit_correction(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return []
        raw = json.dumps({"answer": "Уточните задачу", "used_chunk_ids": [], "lead_offer": False,
                          "manager_evidence": ["На складе 12 приборов."]})
        result = Consultant(Retriever(), lambda _: raw, competition=True).respond(
            message="Исправляю: на складе 20 приборов. Хочу оставить заявку.",
            history=[ChatMessage("user", "На складе 12 приборов.")], brief=ClientBrief())
        self.assertNotIn("12 приборов", result.answer)
        self.assertIn("20 приборов", result.answer)
        result = Consultant(Retriever(), lambda _: raw, competition=True).respond(
            message="Хочу оставить заявку",
            history=[ChatMessage("user", "На складе 12 приборов."),
                     ChatMessage("user", "Исправляю: на складе 20 приборов.")], brief=ClientBrief())
        self.assertNotIn("12 приборов", result.answer)
        self.assertIn("20 приборов", result.answer)

    def test_saved_lead_is_snapshot_and_readable_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.db"
            store = Store(path)
            sid, _, _, _ = store.begin_chat("handoff-request", None, "Раньше было 20 приборов. Что такое солнечные панели?")
            brief = ClientBrief(object="склад", problem="учёт", known_parameters=["12 приборов"],
                                manager_summary="Нужен раздельный учёт на складе с 12 приборами.")
            store.complete_chat("handoff-request", sid,
                                ConsultationResult("Уточнено", [], brief), False)
            lead = store.create_lead(sid, "Тестовый Клиент", "+70000000000", "Вымышленная компания")
            self.assertIn("12 приборами", lead["summary"])
            self.assertNotIn("20", lead["summary"])
            self.assertNotIn("солнечные", lead["summary"])
            self.assertIn("+70000000000", lead["summary"])
            self.assertEqual(Store(path).session_detail(sid)["lead"], lead)
            store.begin_chat("handoff-later", sid, "Другой вопрос")
            store.complete_chat("handoff-later", sid,
                                ConsultationResult("Ответ", [], ClientBrief(object="офис")), False)
            self.assertEqual(store.session_detail(sid)["lead"], lead)

    def test_legacy_brief_has_readable_fallback_without_raw_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "old.db")
            sid, _, _, _ = store.begin_chat("old-request", None, "Посторонний вопрос")
            store.complete_chat("old-request", sid,
                                ConsultationResult("Ответ", [], ClientBrief(object="офис", problem="учёт")), False)
            with store.connect() as db:
                db.execute("UPDATE sessions SET brief_json=? WHERE session_id=?",
                           (json.dumps({"object": "офис", "problem": "учёт"}), sid))
            lead = store.create_lead(sid, "Тест", "+70000000000", "Макет")
            self.assertIn("Объект: офис", lead["summary"])
            self.assertNotIn("Посторонний", lead["summary"])

    def test_lead_waits_for_accepted_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "pending.db")
            sid, _, _, _ = store.begin_chat("pending-first", None, "12 приборов")
            store.complete_chat("pending-first", sid,
                                ConsultationResult("Ответ", [], ClientBrief(manager_summary="12 приборов")), False)
            store.begin_chat("pending-correction", sid, "Исправляю: 20 приборов")
            with self.assertRaises(StoreError) as caught:
                store.create_lead(sid, "Тест", "+70000000000", "Макет")
            self.assertEqual(caught.exception.code, "request_pending")
            store.complete_chat("pending-correction", sid,
                                ConsultationResult("Ответ", [], ClientBrief(manager_summary="20 приборов")), False)
            lead = store.create_lead(sid, "Тест", "+70000000000", "Макет")
            self.assertIn("20 приборов", lead["summary"])
            self.assertNotIn("12 приборов", lead["summary"])
