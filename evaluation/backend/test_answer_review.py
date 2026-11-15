"""Selective answer repair contract with deterministic invented responses."""
import json
import unittest
from unittest.mock import patch

from consultant import Consultant, _review_reason
from contracts import ClientBrief, KnowledgeChunk
from llm_client import ModelUnavailable


CHUNK = KnowledgeChunk("doc", "invented.md", "Макет", "Макет читает бумагу. Стоимость тестового макета 250 руб.")


class Retriever:
    def retrieve(self, query, top_k=6):
        return [CHUNK]


def payload(answer, ids=None, **extra):
    return json.dumps({"answer": answer, "used_chunk_ids": ids if ids is not None else ["C1"],
                       "lead_offer": False, **extra})


class ReviewTests(unittest.TestCase):
    def setUp(self):
        flag = patch.dict("os.environ", {"ANSWER_REVIEW": "1"})
        flag.start()
        self.addCleanup(flag.stop)

    def core(self, responses):
        calls = []
        def generate(messages):
            calls.append(messages)
            result = responses[len(calls)-1]
            if isinstance(result, Exception):
                raise result
            return result
        return Consultant(Retriever(), generate, competition=True), calls

    def respond(self, core, message="Какие возможности у макета?"):
        return core.respond(message=message, history=[], brief=ClientBrief())

    def test_new_number_is_repaired_once(self):
        core, calls = self.core([payload("Экономия 91%."), payload("Числовая экономия в описании не указана.")])
        result = self.respond(core)
        self.assertNotIn("91", result.answer)
        self.assertEqual(len(calls), 2)
        data = json.loads(calls[1][1]["content"])
        self.assertIn("91", data["draft"])
        self.assertEqual(data["knowledge_chunks"][0]["text"], CHUNK.text)

    def test_valid_numbers_and_ordinary_answers_do_not_need_review(self):
        for answer in ("Макет читает бумагу.", "Стоимость тестового макета 250 руб."):
            core, calls = self.core([payload(answer)])
            self.assertEqual(self.respond(core).answer, answer)
            self.assertEqual(len(calls), 1)

    def test_citation_case_or_spacing_does_not_lose_grounded_answer(self):
        for identifier in (" c1 ", " C1", " doc "):
            core, calls = self.core([payload("Макет читает бумагу.", [identifier])])
            result = self.respond(core)
            self.assertEqual(result.answer, "Макет читает бумагу.")
            self.assertEqual(result.sources[0].chunk_id, "doc")
            self.assertEqual(len(calls), 1)

    def test_partial_or_invented_citation_does_not_match(self):
        core, calls = self.core([payload("Макет бесплатен.", ["C1 other"]), ModelUnavailable("timeout")])
        result = self.respond(core)
        self.assertNotIn("бесплатен", result.answer)
        self.assertFalse(result.sources)

    def test_explicit_guarantee_reviewed_even_with_existing_number(self):
        core, calls = self.core([payload("Гарантируем стоимость 250 руб."), payload("В описании указана стоимость тестового макета 250 руб.")])
        self.assertNotIn("Гарантируем", self.respond(core).answer)
        self.assertEqual(len(calls), 2)

    def test_missing_citation_can_be_recovered_against_actual_chunks(self):
        core, calls = self.core([payload("Макет читает бумагу.", []), payload("Макет читает бумагу.")])
        result = self.respond(core)
        self.assertEqual(result.answer, "Макет читает бумагу.")
        self.assertEqual(result.sources[0].chunk_id, "doc")
        self.assertEqual(len(calls), 2)

    def test_unavailable_or_malformed_reviewer_does_not_lose_first_response(self):
        for failure in (ModelUnavailable("timeout"), "{broken", payload("Произвольное обещание", ["C99"])):
            core, calls = self.core([payload("Гарантируем стоимость 250 руб."), failure])
            result = self.respond(core)
            self.assertIn("250", result.answer)
            self.assertEqual(len(calls), 2)

    def test_failed_review_cannot_bypass_existing_unknown_source_guard(self):
        core, calls = self.core([payload("Макет бесплатен.", ["C99"]), ModelUnavailable("timeout")])
        result = self.respond(core)
        self.assertNotIn("бесплатен", result.answer)
        self.assertFalse(result.sources)

    def test_reviewer_does_not_overwrite_client_brief_or_contact_decision(self):
        brief = {"object": "склад", "problem": "бумага", "known_parameters": [],
                 "recommended_products": [], "missing_fields": []}
        core, calls = self.core([payload("Экономия 91%.", brief=brief),
                                  payload("Экономия не указана.", brief={**brief, "object": "завод"}, lead_offer=True)])
        result = self.respond(core)
        self.assertEqual(result.brief.object, "склад")
        self.assertFalse(result.lead_offer)

    def test_safe_questions_and_direct_handoff_without_product_claims_skip_review(self):
        for answer, message in (("Сколько приборов у вас?", "Нужна помощь"),
                                ("Оставьте контакты", "Перезвоните мне")):
            core, calls = self.core([payload(answer, [])])
            self.respond(core, message)
            self.assertEqual(len(calls), 1)

    def test_review_can_be_disabled_for_a_control_run(self):
        core, calls = self.core([payload("Экономия 91%.")])
        with patch.dict("os.environ", {"ANSWER_REVIEW": "0"}):
            self.assertIn("91", self.respond(core).answer)
        self.assertEqual(len(calls), 1)

    def test_units_or_decimal_formatting_do_not_act_as_automatic_rejection(self):
        parsed = {"answer": "Параметр 250,0.", "used_chunk_ids": ["C1"]}
        self.assertIsNone(_review_reason(parsed, {"C1": "doc"}, [CHUNK], ""))
        parsed["answer"] = "Параметр 0,25 тыс. руб."
        self.assertEqual(_review_reason(parsed, {"C1": "doc"}, [CHUNK], ""), "new_numbers")
        # A conversion is a reason to ask the model, never an automatic refusal.


if __name__ == "__main__":
    unittest.main()
